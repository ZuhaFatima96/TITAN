import tkinter as tk
from tkinter import ttk, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import random
import torch
import torch.nn as nn
import threading
from torch.distributions import Categorical
from collections import deque, defaultdict
from dataclasses import dataclass

# ------ Hyperparameters ------
PPO_EPOCHS = 10
CLIP_EPSILON = 0.1
GAMMA = 0.95
ENTROPY_COEF = 0.2
BATCH_SIZE = 256
BC_EPOCHS = 10
NUM_TASKS = 32
BUFFER_SIZE = 10_000
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ------ Data Classes ------
@dataclass
class TaskConfig:
    task_id: int
    arrival_time: float
    expected_duration: float
    deadline: float
    is_io_bound: bool = False
    is_foreground: bool = False

@dataclass 
class TaskState:
    remaining: float

@dataclass
class Task:
    config: TaskConfig
    state: TaskState

# ------ Scheduling Environment ------
class SchedulingEnv:
    def __init__(self, max_tasks=NUM_TASKS):
        self.max_tasks = max_tasks
        self.current_time = 0.0
        self.tasks = []
        self.reset()

    def reset(self):
        self.current_time = 0.0
        self.tasks = self._generate_tasks()
        return self._get_state()

    def _generate_tasks(self):
        tasks = []
        for i in range(NUM_TASKS):
            cfg = TaskConfig(
                task_id=i,
                arrival_time=random.uniform(0, 10),
                expected_duration=random.uniform(1, 20),
                deadline=random.uniform(15, 60),
                is_io_bound=random.choice([True, False]),
                is_foreground=random.choice([True, False])
            )
            tasks.append(Task(cfg, TaskState(cfg.expected_duration)))
        return tasks

    def _get_state(self):
        state = torch.zeros((self.max_tasks, 5), device=DEVICE)
        for i, t in enumerate(self.tasks):
            state[i] = torch.tensor([
                t.config.arrival_time,
                t.config.expected_duration,
                t.config.deadline,
                float(t.config.is_io_bound),
                float(t.config.is_foreground)
            ], device=DEVICE)
        return state

    def step(self, action):
        if action >= len(self.tasks):
            return self._get_state(), -10.0, True, {}
        t = self.tasks[action]
        dt = min(t.state.remaining, max(0.1, t.config.deadline - self.current_time))
        t.state.remaining -= dt
        self.current_time += dt
        r = (4.0 if (self.current_time <= t.config.deadline) else -5.0)
        r -= 0.1 * (self.current_time - t.config.arrival_time)
        r += 1.0 * dt
        r /= 1000.0
        if t.state.remaining <= 0:
            self.tasks.pop(action)
        done = (len(self.tasks)==0)
        return self._get_state(), r, done, {}

# ------ Models ------
class TaskPriorityTransformer(nn.Module):
    def __init__(self, input_dim=5, model_dim=128, num_heads=8, num_layers=4, max_tasks=NUM_TASKS):
        super().__init__()
        self.model_dim = model_dim
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.positional_encoding = nn.Parameter(torch.randn(1, max_tasks, model_dim))
        encoder = nn.TransformerEncoderLayer(d_model=model_dim, nhead=num_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder, num_layers=num_layers)
        self.output_layer = nn.Linear(model_dim, 1)

    def forward(self, x):
        x = self.input_proj(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.transformer(x)
        return self.output_layer(x).squeeze(-1)

class PPOPolicy(nn.Module):
    def __init__(self, sl_model):
        super().__init__()
        self.transformer = sl_model.transformer
        self.input_proj = sl_model.input_proj
        self.positional_encoding = sl_model.positional_encoding
        md = sl_model.model_dim
        self.actor = nn.Sequential(nn.Linear(md,128), nn.ReLU(), nn.Linear(128, NUM_TASKS))
        self.critic = nn.Sequential(nn.Linear(md,128), nn.ReLU(), nn.Linear(128, 1))
        self.optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        self.memory = deque(maxlen=BUFFER_SIZE)

    def forward(self, x):
        x = self.input_proj(x) + self.positional_encoding[:, :x.size(1), :]
        h = self.transformer(x)
        mask = (x[:,:,0]!=0).float().unsqueeze(-1)
        feats = (h * mask).mean(dim=1)
        return self.actor(feats), self.critic(feats).squeeze(-1)

# ------ Evaluation Functions ------
def edf_heuristic(state):
    mask = (state[:,0]!=0)
    dln = state[:,2].clone()
    dln[~mask] = float('inf')
    return torch.argmin(dln).item()

def infer_sl(sl, s):
    sl.eval()
    with torch.no_grad():
        lg = sl(s.unsqueeze(0))[0]
        lg[~(s[:,0]!=0)] = -1e9
    return torch.argmax(lg).item()

def infer_rl(rl, s):
    rl.eval()
    with torch.no_grad():
        lg,_ = rl(s.unsqueeze(0))
        lg[0,~(s[:,0]!=0)] = -1e9
    return Categorical(logits=lg).sample().item()

def fcfs_policy(env, state):
    arrivals = [t.config.arrival_time for t in env.tasks]
    return int(min(range(len(arrivals)), key=lambda i: arrivals[i]))

def sjf_policy(env, state):
    remains = [t.state.remaining for t in env.tasks]
    return int(min(range(len(remains)), key=lambda i: remains[i]))

def make_rr_policy():
    rr = {'idx': 0}
    def rr_policy(env, state):
        n = len(env.tasks)
        choice = rr['idx'] % n
        rr['idx'] += 1
        return choice
    return rr_policy

def evaluate_metrics(policy_fn, task_list):
    env = FixedTaskEnv(task_list)
    state = env.reset()
    all_ids = [t.config.task_id for t in task_list]
    arr_time = {t.config.task_id: t.config.arrival_time for t in task_list}
    deadline = {t.config.task_id: t.config.deadline for t in task_list}
    start_time = {tid: None for tid in all_ids}
    comp_time = {}
    remaining_ids = set(all_ids)

    while remaining_ids:
        a = policy_fn(env, state)
        tid = env.tasks[a].config.task_id
        if start_time[tid] is None:
            start_time[tid] = env.current_time
        prev_task_ids = set(t.config.task_id for t in env.tasks)
        state, _, done, _ = env.step(a)
        curr_task_ids = set(t.config.task_id for t in env.tasks)
        finished = prev_task_ids - curr_task_ids
        for finished_tid in finished:
            if finished_tid in remaining_ids:
                comp_time[finished_tid] = env.current_time
                remaining_ids.remove(finished_tid)
        if done:
            # Mark any remaining tasks as completed at current_time
            for tid_left in remaining_ids:
                comp_time[tid_left] = env.current_time
            break

    N = len(all_ids)
    tat = [comp_time[tid] - arr_time[tid] for tid in all_ids]
    wait = [start_time[tid] - arr_time[tid] for tid in all_ids]
    misses = sum(1 for tid in all_ids if comp_time[tid] > deadline[tid])
    sum_w = sum(wait)
    sum_w2 = sum(w*w for w in wait)
    fairness_waiting = (sum_w * sum_w) / (N * sum_w2) if sum_w2 > 0 else 1.0
    return {
        'avg_turnaround': sum(tat)/N,
        'avg_waiting': sum(wait)/N,
        'miss_rate': misses/N,
        'avg_response': sum(start_time[tid]-arr_time[tid] for tid in all_ids)/N,
        'fairness_waiting': fairness_waiting
    }

class FixedTaskEnv(SchedulingEnv):
    def __init__(self, task_list):
        self.original_tasks = task_list
        self.max_tasks = len(task_list)
        self.current_time = 0.0
        self.tasks = []

    def reset(self):
        self.current_time = 0.0
        self.tasks = [Task(TaskConfig(**vars(t.config)), TaskState(t.state.remaining)) 
                     for t in self.original_tasks]
        return self._get_state()


class SchedulerApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("TITAN Scheduler Benchmark")
        self.geometry("1200x800")
        self.configure(bg='#f0f0f0')
        
        # Model paths
        self.sl_path = ''
        self.rl_path = ''
        
        # Results storage
        self.results = defaultdict(list)
        self.current_episode = 0
        self.running = False
        
        # Create UI
        self.create_widgets()
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        
    def create_widgets(self):
        # Control Panel
        control_frame = ttk.LabelFrame(self, text="Model Configuration", padding=10)
        control_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Model file inputs
        ttk.Button(control_frame, text="Load SL Model", command=self.load_sl).grid(row=0, column=0, padx=5)
        ttk.Button(control_frame, text="Load RL Model", command=self.load_rl).grid(row=0, column=1, padx=5)
        
        # Episode control
        ttk.Label(control_frame, text="Episodes:").grid(row=0, column=2, padx=5)
        self.episode_spin = ttk.Spinbox(control_frame, from_=1, to=1000, width=5)
        self.episode_spin.grid(row=0, column=3, padx=5)
        self.episode_spin.set(100)
        
        self.run_btn = ttk.Button(control_frame, text="Run Evaluation", command=self.start_evaluation)
        self.run_btn.grid(row=0, column=4, padx=10)
        
        # Progress
        self.progress_label = ttk.Label(control_frame, text="Ready")
        self.progress_label.grid(row=0, column=5, padx=10)
        
        # Visualization Frame
        vis_frame = ttk.Frame(self)
        vis_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Dropdown for metric selection in line chart
        self.metric_options = ['avg_turnaround', 'avg_waiting', 'miss_rate', 'avg_response', 'fairness_waiting']
        self.metric_names = {
            'avg_turnaround': 'Turnaround',
            'avg_waiting': 'Waiting',
            'miss_rate': 'Miss Rate',
            'avg_response': 'Response',
            'fairness_waiting': "Jain's Fairness"
        }
        self.selected_metric = tk.StringVar(value=self.metric_options[0])
        metric_frame = ttk.Frame(vis_frame)
        metric_frame.pack(fill=tk.X, pady=2)
        ttk.Label(metric_frame, text="Line Chart Metric:").pack(side=tk.LEFT, padx=5)
        self.metric_dropdown = ttk.Combobox(metric_frame, textvariable=self.selected_metric, values=[self.metric_names[m] for m in self.metric_options], state='readonly', width=18)
        self.metric_dropdown.pack(side=tk.LEFT, padx=5)
        self.metric_dropdown.bind('<<ComboboxSelected>>', lambda e: self.update_visualization())

        # Matplotlib figures: Bar chart and Line chart
        self.fig, (self.ax, self.line_ax) = plt.subplots(2, 1, figsize=(10, 10), gridspec_kw={'height_ratios': [2, 3]})
        plt.tight_layout(pad=3.0)
        self.canvas = FigureCanvasTkAgg(self.fig, master=vis_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Results Table
        results_frame = ttk.LabelFrame(self, text="Latest Results", padding=10)
        results_frame.pack(fill=tk.X, padx=10, pady=5)
        self.results_table = ttk.Treeview(results_frame, columns=('Metric', 'EDF', 'SL', 'TITAN', 'FCFS', 'SJF', 'RR'), show='headings')
        for col in ['Metric', 'EDF', 'SL', 'TITAN', 'FCFS', 'SJF', 'RR']:
            self.results_table.heading(col, text=col)
            self.results_table.column(col, width=100)
        self.results_table.pack(fill=tk.X)

        # Footer
        self.footer = ttk.Label(self, text="TITAN Scheduler Benchmark © 2025 | Developed by Your Name", anchor='center', font=("Segoe UI", 10, "italic"))
        self.footer.pack(side=tk.BOTTOM, fill=tk.X, pady=5)

    def load_sl(self):
        self.sl_path = filedialog.askopenfilename(title="Select SL Model")
        
    def load_rl(self):
        self.rl_path = filedialog.askopenfilename(title="Select RL Model")

    def start_evaluation(self):
        if not self.sl_path or not self.rl_path:
            self.progress_label.config(text="Please select both model files!")
            return
            
        self.running = True
        self.current_episode = 0
        self.results.clear()
        self.run_btn.config(state='disabled')
        
        num_episodes = int(self.episode_spin.get())
        thread = threading.Thread(target=self.run_evaluation, args=(num_episodes,))
        thread.start()
        self.after(100, self.check_thread(thread))

    def run_evaluation(self, num_episodes):
        # Load models with proper initialization
        sl = TaskPriorityTransformer().to(DEVICE)
        sl.load_state_dict(torch.load(self.sl_path, map_location=DEVICE))
        
        # Initialize PPOPolicy (not PROPolicy) with proper state dict handling
        rl = PPOPolicy(sl).to(DEVICE)
        state_dict_rl = torch.load(self.rl_path, map_location=DEVICE)
        fixed = {k.replace('pos_encoding','positional_encoding'):v for k,v in state_dict_rl.items()}  # Added .items()
        rl.load_state_dict(fixed)

        for ep in range(num_episodes):
            if not self.running: 
                break
            
            # Use correct method name with underscore
            task_list = SchedulingEnv()._generate_tasks()
            
            # Fixed function names and proper spacing
            metrics = {
                'EDF': evaluate_metrics(lambda e,s: edf_heuristic(s), task_list),  # Fixed gdf->edf
                'SL': evaluate_metrics(lambda e,s: infer_sl(sl, s), task_list),    # Fixed infen->infer
                'TITAN': evaluate_metrics(lambda e,s: infer_rl(rl, s), task_list),
                'FCFS': evaluate_metrics(fcfs_policy, task_list),
                'SJF': evaluate_metrics(sjf_policy, task_list),
                'RR': evaluate_metrics(make_rr_policy(), task_list)
            }
            
            for algo, vals in metrics.items():
                for metric, value in vals.items():
                    self.results[f"{algo}_{metric}"].append(value)
            
            self.current_episode = ep + 1

            # Real-time UI update after each episode
            self.after(0, self.update_visualization)
            self.after(0, self.update_table)
            self.after(0, lambda ep=ep, num_episodes=num_episodes: self.progress_label.config(text=f"Episode {ep+1}/{num_episodes}"))

    def check_thread(self, thread):
        if thread.is_alive():
            self.after(100, lambda: self.check_thread(thread))
        else:
            self.run_btn.config(state='normal')
            self.update_visualization()
            self.update_table()
            self.progress_label.config(text="Completed")

    def update_visualization(self):
        self.ax.clear()
        self.line_ax.clear()
        metrics = ['avg_turnaround', 'avg_waiting', 'miss_rate', 'avg_response', 'fairness_waiting']
        algorithms = ['EDF', 'SL', 'TITAN', 'FCFS', 'SJF', 'RR']
        colors = ['#4e79a7', '#59a14f', '#f28e2b', '#e15759', '#b07aa1', '#ff9da7']
        # --- Bar Chart (Performance Comparison) ---
        avg_values = {}
        for metric in metrics:
            avg_values[metric] = [
                (sum(self.results[f"{algo}_{metric}"]) / len(self.results[f"{algo}_{metric}"])) if len(self.results[f"{algo}_{metric}"]) > 0 else 0.0
                for algo in algorithms
            ]
        width = 0.15
        x = range(len(metrics))
        for i, (algo, color) in enumerate(zip(algorithms, colors)):
            self.ax.bar([xi + width*i for xi in x], [avg_values[m][i] for m in metrics], width=width, label=algo, color=color)
        self.ax.set_xticks([xi + width*2.5 for xi in x])
        self.ax.set_xticklabels([self.metric_names[m] for m in metrics], fontsize=11)
        self.ax.legend(fontsize=10)
        self.ax.set_title("Performance Comparison (Averaged)", fontsize=14, fontweight='bold')
        self.ax.set_ylabel("Metric Value", fontsize=12)
        self.ax.grid(axis='y', linestyle='--', alpha=0.5)
        # --- Line Chart (Metric Trends) ---
        episode_count = len(self.results['TITAN_avg_turnaround'])
        if episode_count > 0:
            episodes = list(range(1, episode_count+1))
            # Only plot the selected metric
            selected_metric_key = self.metric_options[self.metric_dropdown.current() if self.metric_dropdown.current() != -1 else 0]
            for idx, algo in enumerate(algorithms):
                values = self.results[f"{algo}_{selected_metric_key}"]
                if len(values) == episode_count:
                    self.line_ax.plot(episodes, values, label=algo, color=colors[idx], linewidth=2)
            self.line_ax.set_title(f"{self.metric_names[selected_metric_key]} Trend Over Episodes", fontsize=14, fontweight='bold')
            self.line_ax.set_xlabel("Episode", fontsize=12)
            self.line_ax.set_ylabel(self.metric_names[selected_metric_key], fontsize=12)
            self.line_ax.grid(True, linestyle='--', alpha=0.5)
            self.line_ax.legend(fontsize=10, ncol=3, loc='upper right')
        self.fig.tight_layout()
        self.canvas.draw()

    def update_table(self):
        for row in self.results_table.get_children():
            self.results_table.delete(row)
        
        metrics = ['avg_turnaround', 'avg_waiting', 'miss_rate', 'avg_response', 'fairness_waiting']
        algorithms = ['EDF', 'SL', 'TITAN', 'FCFS', 'SJF', 'RR']
        
        avg_values = {}
        for metric in metrics:
            row = [self.metric_names[metric]]
            for algo in algorithms:
                values = self.results[f"{algo}_{metric}"]
                avg = (sum(values) / len(values)) if values else 0.0
                if metric == 'miss_rate':
                    row.append(f"{avg*100:.2f}%")
                elif metric == 'fairness_waiting':
                    row.append(f"{avg:.4f}")
                else:
                    row.append(f"{avg:.4f}")
            self.results_table.insert('', 'end', values=row)

    def on_close(self):
        self.running = False
        self.destroy()

# ----------------------------
# Run the Application
# ----------------------------

if __name__ == "__main__":
    app = SchedulerApp()
    app.mainloop()