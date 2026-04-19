import os
import math
import torch
import numpy as np
import pandas as pd
import subprocess
import random
import re
import json
import logging
import warnings
import time
import csv
import argparse
import sys
import requests
from tqdm import tqdm

# Correctly import LinearReward from its submodule
import mo_gymnasium as mo_gym
from mo_gymnasium.wrappers import LinearReward

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
import torch.nn as nn
import torch.nn.functional as F

# Our reproducible energy model
import energy_model

# --- Basic Setup ---
warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def set_seed(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# --- Environment Configuration ---
application = "factorizator"
app_env = "factorizator"
NAMESPACE = "factorizator"

# Prometheus (MicroK8s observability stack)
PROMETHEUS_URL = "add the prometheus endpoint here"

# Service endpoint (NodePort reachable from master)
url_app_service = "add your url app service address here"

# Kubernetes command wrapper
KUBECTL_CMD = "microk8s kubectl"

cpu_target_percentage = 50
MIN_REPLICAS = 1
MAX_REPLICAS = 100
columns = 1440              # minutes in a simulated day
SLA_LATENCY = 0.020        # 20 ms SLA
P_IDLE_W = 50.0
P_MAX_W  = 250.0
BETA2 = 1  # energy reward weight (can be overridden via command-line argument)

# --- Data Sampling ---
def add_day_column(df, minutes_per_day):
    timestamps = pd.to_datetime(df['end_timestamp'], unit='s')
    t0 = timestamps.min()
    df['minute'] = (timestamps - t0).dt.total_seconds() // 60
    df['minute'] = df['minute'].astype(int)
    df['day'] = df['minute'] // minutes_per_day
    return df

def get_random_days(df, n_days, train_days, test_days, seed=42):
    all_days = sorted(df['day'].unique())
    if len(all_days) < n_days:
        raise ValueError(f"Not enough unique days in the dataset. Found {len(all_days)}, need {n_days}.")
    random.seed(seed)
    selected_days = random.sample(list(all_days), n_days)
    selected_days.sort()
    return selected_days[:train_days], selected_days[train_days:train_days+test_days]

# --- Transformer feature extractor ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class CustomTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=256, nhead=4, num_layers=2):
        super().__init__(observation_space, features_dim)
        input_dim = np.prod(observation_space.shape)
        self.input_proj = nn.Linear(input_dim, features_dim)
        self.pos_encoder = PositionalEncoding(features_dim)
        encoder_layers = nn.TransformerEncoderLayer(d_model=features_dim, nhead=nhead, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        self.output_fc = nn.Linear(features_dim, features_dim)

    def forward(self, observations):
        if observations.dim() == 2:
            observations = observations.unsqueeze(1)
        x = self.input_proj(observations)
        x = x.permute(1, 0, 2)
        x = self.pos_encoder(x)
        x = x.permute(1, 0, 2)
        transformer_output = self.transformer_encoder(x)
        pooled_output = transformer_output[:, -1, :]
        features = F.relu(self.output_fc(pooled_output))
        return features

class CustomTransformerPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs,
                         features_extractor_class=CustomTransformerExtractor,
                         features_extractor_kwargs=dict(features_dim=256))

def lr_schedule(progress_remaining: float) -> float:
    lr_min, lr_max = 1e-5, 2e-4
    return lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * (1 - progress_remaining)))

def parse_k8s_metric(val: str) -> float:
    val = str(val)
    if val.endswith("m"): return float(val[:-1])
    if val.endswith("Ki"): return float(val[:-2]) / 1024
    if val.endswith("Mi"): return float(val[:-2])
    return float(val)

# --- Environment ---
class KubernetesEnv(gym.Env):
    def __init__(self, df, minutes, day_list):
        super().__init__()
        self.full_df = df
        self.action_space = gym.spaces.MultiDiscrete([5, 3, 3, 3])
        self.day_list = day_list
        self.minutes_per_day = minutes
        self.invocation_matrix = self._make_invocation_matrix()
        self.hpa_target = cpu_target_percentage
        self.steps, self.days, self.global_step = 0, 0, 0
        low = np.array([0.0001, 1, 0, 0, 0, 0, 0, 0, 10, 1.0, 0, -1, -1, 0, 0], dtype=np.float32)
        high = np.array([10.0, MAX_REPLICAS, 1000, 1000, 100000, 10000, 10000, 1, 100, 3.0, 2, 1, 1, 100000, 1000], dtype=np.float32)
        self.observation_space = gym.spaces.Box(low=low, high=high, dtype=np.float32)
        self.reward_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.reset()
        self.max_steps = self.minutes_per_day  # e.g., 1080 control intervals per simulated day
        self.current_step = 0

    def _make_invocation_matrix(self):
        matrix = np.zeros((len(self.day_list), self.minutes_per_day))
        for i, day in enumerate(self.day_list):
            mask = self.full_df['day'] == day
            df_day = self.full_df[mask].groupby(self.full_df['minute'] % self.minutes_per_day).size().reset_index(name='invocations')
            for _, row in df_day.iterrows():
                minute_of_day = int(row['minute'])
                if minute_of_day < self.minutes_per_day:
                    matrix[i, minute_of_day] = row['invocations']
        return matrix

    def _get_k8s_metrics(self):
        try:
            cpu_req_str = subprocess.check_output(
                f"{KUBECTL_CMD} get deployment {application} -n {app_env} -o jsonpath='{{.spec.template.spec.containers[0].resources.requests.cpu}}'",
                shell=True, encoding="utf-8").strip()
            cpu_req = float(cpu_req_str[:-1]) if cpu_req_str.endswith('m') else float(cpu_req_str) * 1000

            output = subprocess.check_output(
                f"{KUBECTL_CMD} top pod -l app={application} -n {app_env} --no-headers",
                shell=True, encoding="utf-8", stderr=subprocess.DEVNULL)
            lines = output.strip().split('\n')
            if not lines or not lines[0]:
                return 1, 0, 0, 0, 0, 0

            cpu_vals, ram_vals = [], []
            for line in lines:
                cols = line.split()
                if len(cols) > 2:
                    cpu_vals.append(parse_k8s_metric(cols[1]))
                    ram_vals.append(parse_k8s_metric(cols[2]))

            total_cpu_m, total_ram_mi = sum(cpu_vals), sum(ram_vals)
            avg_cpu_percent = (np.mean(cpu_vals) / cpu_req * 100) if cpu_vals and cpu_req > 0 else 0

            replicas = int(subprocess.check_output(
                f"{KUBECTL_CMD} get deployment {application} -n {app_env} -o jsonpath='{{.spec.replicas}}'",
                shell=True, encoding="utf-8").strip())

            node_name = subprocess.check_output(
                f"{KUBECTL_CMD} get nodes -o jsonpath='{{.items[0].metadata.name}}'",
                shell=True, encoding="utf-8").strip()
            node_metrics = subprocess.check_output(
                f"{KUBECTL_CMD} top node {node_name} --no-headers",
                shell=True, encoding="utf-8").strip().split()
            node_cpu_usage_cores = parse_k8s_metric(node_metrics[1]) / 1000
            node_cpu_capacity_cores = float(subprocess.check_output(
                f"{KUBECTL_CMD} get node {node_name} -o jsonpath='{{.status.capacity.cpu}}'",
                shell=True, encoding="utf-8").strip())
            node_cpu_util_fraction = node_cpu_usage_cores / node_cpu_capacity_cores if node_cpu_capacity_cores > 0 else 0

            node_power = energy_model.estimate_node_power(node_cpu_util_fraction)
            total_pod_power = energy_model.attribute_pod_power(total_cpu_m / 1000, node_cpu_usage_cores, node_power, replicas)

        except Exception as e:
            logging.warning(f"Metric collection failed: {e}")
            return 1, 0, 0, 0, 0, 0

        return replicas, avg_cpu_percent, 0, total_cpu_m, total_ram_mi, total_pod_power, node_power

    def _run_hey(self):
        if self.days >= len(self.day_list): 
            return 1, 5.0, 0.0
        if self.steps >= self.minutes_per_day:
            self.days += 1
            self.steps = 0
        if self.days >= len(self.invocation_matrix): 
            return 1, 5.0, 0.0

        num_requests = int(self.invocation_matrix[self.days, self.steps])
        num_requests = max(num_requests, 1)
        concurrency = min(num_requests, 50)

        try:
            command = f"hey -n {num_requests} -c {concurrency} -z 60s -m GET '{url_app_service}'"
            process = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
            output = process.stdout

            # --- Extract latency (in seconds) ---
            latency = 5.0  # default fallback
            try:
                latency_match = re.search(r"Average:\s+(\d+\.\d+)\s*(ms|secs)?", output)
                if latency_match:
                    latency_value = float(latency_match.group(1))
                    unit = latency_match.group(2)
                    if unit == "ms":
                        latency = latency_value / 1000.0
                    else:
                        latency = latency_value
            except Exception as e:
                logging.warning(f"Latency parsing failed: {e}")

            # --- Extract successful requests ---
            total_match = re.search(r"Requests/sec:\s+(\d+\.\d+)", output)
            requests_per_sec = float(total_match.group(1)) if total_match else 0.0

            # --- Extract 2xx responses ---
            success_match = re.search(r"\[2..]\s+(\d+)\s+responses", output)
            success_count = int(success_match.group(1)) if success_match else 0

            success_ratio = min(success_count / num_requests, 1.0) if num_requests > 0 else 0.0

            return num_requests, latency, success_ratio

        except Exception as e:
            logging.error(f"Error in _run_hey: {e}")
            return num_requests, 5.0, 0.0



    def compute_reward_vector(self, state):
        lat = float(state[0])       
        reps = float(state[1])      
        Pt = float(state[14])     

        r_perf = 1.0 - (lat / SLA_LATENCY)                
        r_cost = 1.0 - (reps / float(MAX_REPLICAS))      
        r_energy = -BETA2 * (Pt / float(P_MAX_W))       

        return np.array([r_perf, r_cost, r_energy], dtype=np.float32)


    def apply_action(self, action):
        hpaoptions = [10, 30, 50, 70, 90]
        newhpatarget = int(hpaoptions[int(action[0])])

        if newhpatarget != self.hpatarget:
            self.hpatarget = newhpatarget

            patch = {
                "spec": {
                    "metrics": [{
                        "type": "Resource",
                        "resource": {
                            "name": "cpu",
                            "target": {
                                "type": "Utilization",
                                "averageUtilization": self.hpatarget
                            }
                        }
                    }]
                }
            }

            try:
                subprocess.run(
                    f"{KUBECTL_CMD} patch hpa {application} -n {app_env} --patch '{json.dumps(patch)}'",
                    shell=True,
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL
                )
            except Exception as e:
                logging.error(f"HPA patch failed: {e}")

    def step(self, action):
        """Main step logic for Gymnasium API"""
        self.apply_action(action)
        num_req, latency, success_ratio = self._run_hey()
        replicas, cpu_usage, ram_usage, total_cpu, total_ram, total_pod_power = self._get_k8s_metrics()
        angle = (self.steps / self.minutes_per_day) * (2 * math.pi)
        forecast = 0.0

        self.state = np.array([
            latency, replicas, cpu_usage, ram_usage, num_req, total_cpu, total_ram,
            success_ratio, self.hpa_target, 1.0, 0, math.cos(angle), math.sin(angle),
            forecast, total_pod_power
        ], dtype=np.float32)

        reward_vector = self.compute_reward_vector(self.state)

        # --- Step counters and termination logic ---
        self.current_step += 1
        self.global_step += 1
        self.steps += 1

        terminated = self.current_step >= self.max_steps
        truncated = False  # explicit for Gymnasium API

        # reset daily cycle counters (so next day starts cleanly)
        if terminated:
            self.current_step = 0
            self.days += 1

        info = {"raw_reward": reward_vector}

        return self.state, reward_vector, terminated, truncated, info

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        logging.info("Resetting environment to a clean state...")
        subprocess.run(f"{KUBECTL_CMD} delete hpa {application} -n {app_env} --ignore-not-found=true",
                       shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(f"{KUBECTL_CMD} scale deployment/{application} --replicas=1 -n {app_env}",
                       shell=True, check=True)
        time.sleep(15)
        subprocess.run(f"{KUBECTL_CMD} autoscale deployment {application} -n {app_env} "
                       f"--cpu-percent={cpu_target_percentage} --min=1 --max={MAX_REPLICAS}",
                       shell=True, check=True)
        time.sleep(5)
        logging.info("Environment reset complete.")
        self.hpa_target = cpu_target_percentage
        self.steps = self.days = self.global_step = 0
        self.state = np.zeros(self.observation_space.shape, dtype=np.float32)
        self.state[1] = 1
        self.state[8] = cpu_target_percentage
        return self.state, {}

# --- Logging callback ---
class MultiObjectiveLoggingCallback(BaseCallback):
    def __init__(self, log_path, policy_name, verbose=0):
        super().__init__(verbose)
        self.log_path = log_path
        self.policy_name = policy_name
        os.makedirs(self.log_path, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        self.csv_path = os.path.join(self.log_path, f"{self.policy_name}_train_{timestamp}.csv")

    def _on_training_start(self):
        self.csv_file = open(self.csv_path, "w", newline="")
        headers = ["timestep", "r_perf", "r_cost", "r_energy", "latency", "replicas", "avg_cpu_percent", "ram_usage",
                   "num_requests", "total_cpu_m", "total_ram_mi", "success_ratio", "deployment_power_watts", "hpa_target"]
        self.csv_writer = csv.DictWriter(self.csv_file, fieldnames=headers)
        self.csv_writer.writeheader()

    def _on_step(self) -> bool:
        info = self.locals['infos'][0]
        if 'raw_reward' in info:
            reward_vector = info['raw_reward']
            state = self.training_env.get_attr('state')[0]
            self.logger.record('rewards/performance', reward_vector[0])
            self.logger.record('rewards/cost', reward_vector[1])
            self.logger.record('rewards/energy', reward_vector[2])
            self.logger.record('performance/latency_ms', float(state[0]) * 1000)
            self.logger.record('performance/success_ratio', float(state[7]))
            self.logger.record('cost/replicas', state[1])
            self.logger.record('energy/deployment_power_watts', state[14])
            self.logger.record('resources/avg_cpu_percent', state[2])
            self.logger.record('resources/total_cpu_millicores', state[5])
            self.logger.record('resources/total_ram_mib', state[6])
            self.logger.record('workload/requests_per_min', state[4])
            self.logger.record('actions/hpa_target', state[8])
            self.logger.dump(step=self.num_timesteps)
            self.csv_writer.writerow({
                "timestep": self.num_timesteps, "r_perf": reward_vector[0], "r_cost": reward_vector[1],
                "r_energy": reward_vector[2], "latency": state[0], "replicas": state[1],
                "avg_cpu_percent": state[2], "ram_usage": state[3], "num_requests": state[4],
                "total_cpu_m": state[5], "total_ram_mi": state[6], "success_ratio": state[7],
                "deployment_power_watts": float(state[14]), "hpa_target": state[8]
            })
            logging.info(f"Step: {self.num_timesteps:<5} | Policy: {self.policy_name:<15} | "
                         f"Reward: [{reward_vector[0]:>5.2f},{reward_vector[1]:>5.2f},{reward_vector[2]:>5.2f}] | "
                         f"Replicas: {state[1]:<2.0f} | Latency: {state[0]*1000:<6.1f} ms | "
                         f"Success: {state[7]:<4.2f} | DeploymentPower: {float(state[14]):<5.1f} W")
        return True

    def _on_training_end(self):
        if hasattr(self, 'csv_file'):
            self.csv_file.close()

# --- Main ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["train", "test"], default="train",
                        help="Run mode: train or test")
    parser.add_argument("--profile", choices=["perf_focused", "cost_focused", "energy_focused", "balanced"],
                        default="perf_focused", help="Profile to train/test")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--beta2", type=float, default=None, help="Override energy reward weight (optional)")
    args = parser.parse_args()

    set_seed(args.seed)

    invocation_file = "AzureFunctionsInvocationTraceForTwoWeeksJan2021.txt"
    try:
        logging.info(f"Loading trace file: {invocation_file}")
        df_full = pd.read_csv(invocation_file)
        df_full['end_timestamp'] = pd.to_numeric(df_full['end_timestamp'], errors='coerce')
        df_full.dropna(subset=['end_timestamp'], inplace=True)
        df_full = add_day_column(df_full, minutes_per_day=columns)
        train_days, test_days = get_random_days(df_full, n_days=4, train_days=3, test_days=1, seed=args.seed)
        logging.info(f"Train days: {train_days}, Test days: {test_days}")
    except Exception as e:
        logging.error(f"Error loading trace file: {e}")
        sys.exit(1)

    weights_dict = {
        "perf_focused": np.array([0.8, 0.1, 0.1]),
        "cost_focused": np.array([0.1, 0.8, 0.1]),
        "energy_focused": np.array([0.1, 0.1, 0.8]),
        "balanced": np.array([0.4, 0.3, 0.3])
    }

    if args.beta2 is not None:
        print(f"[CONFIG] Overriding beta2 for profile '{args.profile}' with {args.beta2}")
        weights_dict[args.profile][2] = args.beta2

    selected_profile = args.profile
    weights = weights_dict[selected_profile]

    if args.mode == "train":
        logging.info(f"===== TRAIN MODE: {selected_profile} =====")
        logging.info(f"Training with weights: {weights}")

        def make_env():
            mo_env = KubernetesEnv(df_full.copy(), minutes=columns, day_list=train_days)
            return LinearReward(mo_env, weight=weights)

        env = DummyVecEnv([make_env])
        env.seed(args.seed)
        env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
        device = "cuda" if torch.cuda.is_available() else "cpu"

        model = PPO(
            CustomTransformerPolicy, env,
            n_steps=256, batch_size=64, ent_coef=0.01,
            gamma=0.99, gae_lambda=0.93, learning_rate=lr_schedule,
            verbose=0, tensorboard_log=f"./results/tb/{selected_profile}",
            device=device, seed=args.seed
        )

        ckpt_dir, log_dir = f"./results/checkpoints/{selected_profile}", "./results/logs"
        os.makedirs(ckpt_dir, exist_ok=True)
        callback = MultiObjectiveLoggingCallback(log_path=log_dir, policy_name=selected_profile)

        # --- Autosave every 360 steps ---
        checkpoint_callback = CheckpointCallback(
            save_freq=360,
            save_path=ckpt_dir,
            name_prefix=f"{selected_profile}_step",
            save_replay_buffer=False,
            save_vecnormalize=True,
        )

        total_steps = len(train_days) * columns

        try:
            model.learn(
                total_timesteps=total_steps,
                progress_bar=True,
                callback=[callback, checkpoint_callback]
            )
        finally:
            # Always save model and normalization stats, even if interrupted
            logging.info("Saving checkpoint and normalization state...")
            model.save(os.path.join(ckpt_dir, "final_model.zip"))
            try:
                env.save(os.path.join(ckpt_dir, "vecnorm.pkl"))
                logging.info("VecNormalize state saved successfully.")
            except Exception as e:
                logging.warning(f"VecNormalize save failed: {e}")
            logging.info(f"Training complete or interrupted for '{selected_profile}'.")

    elif args.mode == "test":
        from stable_baselines3.common.logger import configure as configure_logger

        logging.info(f"===== TEST MODE: {selected_profile} =====")
        ckpt_dir = f"./results/checkpoints/{selected_profile}"
        model_path = os.path.join(ckpt_dir, "final_model.zip")
        vecnorm_path = os.path.join(ckpt_dir, "vecnorm.pkl")

        if not os.path.exists(model_path):
            logging.error(f"Model not found at {model_path}. Train it first.")
            sys.exit(1)

        def make_env_test():
            mo_env = KubernetesEnv(df_full.copy(), minutes=columns, day_list=test_days)
            return LinearReward(mo_env, weight=weights)

        env = DummyVecEnv([make_env_test])
        env.seed(args.seed)
        try:
            env = VecNormalize.load(vecnorm_path, env)
            env.training = False
            env.norm_reward = False
        except Exception as e:
            logging.warning(f"VecNormalize not found or failed to load: {e}")

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logging.info(f"Using device: {device.upper()}")

        tensorboard_log = f"./results/tb_test/{selected_profile}"
        os.makedirs(tensorboard_log, exist_ok=True)
        model = PPO.load(model_path, env=env, device=device)
        model.set_logger(configure_logger(tensorboard_log, ["tensorboard"]))
        logging.info(f"Logging to {tensorboard_log}")

        test_dir = f"./results/test/{selected_profile}"
        os.makedirs(test_dir, exist_ok=True)
        csv_path = os.path.join(test_dir, f"{selected_profile}_test_run.csv")

        total_steps = len(test_days) * columns
        logging.info(f"Starting test run ({total_steps} steps)...")

        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=[
                "timestep", "r_perf", "r_cost", "r_energy", "latency", "success_ratio",
                "replicas", "avg_cpu_percent", "total_cpu_m", "total_ram_mi",
                "deployment_power_watts", "hpa_target", "reward_scalar"
            ])
            writer.writeheader()

            obs = env.reset()
            pbar = tqdm(range(total_steps), desc=f"Testing {selected_profile}", ncols=100)

            for t in pbar:
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, info = env.step(action)
                raw_state = env.envs[0].unwrapped.state
                raw_info = info[0] if isinstance(info, (list, tuple)) else info

                r_perf, r_cost, r_energy = (np.nan, np.nan, np.nan)
                if isinstance(raw_info, dict) and "raw_reward" in raw_info:
                    try:
                        r_perf, r_cost, r_energy = map(float, raw_info["raw_reward"])
                    except Exception:
                        pass

                writer.writerow({
                    "timestep": t + 1,
                    "r_perf": r_perf,
                    "r_cost": r_cost,
                    "r_energy": r_energy,
                    "latency": float(raw_state[0]),
                    "success_ratio": float(raw_state[7]),
                    "replicas": int(raw_state[1]),
                    "avg_cpu_percent": float(raw_state[2]),
                    "total_cpu_m": float(raw_state[5]),
                    "total_ram_mi": float(raw_state[6]),
                    "pod_power": float(raw_state[14]),
                    "hpa_target": float(raw_state[8]),
                    "reward_scalar": float(reward[0] if np.ndim(reward) else reward),
                })

                model.logger.record('test/r_perf', r_perf)
                model.logger.record('test/r_cost', r_cost)
                model.logger.record('test/r_energy', r_energy)
                model.logger.record('test/latency_ms', raw_state[0] * 1000)
                model.logger.record('test/success_ratio', raw_state[7])
                model.logger.record('test/replicas', raw_state[1])
                model.logger.record('test/power_watts', raw_state[14])
                model.logger.record('test/reward_scalar', float(reward[0] if np.ndim(reward) else reward))
                model.logger.dump(step=t)

                pbar.set_postfix({
                    "lat(ms)": f"{raw_state[0]*1000:.2f}",
                    "succ": f"{raw_state[7]:.2f}",
                    "rep": f"{raw_state[1]:.0f}",
                    "power(W)": f"{raw_state[14]:.1f}"
                })

                if done[0]:
                    break

        logging.info(f"Test complete. Results saved at {csv_path}")


if __name__ == "__main__":
    main()
