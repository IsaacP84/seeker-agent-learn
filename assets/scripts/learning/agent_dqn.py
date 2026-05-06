import sys
print(sys.prefix)
print(sys.path)

import os
import random
import shutil
from datetime import datetime, timedelta
import argparse
import itertools

import torch
from torch import nn

import yaml
import numpy as np
import json
import io
import zipfile
import tempfile
import math

import matplotlib
import matplotlib.pyplot as plt

from learning.dqn import DQN
from learning.experience_replay import ReplayMemory

import Magic

DATE_FORMAT = "%m-%d %H:%M:%S"

ACTION_NAMES = ['Forward', 'Backward', 'Strafe L', 'Strafe R', 'Turn L', 'Turn R']

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

# Secondary search location — assets folder bundled with the project.
# Resolved at runtime so Magic.ASSETS_FOLDER is available.
def _assets_runs_dir():
    try:
        import Magic
        return os.path.join(Magic.ASSETS_FOLDER, "learning", "runs")
    except Exception:
        return None

matplotlib.use('Agg')


device = "cuda" if torch.cuda.is_available() else "cpu"

def log_message(file, msg):
    print(msg)
    with open(file, 'a') as file:
        file.write(msg + '\n')

def find_latest_checkpoint(model_file=None):
    # Prefer RUNS_DIR (bin/runs) — most recent .pt there takes priority.
    pt_files = [
        os.path.join(RUNS_DIR, f)
        for f in os.listdir(RUNS_DIR)
        if f.endswith('.pt')
    ] if os.path.isdir(RUNS_DIR) else []

    if pt_files:
        return max(pt_files, key=os.path.getmtime)

    # Fall back to assets learning/runs folder.
    assets_runs = _assets_runs_dir()
    if assets_runs and os.path.isdir(assets_runs):
        asset_pt_files = [
            os.path.join(assets_runs, f)
            for f in os.listdir(assets_runs)
            if f.endswith('.pt')
        ]
        if asset_pt_files:
            chosen = max(asset_pt_files, key=os.path.getmtime)
            print(f'No checkpoint in runs/, falling back to assets: {chosen}')
            return chosen

    if model_file and os.path.exists(model_file):
        return model_file
    return None


def _load_patch_config(config_json=None):
    if config_json is not None:
        with open(config_json, 'r') as f:
            return json.load(f)

    try:
        env = Magic.NavigationEnv()
        return env.get_config_data()
    except Exception as e:
        raise RuntimeError(f'Unable to obtain current env config from Magic.NavigationEnv(): {e}')


def patch_checkpoint_epsilon(checkpoint_file=None, epsilon=1.0, backup=True, model_file=None):
    if checkpoint_file is None:
        checkpoint_file = find_latest_checkpoint(model_file=model_file)

    if checkpoint_file is None:
        raise FileNotFoundError("No checkpoint file found to patch.")

    if backup:
        backup_file = checkpoint_file + '.bak'
        shutil.copy2(checkpoint_file, backup_file)
        print(f'Created backup: {backup_file}')

    checkpoint = torch.load(checkpoint_file, map_location=device)
    checkpoint['epsilon'] = epsilon
    torch.save(checkpoint, checkpoint_file)
    print(f'Patched checkpoint: {checkpoint_file} (epsilon={epsilon})')


def patch_checkpoint_env_config(checkpoint_file=None, backup=True, model_file=None, config_json=None):
    if checkpoint_file is None:
        checkpoint_file = find_latest_checkpoint(model_file=model_file)

    if checkpoint_file is None:
        raise FileNotFoundError("No checkpoint file found to patch.")

    if backup:
        backup_file = checkpoint_file + '.bak'
        shutil.copy2(checkpoint_file, backup_file)
        print(f'Created backup: {backup_file}')

    checkpoint = torch.load(checkpoint_file, map_location=device)
    env_config = _load_patch_config(config_json)
    checkpoint['env_config'] = env_config
    torch.save(checkpoint, checkpoint_file)
    print(f'Patched checkpoint config: {checkpoint_file}')


class Agent:
    def __init__(self, hyperparameter_set):
        with open(Magic.ASSETS_FOLDER + "\\hyperparameters.yml", "r") as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set
        self.replay_memory_size = hyperparameters["replay_memory_size"]
        self.mini_batch_size = hyperparameters["mini_batch_size"]
        self.epsilon_init = hyperparameters["epsilon_init"]
        self.epsilon_decay = hyperparameters["epsilon_decay"]
        self.epsilon_limit = hyperparameters["epsilon_limit"]
        self.network_sync_rate = hyperparameters["network_sync_rate"]
        self.learning_rate_a = hyperparameters["learning_rate_a"]
        self.discount_factor_g = hyperparameters["discount_factor_g"]
        self.stop_on_reward = hyperparameters["stop_on_reward"]
        self.fc1_nodes = hyperparameters["fc1_nodes"]
        # Backwards-compatible: prefer `hidden_dims` (int or list), fall back to `fc1_nodes`
        raw_hidden = hyperparameters.get("hidden_dims", None)
        if raw_hidden is None:
            self.hidden_dims = self.fc1_nodes
        else:
            if isinstance(raw_hidden, str):
                try:
                    parsed = yaml.safe_load(raw_hidden)
                except Exception:
                    parsed = raw_hidden
            else:
                parsed = raw_hidden
            self.hidden_dims = parsed

        self.env_make_params = hyperparameters.get("env_make_params", {})

        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
    
    def setup(self, num_states, num_actions, is_training=True, load_model_file = None):
        self.num_states = num_states
        self.num_actions = num_actions
        self.is_training = is_training

        self.policy_dqn = DQN(num_states, num_actions, self.hidden_dims).to(device)
        self.target_dqn = DQN(num_states, num_actions, self.hidden_dims).to(device)
        self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

        self.optimizer = torch.optim.Adam(
            self.policy_dqn.parameters(), lr=self.learning_rate_a
        )
        self.memory = ReplayMemory(self.replay_memory_size)
        self.epsilon = self.epsilon_init
        self.best_reward = float('-inf')
        self.rewards_per_episode = []
        self.epsilon_history = []
        self.goal_time_limit_history = []
        self._episode_reward = 0.0
        self._env = None
        self._loaded_env_data = None

        # --- Per-episode diagnostic accumulators (reset each episode) ---
        self._episode_steps = 0
        self._action_counts = [0] * num_actions
        self._goal_count = 0
        self._episode_losses = []
        self._episode_q_values = []   # list of per-step max-Q values
        self._gradient_norms = []
        self._obs_stats_samples = []  # (mean, std, min, max) sampled every 10 steps

        # --- Persistent diagnostic histories (saved in checkpoint) ---
        self.action_dist_history = []      # list[list[int]] — action counts per episode
        self.loss_history = []             # mean TD loss per episode
        self.q_value_history = []          # (mean, min, max) Q-values per episode
        self.episode_length_history = []   # steps per episode
        self.buffer_fill_history = []      # replay buffer fill fraction
        self.gradient_norm_history = []    # mean gradient norm per episode
        self.goal_reach_history = []       # goals reached per episode

        # Epsilon milestone markers: list of (episode_index, threshold)
        # Filled once as epsilon crosses each threshold.
        self._epsilon_phase_milestones = [0.75, 0.5, 0.25, 0.15]
        self._epsilon_phase_markers = []

        # Load the most recent checkpoint — prefer RUNS_DIR (bin/runs),
        # fall back to assets/learning/runs if nothing found there.
        pt_files = [
            os.path.join(RUNS_DIR, f)
            for f in os.listdir(RUNS_DIR)
            if f.endswith('.pt')
        ] if os.path.isdir(RUNS_DIR) else []
        latest_checkpoint = (
            max(pt_files, key=os.path.getmtime)
            if pt_files else
            self.MODEL_FILE if os.path.exists(self.MODEL_FILE) else None
        )

        if latest_checkpoint is None:
            assets_runs = _assets_runs_dir()
            if assets_runs and os.path.isdir(assets_runs):
                asset_pt_files = [
                    os.path.join(assets_runs, f)
                    for f in os.listdir(assets_runs)
                    if f.endswith('.pt')
                ]
                if asset_pt_files:
                    latest_checkpoint = max(asset_pt_files, key=os.path.getmtime)
                    print(f'No checkpoint in runs/, falling back to assets: {latest_checkpoint}')
        
        if(load_model_file):
            latest_checkpoint = load_model_file

        if latest_checkpoint:
            print(f'Loading checkpoint: {latest_checkpoint}')
            try:
                checkpoint = torch.load(latest_checkpoint, map_location=device)

                # Validate that the saved model's input/output dimensions match
                # the current environment before loading weights.
                state_dict = checkpoint['policy_dqn']
                # Find the first Linear layer's input size and the last one's output size.
                first_weight_key = next(
                    (k for k in state_dict if k.endswith('.weight')), None
                )
                last_weight_key = next(
                    (k for k in reversed(list(state_dict.keys())) if k.endswith('.weight')), None
                )
                if first_weight_key and last_weight_key:
                    ckpt_inputs = state_dict[first_weight_key].shape[1]
                    ckpt_outputs = state_dict[last_weight_key].shape[0]
                    if ckpt_inputs != num_states or ckpt_outputs != num_actions:
                        if 'env_config' not in checkpoint:
                            print(
                                f'Warning: checkpoint input/output mismatch '
                                f'(checkpoint: {ckpt_inputs} inputs, {ckpt_outputs} outputs; '
                                f'current: {num_states} inputs, {num_actions} outputs) and env_config is missing. '
                                'Checkpoint loading will fail.'
                            )
                            latest_checkpoint = None
                            adapted_checkpoint = False
                        else:
                            print(
                                f'Warning: checkpoint input/output mismatch '
                                f'(checkpoint: {ckpt_inputs} inputs, {ckpt_outputs} outputs; '
                                f'current: {num_states} inputs, {num_actions} outputs). '
                                f'Attempting to adapt checkpoint to current model.'
                            )
                            try:
                                from learning.checkpoint_adapter import load_state_dict_with_adapt
                                new_config = None
                                try:
                                    new_config = Magic.NavigationEnv().get_config_data()
                                except Exception:
                                    pass
                                load_state_dict_with_adapt(
                                    self.policy_dqn,
                                    state_dict,
                                    old_config=checkpoint.get('env_config'),
                                    new_config=new_config,
                                )
                                load_state_dict_with_adapt(
                                    self.target_dqn,
                                    state_dict,
                                    old_config=checkpoint.get('env_config'),
                                    new_config=new_config,
                                )
                                adapted_checkpoint = True
                            except Exception as e:
                                print(f'Failed to adapt checkpoint: {e}. Starting fresh.')
                                latest_checkpoint = None
                                adapted_checkpoint = False
                    else:
                        adapted_checkpoint = False
                else:
                    adapted_checkpoint = False

                if latest_checkpoint:
                    if not adapted_checkpoint:
                        self.policy_dqn.load_state_dict(state_dict)
                        self.target_dqn.load_state_dict(state_dict)
                    if is_training:
                        if 'optimizer' in checkpoint:
                            self.optimizer.load_state_dict(checkpoint['optimizer'])
                        if 'epsilon' in checkpoint:
                            self.epsilon = checkpoint['epsilon']
                        if 'rewards_per_episode' in checkpoint:
                            self.rewards_per_episode = checkpoint['rewards_per_episode']
                        if 'epsilon_history' in checkpoint:
                            self.epsilon_history = checkpoint['epsilon_history']
                        if 'goal_time_limit_history' in checkpoint:
                            self.goal_time_limit_history = checkpoint['goal_time_limit_history']
                        if 'action_dist_history' in checkpoint:
                            self.action_dist_history = checkpoint['action_dist_history']
                        if 'loss_history' in checkpoint:
                            self.loss_history = checkpoint['loss_history']
                        if 'q_value_history' in checkpoint:
                            self.q_value_history = checkpoint['q_value_history']
                        if 'episode_length_history' in checkpoint:
                            self.episode_length_history = checkpoint['episode_length_history']
                        if 'buffer_fill_history' in checkpoint:
                            self.buffer_fill_history = checkpoint['buffer_fill_history']
                        if 'gradient_norm_history' in checkpoint:
                            self.gradient_norm_history = checkpoint['gradient_norm_history']
                        if 'goal_reach_history' in checkpoint:
                            self.goal_reach_history = checkpoint['goal_reach_history']
                        if 'epsilon_phase_markers' in checkpoint:
                            self._epsilon_phase_markers = checkpoint['epsilon_phase_markers']
                            # Remove already-passed thresholds from milestones.
                            passed = {t for _, t in self._epsilon_phase_markers}
                            self._epsilon_phase_milestones = [m for m in self._epsilon_phase_milestones if m not in passed]
                        if 'env_data' in checkpoint:
                            self._loaded_env_data = checkpoint['env_data']
                    log_message(self.LOG_FILE, f'Loaded checkpoint: {latest_checkpoint} (epsilon={self.epsilon:.4f})')
            except Exception as e:
                print(f'Warning: failed to load checkpoint {latest_checkpoint}: {e}. Starting fresh.')
        log_message(self.LOG_FILE, f"Using model file: {self.MODEL_FILE}")
        if is_training:
            start_time = datetime.now()
            self.last_graph_update_time = start_time
            msg = f'{start_time.strftime(DATE_FORMAT)}: Training started...'
            log_message(self.LOG_FILE, msg)
        
    def bind_env(self, env):
        """Attach a NavigationEnv instance so env data is saved/restored with checkpoints."""
        self._env = env
        if self._loaded_env_data is not None:
            env.set_env_data(self._loaded_env_data)
            print(f'Restored env data from checkpoint: {self._loaded_env_data}')
            self._loaded_env_data = None

    def step(self, obs):
        if not hasattr(self, 'policy_dqn'):
            raise RuntimeError('Agent not initialized. Call setup() before step().')

        state = torch.tensor(obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            q_vals = self.policy_dqn(state.unsqueeze(0)).squeeze()

        # Track Q-value stats every step (even when action is random).
        if hasattr(self, '_episode_q_values'):
            self._episode_q_values.append(float(q_vals.max().item()))

        if self.is_training and random.random() < self.epsilon:
            return random.randrange(self.num_actions)

        return int(q_vals.argmax().item())

    def observe(self, state, action, reward, next_state, done):
        if not hasattr(self, 'memory'):
            raise RuntimeError('Agent not initialized. Call setup() before observe().')

        state = torch.tensor(state, dtype=torch.float32, device=device)
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        action = torch.tensor(action, dtype=torch.int64, device=device)
        reward = torch.tensor(reward, dtype=torch.float32, device=device)
        done = torch.tensor(done, dtype=torch.float32, device=device)

        # Accumulate reward for the current episode.
        self._episode_reward += reward.item()

        # --- Per-step diagnostic tracking ---
        self._episode_steps += 1
        if hasattr(self, '_action_counts'):
            self._action_counts[action.item()] += 1
        # Detect goal reach: reward > 9.0 means a goal was found (partial reward +10).
        if reward.item() > 9.0 and hasattr(self, '_goal_count'):
            self._goal_count += 1
        # Observation health: sample mean/std/min/max every 10 steps.
        if self._episode_steps % 10 == 0 and hasattr(self, '_obs_stats_samples'):
            obs_arr = state.cpu().numpy()
            self._obs_stats_samples.append(
                (float(obs_arr.mean()), float(obs_arr.std()),
                 float(obs_arr.min()), float(obs_arr.max()))
            )
            # Warn if observation looks dead or out of expected range.
            if obs_arr.std() < 1e-6:
                print(f'[OBS WARNING] Step {self._episode_steps}: observation is near-zero/constant (std={obs_arr.std():.2e})')

        if not self.is_training:
            return

        self.memory.append((state, action, next_state, reward, done))

        if len(self.memory) > self.mini_batch_size:
            mini_batch = self.memory.sample(self.mini_batch_size)
            self.optimize(mini_batch, self.policy_dqn, self.target_dqn)

        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_limit)
        self.epsilon_history.append(self.epsilon)

        if bool(done) and hasattr(self, 'target_dqn'):
            self.target_dqn.load_state_dict(self.policy_dqn.state_dict())

            self.rewards_per_episode.append(self._episode_reward)
            self._episode_reward = 0.0

            current_limit = self._env.get_env_data()['current_goal_time_limit'] if self._env is not None else 0.0
            self.goal_time_limit_history.append(current_limit)

            # --- Record per-episode diagnostic histories ---
            self.action_dist_history.append(list(self._action_counts))
            self.episode_length_history.append(self._episode_steps)
            self.goal_reach_history.append(self._goal_count)
            self.buffer_fill_history.append(len(self.memory) / self.replay_memory_size)

            mean_loss = float(np.mean(self._episode_losses)) if self._episode_losses else 0.0
            self.loss_history.append(mean_loss)

            if self._episode_q_values:
                self.q_value_history.append((
                    float(np.mean(self._episode_q_values)),
                    float(np.min(self._episode_q_values)),
                    float(np.max(self._episode_q_values)),
                ))
            else:
                self.q_value_history.append((0.0, 0.0, 0.0))

            mean_grad = float(np.mean(self._gradient_norms)) if self._gradient_norms else 0.0
            self.gradient_norm_history.append(mean_grad)

            # Check epsilon phase milestones.
            for threshold in list(self._epsilon_phase_milestones):
                if self.epsilon <= threshold:
                    self._epsilon_phase_markers.append((len(self.rewards_per_episode), threshold))
                    self._epsilon_phase_milestones.remove(threshold)

            episode = len(self.rewards_per_episode)
            q_mean, q_min, q_max = self.q_value_history[-1]
            print(f'Episode {episode:4d} | '
                  f'reward={self.rewards_per_episode[-1]:8.2f} | '
                  f'epsilon={self.epsilon:.4f} | '
                  f'goal_limit={current_limit:.1f}s | '
                  f'steps={self._episode_steps:4d} | '
                  f'goals={self._goal_count} | '
                  f'loss={mean_loss:.4f} | '
                  f'q_mean={q_mean:.3f} | '
                  f'grad={mean_grad:.3f} | '
                  f'buf={self.buffer_fill_history[-1]*100:.1f}%')
            log_message(self.LOG_FILE,
                f'Episode {episode:4d} | '
                f'reward={self.rewards_per_episode[-1]:8.2f} | '
                f'epsilon={self.epsilon:.4f} | '
                f'goal_limit={current_limit:.1f}s | '
                f'steps={self._episode_steps:4d} | '
                f'goals={self._goal_count} | '
                f'loss={mean_loss:.4f} | '
                f'q_mean={q_mean:.3f} | '
                f'grad={mean_grad:.3f} | '
                f'buf={self.buffer_fill_history[-1]*100:.1f}%'
            )

            # Obs health summary for the episode.
            if self._obs_stats_samples:
                means = [s[0] for s in self._obs_stats_samples]
                stds  = [s[1] for s in self._obs_stats_samples]
                if np.mean(stds) < 1e-5:
                    log_message(self.LOG_FILE, f'  [OBS WARNING] Episode {episode}: very low obs std ({np.mean(stds):.2e}) — sensor may be dead.')

            # Reset per-episode accumulators.
            self._action_counts = [0] * self.num_actions
            self._episode_losses = []
            self._episode_q_values = []
            self._episode_steps = 0
            self._goal_count = 0
            self._obs_stats_samples = []
            self._gradient_norms = []

            self.save_graph(self.rewards_per_episode, self.epsilon_history, self.goal_time_limit_history)

            episode = len(self.rewards_per_episode)
            checkpoint_data = {
                'policy_dqn':              self.policy_dqn.state_dict(),
                'optimizer':               self.optimizer.state_dict(),
                'epsilon':                 self.epsilon,
                'rewards_per_episode':     self.rewards_per_episode,
                'epsilon_history':         self.epsilon_history,
                'goal_time_limit_history': self.goal_time_limit_history,
                # diagnostic histories
                'action_dist_history':     self.action_dist_history,
                'loss_history':            self.loss_history,
                'q_value_history':         self.q_value_history,
                'episode_length_history':  self.episode_length_history,
                'buffer_fill_history':     self.buffer_fill_history,
                'gradient_norm_history':   self.gradient_norm_history,
                'goal_reach_history':      self.goal_reach_history,
                'epsilon_phase_markers':   self._epsilon_phase_markers,
            }
            if self._env is not None:
                checkpoint_data['env_data'] = self._env.get_env_data()
                checkpoint_data['env_config'] = self._env.get_config_data()

            # Save rolling checkpoint every episode.
            torch.save(checkpoint_data, self.MODEL_FILE)

            # Save a new timestamped checkpoint every 100 episodes.
            if episode % 100 == 0:
                ckpt_file = os.path.join(
                    RUNS_DIR,
                    f'{self.hyperparameter_set}_ep{episode:08d}.pt'
                )
                torch.save(checkpoint_data, ckpt_file)
                print(f'Checkpoint saved: {ckpt_file}')

    def run(self, is_training=True, render=False):
        raise NotImplementedError(
            "Agent.run() is not supported in this scene-driven setup. Use Agent.step(obs) from the engine update loop instead."
        )

    def optimize(self, mini_batch, policy_dqn, target_dqn):
        
        states, actions, new_states, rewards, terminations = zip(*mini_batch)
        
        states = torch.stack(states)
        actions = torch.stack(actions)
        new_states = torch.stack(new_states)
        rewards = torch.tensor(rewards).float().to(device)
        terminations = torch.tensor(terminations).float().to(device)
        
        with torch.no_grad():
            target_q = rewards + (1-terminations) * self.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
        
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        loss = self.loss_fn(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()

        # --- Gradient norm (#12) ---
        total_norm = 0.0
        for p in policy_dqn.parameters():
            if p.grad is not None:
                total_norm += p.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        if hasattr(self, '_gradient_norms'):
            self._gradient_norms.append(total_norm)

        self.optimizer.step()

        # --- Loss tracking (#2) ---
        if hasattr(self, '_episode_losses'):
            self._episode_losses.append(loss.item())
        
    def save_graph(self, rewards_per_episode, epsilon_history, goal_time_limit_history=None):
        n = len(rewards_per_episode)
        if n == 0:
            return

        def rolling_mean(data, window=100):
            arr = np.array(data, dtype=float)
            result = np.zeros(len(arr))
            for i in range(len(arr)):
                result[i] = arr[max(0, i - window + 1):i + 1].mean()
            return result

        fig, axes = plt.subplots(3, 3, figsize=(21, 13))
        fig.suptitle(f'Agent Diagnostics — {self.hyperparameter_set}  (ep {n})', fontsize=13)

        # ── (0,0) Mean Reward + Curriculum Difficulty ──────────────────────────
        ax = axes[0, 0]
        mean_rewards = rolling_mean(rewards_per_episode)
        ax.plot(mean_rewards, color='tab:blue', label='100-ep mean')
        ax.set_ylabel('Mean Reward', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax.set_xlabel('Episode')
        ax.set_title('Reward + Curriculum')
        if goal_time_limit_history:
            ax2 = ax.twinx()
            offset = n - len(goal_time_limit_history)
            ax2.plot(range(offset, offset + len(goal_time_limit_history)),
                     goal_time_limit_history, color='tab:orange', alpha=0.6, label='Goal limit (s)')
            ax2.set_ylabel('Goal Time Limit (s)', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')

        # ── (0,1) Episode Length ────────────────────────────────────────────────
        ax = axes[0, 1]
        if self.episode_length_history:
            lengths = np.array(self.episode_length_history, dtype=float)
            ax.plot(lengths, alpha=0.3, color='tab:purple', linewidth=0.8)
            ax.plot(rolling_mean(lengths, 100), color='tab:purple', label='100-ep mean')
        ax.set_ylabel('Steps / Episode')
        ax.set_xlabel('Episode')
        ax.set_title('Episode Length')

        # ── (0,2) Goals Reached per Episode ────────────────────────────────────
        ax = axes[0, 2]
        if self.goal_reach_history:
            goals = np.array(self.goal_reach_history, dtype=float)
            ax.plot(goals, alpha=0.3, color='tab:green', linewidth=0.8)
            ax.plot(rolling_mean(goals, 100), color='tab:green')
        ax.set_ylabel('Goals Reached')
        ax.set_xlabel('Episode')
        ax.set_title('Goal-Find Rate')

        # ── (1,0) TD Loss ───────────────────────────────────────────────────────
        ax = axes[1, 0]
        if self.loss_history:
            loss_arr = np.array(self.loss_history, dtype=float)
            ax.plot(loss_arr, alpha=0.3, color='tab:red', linewidth=0.8)
            ax.plot(rolling_mean(loss_arr, 100), color='tab:red')
        ax.set_ylabel('Mean TD Loss')
        ax.set_xlabel('Episode')
        ax.set_title('Loss Curve')
        ax.set_yscale('log') if self.loss_history and max(self.loss_history) > 0 else None

        # ── (1,1) Q-Value Statistics ────────────────────────────────────────────
        ax = axes[1, 1]
        if self.q_value_history:
            q_means = np.array([q[0] for q in self.q_value_history], dtype=float)
            q_mins  = np.array([q[1] for q in self.q_value_history], dtype=float)
            q_maxs  = np.array([q[2] for q in self.q_value_history], dtype=float)
            xs = np.arange(len(q_means))
            ax.plot(rolling_mean(q_means, 100), color='tab:blue', label='Q mean')
            ax.fill_between(xs, rolling_mean(q_mins, 100), rolling_mean(q_maxs, 100),
                            alpha=0.2, color='tab:blue', label='Q min/max')
            ax.legend(fontsize=7)
        ax.set_ylabel('Q-Value')
        ax.set_xlabel('Episode')
        ax.set_title('Q-Value Statistics')

        # ── (1,2) Gradient Norm ─────────────────────────────────────────────────
        ax = axes[1, 2]
        if self.gradient_norm_history:
            gnorm = np.array(self.gradient_norm_history, dtype=float)
            ax.plot(gnorm, alpha=0.3, color='tab:brown', linewidth=0.8)
            ax.plot(rolling_mean(gnorm, 100), color='tab:brown')
        ax.set_ylabel('Gradient L2 Norm')
        ax.set_xlabel('Episode')
        ax.set_title('Gradient Norm')

        # ── (2,0) Epsilon Decay + Phase Markers ────────────────────────────────
        ax = axes[2, 0]
        ax.plot(epsilon_history, color='tab:cyan', linewidth=0.8)
        ax.set_ylabel('Epsilon')
        ax.set_xlabel('Step')
        ax.set_title('Epsilon Decay + Phase Markers')
        marker_colors = {0.75: 'red', 0.5: 'orange', 0.25: 'green', 0.15: 'blue'}
        for ep_idx, threshold in self._epsilon_phase_markers:
            # ep_idx is episode number; convert to approximate step index.
            approx_step = ep_idx * (len(epsilon_history) // max(n, 1))
            ax.axvline(x=approx_step, color=marker_colors.get(threshold, 'gray'),
                       linestyle='--', alpha=0.7, linewidth=1)
            ax.text(approx_step, threshold + 0.02, f'{threshold}',
                    fontsize=7, color=marker_colors.get(threshold, 'gray'))

        # ── (2,1) Replay Buffer Fill % ──────────────────────────────────────────
        ax = axes[2, 1]
        if self.buffer_fill_history:
            fill = np.array(self.buffer_fill_history, dtype=float) * 100
            ax.plot(fill, color='tab:olive', linewidth=0.8)
            ax.axhline(y=100, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
            ax.set_ylim(0, 105)
        ax.set_ylabel('Buffer Fill (%)')
        ax.set_xlabel('Episode')
        ax.set_title('Replay Buffer Utilization')

        # ── (2,2) Action Distribution (last 100 episodes avg) ──────────────────
        ax = axes[2, 2]
        if self.action_dist_history:
            recent = np.array(self.action_dist_history[-100:], dtype=float)
            avg_counts = recent.mean(axis=0)
            total = avg_counts.sum()
            pct = (avg_counts / total * 100) if total > 0 else avg_counts
            num_actions = len(pct)
            labels = ACTION_NAMES[:num_actions] if num_actions <= len(ACTION_NAMES) else [str(i) for i in range(num_actions)]
            colors = plt.cm.tab10(np.linspace(0, 0.9, num_actions))
            bars = ax.bar(labels, pct, color=colors)
            for bar, val in zip(bars, pct):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=7)
            ax.set_ylabel('Usage (%)')
            ax.set_title('Action Distribution (last 100 eps)')
            ax.set_ylim(0, 100)

        plt.tight_layout()
        fig.savefig(self.GRAPH_FILE, dpi=100)
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hyperparameters', help='')
    parser.add_argument('--train', help="Training mode", action='store_true')
    parser.add_argument('--patch-epsilon', type=float,
                        help='Patch checkpoint epsilon without altering training data')
    parser.add_argument('--patch-config', action='store_true',
                        help='Patch checkpoint to include env_config for layout-aware loading')
    parser.add_argument('--config-json', help='Optional JSON file for env_config instead of current env')
    parser.add_argument('--checkpoint-file', help='Optional checkpoint file path to patch')
    args = parser.parse_args()

    if args.patch_epsilon is not None:
        patch_checkpoint_epsilon(args.checkpoint_file, args.patch_epsilon,
                                 backup=True, model_file=os.path.join(RUNS_DIR, f'{args.hyperparameters}.pt'))
    elif args.patch_config:
        patch_checkpoint_env_config(
            args.checkpoint_file,
            backup=True,
            model_file=os.path.join(RUNS_DIR, f'{args.hyperparameters}.pt'),
            config_json=args.config_json,
        )
    else:
        dql = Agent(hyperparameter_set=args.hyperparameters)
        if args.train:
            dql.run(is_training=True)
        else:
            dql.run(is_training=False, render=True)
