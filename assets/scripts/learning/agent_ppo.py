import sys
print(sys.prefix)
print(sys.path)

import os
import io
import json
import zipfile
import shutil
import threading
import time
from datetime import datetime
import argparse

import torch

import yaml
import numpy as np

import matplotlib
import matplotlib.pyplot as plt

from learning.ppo_network import Actor, Critic
from learning.trajectory_buffer import RolloutBuffer

import Magic

DATE_FORMAT = "%m-%d %H:%M:%S"

ACTION_NAMES = ['Forward', 'Backward', 'Strafe L', 'Strafe R', 'Turn L', 'Turn R', 'Jump', 'Idle']

RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)


def _assets_runs_dir():
    try:
        return os.path.join(Magic.ASSETS_FOLDER, "learning", "runs")
    except Exception:
        return None


matplotlib.use('Agg')

device = "cuda" if torch.cuda.is_available() else "cpu"


def log_message(file, msg):
    print(msg)
    with open(file, 'a') as f:
        f.write(msg + '\n')


def _is_checkpoint_file(name):
    return name.endswith('.pt') or name.endswith('.pt.zip')


def _load_checkpoint(path):
    if path.endswith('.zip'):
        with zipfile.ZipFile(path, 'r') as archive:
            members = [m for m in archive.namelist() if m.endswith('.pt')]
            if not members:
                raise FileNotFoundError(f'No .pt file found inside archive: {path}')
            with archive.open(members[0], 'r') as fh:
                return torch.load(io.BytesIO(fh.read()), map_location=device)
    return torch.load(path, map_location=device)


def find_latest_checkpoint(model_file=None):
    pt_files = [
        os.path.join(RUNS_DIR, f)
        for f in os.listdir(RUNS_DIR)
        if _is_checkpoint_file(f)
    ] if os.path.isdir(RUNS_DIR) else []
    if pt_files:
        return max(pt_files, key=os.path.getmtime)
    assets_runs = _assets_runs_dir()
    if assets_runs and os.path.isdir(assets_runs):
        asset_pt_files = [
            os.path.join(assets_runs, f)
            for f in os.listdir(assets_runs)
            if _is_checkpoint_file(f)
        ]
        if asset_pt_files:
            chosen = max(asset_pt_files, key=os.path.getmtime)
            print(f'No checkpoint in runs/, falling back to assets: {chosen}')
            return chosen
    if model_file and os.path.exists(model_file):
        return model_file
    return None


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

class Agent:
    def __init__(self, hyperparameter_set):
        with open(Magic.ASSETS_FOLDER + "\\hyperparameters.yml", "r") as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.n_steps           = hyperparameters["n_steps"]
        self.n_epochs          = hyperparameters["n_epochs"]
        self.mini_batch_size   = hyperparameters["mini_batch_size"]
        self.clip_epsilon      = hyperparameters["clip_epsilon"]
        self.gae_lambda        = hyperparameters["gae_lambda"]
        self.discount_factor_g = hyperparameters["discount_factor_g"]
        self.entropy_coeff     = hyperparameters["entropy_coeff"]
        self.value_loss_coeff  = hyperparameters["value_loss_coeff"]
        self.learning_rate_a   = hyperparameters["learning_rate_a"]
        self.max_grad_norm     = hyperparameters["max_grad_norm"]
        self.stop_on_reward    = hyperparameters.get("stop_on_reward", 999999)
        self.lr_min            = hyperparameters.get("lr_min", self.learning_rate_a)
        self.lr_decay_episodes = hyperparameters.get("lr_decay_episodes", 0)
        self.reward_scale      = hyperparameters.get("reward_scale", 1.0)

        raw_hidden = hyperparameters.get("hidden_dims", 256)
        if isinstance(raw_hidden, int):
            raw_hidden = [raw_hidden]
        self.hidden_dims = raw_hidden

        self.env_make_params = hyperparameters.get("env_make_params", {})

        self.optimizer = None
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')
        self.DIAGNOSTICS_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_diagnostics.json')

    # ------------------------------------------------------------------
    def setup(self, num_states, num_actions, is_training=True, load_model_file=None, num_agents=1, checkpoint_config=None):
        self.num_states  = num_states
        self.num_actions = num_actions
        self.is_training = is_training

        checkpoint_config = checkpoint_config or {}
        self.checkpoint_config = checkpoint_config
        self.rolling_mode = checkpoint_config.get('rolling_mode', 'compact')
        self.max_history_entries = int(checkpoint_config.get('max_history_entries', 0))
        self.downsample_rate = max(1, int(checkpoint_config.get('downsample_rate', 1)))
        self.compress = checkpoint_config.get('compress', None)
        self.full_checkpoint_interval = int(checkpoint_config.get('full_checkpoint_interval', 100))
        self.save_replay_buffer_on_full = bool(checkpoint_config.get('save_replay_buffer_on_full', False))
        self.save_diagnostics_separately = bool(checkpoint_config.get('save_diagnostics_separately', False))
        self.archival_float16 = bool(checkpoint_config.get('archival_float16', False))
        self.diagnostics_filename_template = checkpoint_config.get('diagnostics_filename_template', '{name}_diagnostics.json')
        self.DIAGNOSTICS_FILE = os.path.join(
            RUNS_DIR,
            self.diagnostics_filename_template.format(name=self.hyperparameter_set)
        )

        self.actor  = Actor( num_states, num_actions, self.hidden_dims).to(device)
        self.critic = Critic(num_states,              self.hidden_dims).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate_a,
        )

        if self.lr_decay_episodes > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(1, self.lr_decay_episodes * num_agents),
                eta_min=self.lr_min,
            )
        else:
            self.scheduler = None

        self.num_agents = num_agents

        self.buffer = RolloutBuffer(self.n_steps, device=device)

        self.best_reward             = float('-inf')
        self.rewards_per_episode     = []
        self.goal_time_limit_history = []
        self._env                    = None
        self._loaded_env_data        = None

        self._last_log_probs = {}  # keyed by agent_id
        self._last_values    = {}  # keyed by agent_id
        self._steps_since_ppo = 0  # steps added to buffer since last _run_ppo_update

        # Background PPO worker thread — update runs off the main game thread
        # so the C++ loop is not stalled during the 160-step gradient descent.
        self._ppo_lock          = threading.Lock()   # guards actor/critic writes
        self._ppo_worker        = None               # current worker thread (or None)

        # Per-agent episode accumulators (keyed by agent_id; reset when agent's episode ends)
        self._agent_slots          = {}

        # Per-episode PPO loss accumulators (global; reset when any agent's episode ends)
        self._episode_pg_losses    = []
        self._episode_value_losses = []
        self._episode_entropies    = []
        self._episode_clip_fracs   = []
        self._episode_approx_kls   = []
        self._gradient_norms       = []
        self._gradient_norms_post_clip = []

        # Persistent diagnostic histories
        self.action_dist_history    = []
        self.pg_loss_history        = []
        self.value_loss_history     = []
        self.entropy_history        = []
        self.clip_frac_history      = []
        self.approx_kl_history      = []
        self.episode_length_history = []
        self.gradient_norm_history  = []
        self.gradient_norm_post_clip_history = []
        self.goal_reach_history     = []

        # Locate checkpoint
        pt_files = [
            os.path.join(RUNS_DIR, f)
            for f in os.listdir(RUNS_DIR)
            if f.endswith('.pt')
        ] if os.path.isdir(RUNS_DIR) else []
        latest_checkpoint = (
            max(pt_files, key=os.path.getmtime) if pt_files
            else self.MODEL_FILE if os.path.exists(self.MODEL_FILE) else None
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

        if load_model_file:
            latest_checkpoint = load_model_file

        if latest_checkpoint:
            print(f'Loading checkpoint: {latest_checkpoint}')
            try:
                checkpoint = _load_checkpoint(latest_checkpoint)

                if 'actor' not in checkpoint:
                    print('Warning: checkpoint has no "actor" key (may be a DQN checkpoint). Starting fresh.')
                    latest_checkpoint = None
                else:
                    actor_sd   = checkpoint['actor']
                    first_key  = next((k for k in actor_sd if k.endswith('.weight')), None)
                    last_key   = next((k for k in reversed(list(actor_sd.keys())) if k.endswith('.weight')), None)
                    if first_key and last_key:
                        ckpt_in  = actor_sd[first_key].shape[1]
                        ckpt_out = actor_sd[last_key].shape[0]
                        if ckpt_in != num_states or ckpt_out != num_actions:
                            print(
                                f'Warning: checkpoint shape mismatch '
                                f'(ckpt: {ckpt_in} in, {ckpt_out} out; '
                                f'current: {num_states} in, {num_actions} out). Starting fresh.'
                            )
                            latest_checkpoint = None

                if latest_checkpoint:
                    self.actor.load_state_dict(checkpoint['actor'])
                    self.critic.load_state_dict(checkpoint['critic'])

                    # If reward_scale != 1 and the checkpoint was saved without
                    # scaling, the critic head outputs are in the old reward range.
                    # Rescale the output layer immediately so the first rollout's
                    # advantages are valid (avoids a catastrophic first update).
                    if self.reward_scale != 1.0 and checkpoint.get('reward_scale', 1.0) != self.reward_scale:
                        scale = self.reward_scale / checkpoint.get('reward_scale', 1.0)
                        with torch.no_grad():
                            self.critic.head.weight.mul_(scale)
                            self.critic.head.bias.mul_(scale)
                        print(f'Rescaled critic head by {scale:.4f} to match reward_scale={self.reward_scale}')

                    if is_training:
                        if 'optimizer' in checkpoint:
                            self.optimizer.load_state_dict(checkpoint['optimizer'])
                        if 'scheduler' in checkpoint and self.scheduler is not None:
                            self.scheduler.load_state_dict(checkpoint['scheduler'])

                        history_round = checkpoint.get('history_round', 0)
                        if history_round == 0 and 'rewards_per_episode' in checkpoint:
                            history_round = len(checkpoint['rewards_per_episode'])

                        def _trim_to_checkpoint(hist):
                            return hist[:history_round] if isinstance(hist, list) and history_round > 0 else hist

                        for key in ('rewards_per_episode', 'goal_time_limit_history',
                                    'action_dist_history', 'pg_loss_history',
                                    'value_loss_history', 'entropy_history',
                                    'clip_frac_history', 'approx_kl_history',
                                    'episode_length_history', 'gradient_norm_history',
                                    'gradient_norm_post_clip_history', 'goal_reach_history'):
                            if key in checkpoint:
                                setattr(self, key, _trim_to_checkpoint(checkpoint[key]))

                        if self.save_diagnostics_separately and os.path.exists(self.DIAGNOSTICS_FILE):
                            try:
                                with open(self.DIAGNOSTICS_FILE, 'r') as df:
                                    diag = json.load(df)
                                for key in ('rewards_per_episode', 'goal_time_limit_history',
                                            'action_dist_history', 'pg_loss_history',
                                            'value_loss_history', 'entropy_history',
                                            'clip_frac_history', 'approx_kl_history',
                                            'episode_length_history', 'gradient_norm_history',
                                            'gradient_norm_post_clip_history', 'goal_reach_history'):
                                    if key in diag:
                                        setattr(self, key, _trim_to_checkpoint(diag[key]))
                            except Exception:
                                pass
                        if 'env_data' in checkpoint:
                            self._loaded_env_data = checkpoint['env_data']
                    log_message(self.LOG_FILE, f'Loaded checkpoint: {latest_checkpoint}')
            except Exception as e:
                print(f'Warning: failed to load checkpoint {latest_checkpoint}: {e}. Starting fresh.')

        log_message(self.LOG_FILE, f"Using model file: {self.MODEL_FILE}")
        if is_training:
            start_time = datetime.now()
            self.last_graph_update_time = start_time
            log_message(self.LOG_FILE, f'{start_time.strftime(DATE_FORMAT)}: Training started (PPO)...')

    # ------------------------------------------------------------------
    def bind_env(self, env):
        """Attach a NavigationEnv instance so env data is saved/restored with checkpoints."""
        self._env = env
        if self._loaded_env_data is not None:
            env.set_env_data(self._loaded_env_data)
            print(f'Restored env data from checkpoint: {self._loaded_env_data}')
            self._loaded_env_data = None

    # ------------------------------------------------------------------
    def step(self, obs, agent_id=0):
        """Select an action. Training: stochastic sample. Inference: greedy."""
        if not hasattr(self, 'actor'):
            raise RuntimeError('Agent not initialized. Call setup() before step().')

        state = torch.tensor(obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            if self.is_training:
                action, log_prob, _ = self.actor.get_action(state.unsqueeze(0))
                value = self.critic(state.unsqueeze(0)).squeeze()
                self._last_log_probs[agent_id] = log_prob.item()
                self._last_values[agent_id]    = value.item()
            else:
                logits = self.actor(state.unsqueeze(0)).squeeze()
                action = logits.argmax()
                self._last_log_probs[agent_id] = 0.0
                self._last_values[agent_id]    = 0.0

        return int(action.item())

    # ------------------------------------------------------------------
    def observe(self, state, action, reward, next_state, done, agent_id=0):
        """Record a transition; trigger PPO update when the rollout buffer is ready."""
        if not hasattr(self, 'buffer'):
            raise RuntimeError('Agent not initialized. Call setup() before observe().')

        reward_f = float(reward) * self.reward_scale
        done_b   = bool(done)

        # Per-agent episode accumulators.
        if agent_id not in self._agent_slots:
            self._agent_slots[agent_id] = {
                'reward': 0.0, 'steps': 0,
                'action_counts': [0] * self.num_actions,
                'goals': 0, 'obs_stats': [],
            }
        slot = self._agent_slots[agent_id]
        slot['reward'] += reward_f
        slot['steps']  += 1
        slot['action_counts'][int(action)] += 1
        if reward_f > 9.0 * self.reward_scale:
            slot['goals'] += 1
        if slot['steps'] % 10 == 0:
            obs_arr = np.asarray(state, dtype=np.float32)
            slot['obs_stats'].append(
                (float(obs_arr.mean()), float(obs_arr.std()),
                 float(obs_arr.min()),  float(obs_arr.max()))
            )
            if obs_arr.std() < 1e-6:
                print(f'[OBS WARNING] Agent {agent_id} step {slot["steps"]}: obs near-constant (std={obs_arr.std():.2e})')

        if not self.is_training:
            return

        self.buffer.add(
            obs      = state,
            action   = action,
            reward   = reward_f,
            done     = done_b,
            log_prob = self._last_log_probs.get(agent_id, 0.0),
            value    = self._last_values.get(agent_id, 0.0),
            agent_id = agent_id,
        )
        self._steps_since_ppo += 1

        # Flush when buffer is full (rollout complete — truncated bootstrap)
        _flushed_full = False
        if self.buffer.is_full():
            with torch.no_grad():
                ns_t     = torch.tensor(next_state, dtype=torch.float32, device=device)
                last_val = self.critic(ns_t.unsqueeze(0)).squeeze().item()
            self._run_ppo_update_async(last_val)
            _flushed_full = True

        # Single-agent only: flush on episode end if the buffer has partial data.
        if self.num_agents == 1 and done_b and not _flushed_full and self.buffer._size > 0:
            with torch.no_grad():
                ns_t     = torch.tensor(next_state, dtype=torch.float32, device=device)
                last_val = self.critic(ns_t.unsqueeze(0)).squeeze().item()
            self._run_ppo_update_async(last_val)

    # ------------------------------------------------------------------
    def on_round_complete(self):
        """Called by C++ when all agents have finished their episode.

        1. Flush any remaining rollout buffer data as a PPO update.
        2. Aggregate per-agent episode stats across all agents for this round.
        3. Log a single round-summary line and update diagnostic histories.
        4. Reset all per-agent accumulators and PPO loss accumulators.
        5. Step LR scheduler and save graph/checkpoint on schedule.
        """
        # -- 1. PPO flush (only if enough new data since last update) --
        min_steps = self.mini_batch_size * max(self.num_agents, 1)
        if self.is_training and self._steps_since_ppo >= min_steps:
            self._run_ppo_update_async(last_value=0.0)

        # Wait for any in-flight PPO worker to finish so loss values are
        # available when we build the log line and graph below.
        if self._ppo_worker is not None and self._ppo_worker.is_alive():
            self._ppo_worker.join()

        if not self._agent_slots:
            return

        # -- 2. Aggregate stats across all agent slots for this round --
        all_rewards = [s['reward'] for s in self._agent_slots.values()]
        all_steps   = [s['steps']  for s in self._agent_slots.values()]
        all_goals   = [s['goals']  for s in self._agent_slots.values()]
        combined_actions = [
            sum(s['action_counts'][i] for s in self._agent_slots.values())
            for i in range(self.num_actions)
        ]

        mean_reward = float(np.mean(all_rewards))
        mean_steps  = float(np.mean(all_steps))
        total_goals = int(sum(all_goals))
        mean_goals  = float(np.mean(all_goals))
        current_limit = (
            self._env.get_env_data()['current_goal_time_limit']
            if self._env is not None else 0.0
        )

        self.rewards_per_episode.append(mean_reward)
        self.goal_time_limit_history.append(current_limit)
        self.action_dist_history.append(combined_actions)
        self.episode_length_history.append(int(mean_steps))
        self.goal_reach_history.append(mean_goals)

        def _last_or(hist, default=0.0):
            return hist[-1] if hist else default

        self.pg_loss_history.append(
            float(np.mean(self._episode_pg_losses))    if self._episode_pg_losses    else _last_or(self.pg_loss_history))
        self.value_loss_history.append(
            float(np.mean(self._episode_value_losses)) if self._episode_value_losses else _last_or(self.value_loss_history))
        self.entropy_history.append(
            float(np.mean(self._episode_entropies))    if self._episode_entropies    else _last_or(self.entropy_history))
        self.clip_frac_history.append(
            float(np.mean(self._episode_clip_fracs))   if self._episode_clip_fracs   else _last_or(self.clip_frac_history))
        self.approx_kl_history.append(
            float(np.mean(self._episode_approx_kls))   if self._episode_approx_kls   else _last_or(self.approx_kl_history))
        mean_grad = float(np.mean(self._gradient_norms)) if self._gradient_norms else _last_or(self.gradient_norm_history)
        mean_grad_post = float(np.mean(self._gradient_norms_post_clip)) if self._gradient_norms_post_clip else _last_or(self.gradient_norm_post_clip_history)
        self.gradient_norm_history.append(mean_grad)
        self.gradient_norm_post_clip_history.append(mean_grad_post)
        # True when we have fresh loss values from a completed rollout this round.
        has_fresh_losses = bool(self._episode_pg_losses)

        round_num = len(self.rewards_per_episode)

        # -- 3. Log round summary --
        agent_rewards_str = ' '.join(f'{r:.2f}' for r in all_rewards)
        msg = (
            f'Round {round_num:4d} | '
            f'mean_reward={mean_reward:8.2f} | [{agent_rewards_str}] | '
            f'goal_limit={current_limit:.1f}s | '
            f'mean_steps={mean_steps:.0f} | '
            f'goals={total_goals} (mean/agent={mean_goals:.2f})'
            + (f' | pg={self.pg_loss_history[-1]:.4f}'
               f' | vf={self.value_loss_history[-1]:.4f}'
               f' | ent={self.entropy_history[-1]:.4f}'
               f' | kl={self.approx_kl_history[-1]:.4f}'
               f' | clip={self.clip_frac_history[-1]:.3f}'
               f' | grad={mean_grad:.3f}'
               if has_fresh_losses else ' | (no rollout this round)')
        )
        log_message(self.LOG_FILE, msg)

        for agent_id, slot in self._agent_slots.items():
            if slot['obs_stats']:
                stds = [s[1] for s in slot['obs_stats']]
                if np.mean(stds) < 1e-5:
                    log_message(self.LOG_FILE,
                        f'  [OBS WARNING] Round {round_num} agent {agent_id}: very low obs std ({np.mean(stds):.2e})')

        # -- 4. Reset accumulators --
        self._agent_slots          = {}
        self._episode_pg_losses    = []
        self._episode_value_losses = []
        self._episode_entropies    = []
        self._episode_clip_fracs   = []
        self._episode_approx_kls   = []
        self._gradient_norms       = []
        self._gradient_norms_post_clip = []

        # -- 5. Scheduler + checkpoint/graph --
        # Step once per agent-episode to preserve the same LR decay rate as
        # the original per-episode stepping (T_max was already scaled by num_agents).
        if self.scheduler is not None:
            for _ in range(self.num_agents):
                self.scheduler.step()

        self.save_graph()
        if round_num % 1 == 0:
            self._save_checkpoint()

    # ------------------------------------------------------------------
    def _run_ppo_update_async(self, last_value):
        """Snapshot the buffer and run the PPO update on a background thread.

        If a previous update is still running we wait for it first — this
        prevents a second update from reading the shared buffer/model while
        the first is still writing to the weights.  In practice the training
        time (~160 gradient steps) is well under one rollout (~1024 steps @
        4 agents), so the wait should be near-zero.
        """
        if self._ppo_worker is not None and self._ppo_worker.is_alive():
            # Previous update hasn't finished — wait so we don't stomp it.
            t_wait_start = time.perf_counter()
            self._ppo_worker.join()
            wait_ms = (time.perf_counter() - t_wait_start) * 1000.0
            log_message(self.LOG_FILE, f'[PPO] waited {wait_ms:.1f}ms for previous worker to finish')

        # Snapshot the buffer contents for the worker thread.
        # buffer.prepare() + get_batches() only reads from this snapshot, so
        # the main thread can keep adding new steps while training runs.
        t_snap_start = time.perf_counter()
        buf_snapshot = self.buffer.snapshot()
        snap_ms = (time.perf_counter() - t_snap_start) * 1000.0
        self.buffer.clear()
        self._steps_since_ppo = 0
        log_message(self.LOG_FILE,
            f'[PPO] rollout start | buf_size={buf_snapshot._size} | snapshot={snap_ms:.1f}ms')

        def _worker():
            t0 = time.perf_counter()
            with self._ppo_lock:
                self._run_ppo_update(last_value, buf_snapshot)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            log_message(self.LOG_FILE, f'[PPO] rollout done  | elapsed={elapsed_ms:.0f}ms')

        self._ppo_worker = threading.Thread(target=_worker, daemon=True)
        self._ppo_worker.start()

    # ------------------------------------------------------------------
    def _run_ppo_update(self, last_value, buffer=None):
        """Run PPO update epochs over the given buffer snapshot (or self.buffer)."""
        buf = buffer if buffer is not None else self.buffer
        adv, ret = buf.prepare(last_value, self.discount_factor_g, self.gae_lambda)

        # Normalize advantages and write back so get_batches sees the normalized values.
        # (prepare() stores raw advantages on buffer._advantages; get_batches reads from there.)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        buf._advantages = adv

        for _ in range(self.n_epochs):
            for batch in buf.get_batches(self.mini_batch_size):
                obs_b    = batch['obs']
                act_b    = batch['actions']
                old_lp_b = batch['log_probs']
                adv_b    = batch['advantages']
                ret_b    = batch['returns']

                new_log_prob, entropy = self.actor.evaluate_action(obs_b, act_b)
                value = self.critic(obs_b).squeeze(-1)

                ratio  = torch.exp(new_log_prob - old_lp_b)
                surr1  = ratio * adv_b
                surr2  = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * adv_b
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value function clipping: prevents the critic from making
                # large updates that destabilise advantage estimates.
                old_val_b = batch['values']
                value_clipped = old_val_b + torch.clamp(
                    value - old_val_b, -self.clip_epsilon, self.clip_epsilon)
                value_loss_unclipped = (value         - ret_b).pow(2)
                value_loss_clipped   = (value_clipped - ret_b).pow(2)
                value_loss = 0.5 * torch.max(value_loss_unclipped, value_loss_clipped).mean()

                # Entropy floor: boost entropy_coeff when entropy collapses
                # to prevent the policy becoming irreversibly deterministic.
                entropy_floor = 0.5
                effective_entropy_coeff = self.entropy_coeff * max(
                    1.0, entropy_floor / (entropy.mean().item() + 1e-8))
                entropy_loss = -entropy.mean()

                loss = pg_loss + self.value_loss_coeff * value_loss + effective_entropy_coeff * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                params = list(self.actor.parameters()) + list(self.critic.parameters())
                grad_norm_pre_clip = float(torch.nn.utils.clip_grad_norm_(
                    params,
                    self.max_grad_norm
                ))
                grad_norm_post_clip = float(torch.sqrt(
                    sum(p.grad.detach().norm(2).pow(2) for p in params if p.grad is not None)
                ))
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac  = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    approx_kl  = (old_lp_b - new_log_prob).mean().item()
                    total_norm = grad_norm_pre_clip

                self._episode_pg_losses.append(pg_loss.item())
                self._episode_value_losses.append(value_loss.item())
                self._episode_entropies.append(-entropy_loss.item())
                self._episode_clip_fracs.append(clip_frac)
                self._episode_approx_kls.append(approx_kl)
                self._gradient_norms.append(total_norm)
                self._gradient_norms_post_clip.append(grad_norm_post_clip)

        # buffer.clear() and _steps_since_ppo reset are handled by
        # _run_ppo_update_async before the thread starts.

    # ------------------------------------------------------------------
    def _trim_histories(self):
        if self.max_history_entries <= 0:
            return
        history_attrs = [
            'rewards_per_episode', 'goal_time_limit_history', 'action_dist_history',
            'pg_loss_history', 'value_loss_history', 'entropy_history',
            'clip_frac_history', 'approx_kl_history', 'episode_length_history',
            'gradient_norm_history', 'gradient_norm_post_clip_history', 'goal_reach_history'
        ]
        for attr in history_attrs:
            hist = getattr(self, attr, None)
            if isinstance(hist, list) and len(hist) > self.max_history_entries:
                del hist[:-self.max_history_entries]

    def _downsample_list(self, data):
        if self.downsample_rate <= 1 or not data:
            return data
        result = [data[i] for i in range(0, len(data), self.downsample_rate)]
        if data and result[-1] != data[-1]:
            result.append(data[-1])
        return result

    def _save_diagnostics_file(self):
        diagnostics = {
            'round': len(self.rewards_per_episode),
            'rewards_per_episode':    self._downsample_list(self.rewards_per_episode),
            'goal_time_limit_history':self._downsample_list(self.goal_time_limit_history),
            'action_dist_history':    self._downsample_list(self.action_dist_history),
            'pg_loss_history':        self._downsample_list(self.pg_loss_history),
            'value_loss_history':     self._downsample_list(self.value_loss_history),
            'entropy_history':        self._downsample_list(self.entropy_history),
            'clip_frac_history':      self._downsample_list(self.clip_frac_history),
            'approx_kl_history':      self._downsample_list(self.approx_kl_history),
            'episode_length_history': self._downsample_list(self.episode_length_history),
            'gradient_norm_history':  self._downsample_list(self.gradient_norm_history),
            'gradient_norm_post_clip_history': self._downsample_list(self.gradient_norm_post_clip_history),
            'goal_reach_history':     self._downsample_list(self.goal_reach_history),
        }
        filename = self.diagnostics_filename_template.format(name=self.hyperparameter_set)
        filepath = os.path.join(RUNS_DIR, filename)
        with open(filepath, 'w') as f:
            json.dump(diagnostics, f)
        return filepath

    def _apply_archival_float16(self, checkpoint_data):
        if not self.archival_float16 or 'buffer' not in checkpoint_data:
            return
        for key, value in checkpoint_data['buffer'].items():
            if isinstance(value, torch.Tensor) and value.is_floating_point():
                checkpoint_data['buffer'][key] = value.half()

    def _save_pt_file(self, path, data):
        if self.compress == 'zip':
            zip_path = path + '.zip' if not path.endswith('.zip') else path
            buffer = io.BytesIO()
            torch.save(data, buffer)
            with zipfile.ZipFile(zip_path, 'w', compression=zipfile.ZIP_DEFLATED) as zf:
                zf.writestr(os.path.basename(path), buffer.getvalue())
            return zip_path
        torch.save(data, path)
        return path

    def _prune_old_checkpoints(self):
        prefix = f'{self.hyperparameter_set}_ep'
        checkpoint_files = [
            os.path.join(RUNS_DIR, f)
            for f in os.listdir(RUNS_DIR)
            if f.startswith(prefix) and _is_checkpoint_file(f)
        ]
        if len(checkpoint_files) <= self.max_history_entries:
            return
        checkpoint_files.sort(key=os.path.getmtime)
        for old_ckpt in checkpoint_files[:-self.max_history_entries]:
            try:
                os.remove(old_ckpt)
            except OSError:
                pass

    def _save_checkpoint(self):
        episode = len(self.rewards_per_episode)
        self._trim_histories()

        checkpoint_data = {
            'actor':                  self.actor.state_dict(),
            'critic':                 self.critic.state_dict(),
            'optimizer':              self.optimizer.state_dict(),
            'reward_scale':           self.reward_scale,
            'history_round':          episode,
            **({'scheduler': self.scheduler.state_dict()} if self.scheduler is not None else {}),
        }

        if not self.save_diagnostics_separately:
            checkpoint_data.update({
                'rewards_per_episode':    self.rewards_per_episode,
                'goal_time_limit_history':self.goal_time_limit_history,
                'action_dist_history':    self.action_dist_history,
                'pg_loss_history':        self.pg_loss_history,
                'value_loss_history':     self.value_loss_history,
                'entropy_history':        self.entropy_history,
                'clip_frac_history':      self.clip_frac_history,
                'approx_kl_history':      self.approx_kl_history,
                'episode_length_history': self.episode_length_history,
                'gradient_norm_history':  self.gradient_norm_history,
                'gradient_norm_post_clip_history': self.gradient_norm_post_clip_history,
                'goal_reach_history':     self.goal_reach_history,
            })

        if self._env is not None:
            checkpoint_data['env_data'] = self._env.get_env_data()

        model_path = self._save_pt_file(self.MODEL_FILE, checkpoint_data)
        if model_path != self.MODEL_FILE:
            self.MODEL_FILE = model_path

        if self.save_diagnostics_separately:
            diag_path = self._save_diagnostics_file()
        else:
            diag_path = None

        if self.full_checkpoint_interval > 0 and episode % self.full_checkpoint_interval == 0:
            ckpt_file = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_ep{episode:08d}.pt')
            if self.save_replay_buffer_on_full and self.buffer._size > 0:
                buffer_snapshot = self.buffer.snapshot()
                checkpoint_data['buffer'] = {
                    'obs':       buffer_snapshot._obs[:buffer_snapshot._size].cpu(),
                    'actions':   buffer_snapshot._actions[:buffer_snapshot._size].cpu(),
                    'rewards':   buffer_snapshot._rewards[:buffer_snapshot._size].cpu(),
                    'dones':     buffer_snapshot._dones[:buffer_snapshot._size].cpu(),
                    'log_probs': buffer_snapshot._log_probs[:buffer_snapshot._size].cpu(),
                    'values':    buffer_snapshot._values[:buffer_snapshot._size].cpu(),
                    'agent_ids': buffer_snapshot._agent_ids[:buffer_snapshot._size].cpu(),
                    'size':      buffer_snapshot._size,
                    'ptr':       buffer_snapshot._ptr,
                }
            self._apply_archival_float16(checkpoint_data)
            full_path = self._save_pt_file(ckpt_file, checkpoint_data)
            if self.rolling_mode == 'compact' and self.max_history_entries > 0:
                self._prune_old_checkpoints()
            print(f'Checkpoint saved: {full_path}')

        if self.save_diagnostics_separately and diag_path is not None:
            print(f'Diagnostics saved: {diag_path}')

    # ------------------------------------------------------------------
    def run(self, is_training=True, render=False):
        raise NotImplementedError(
            "Agent.run() is not supported in this scene-driven setup. "
            "Use Agent.step(obs) from the engine update loop instead."
        )

    # ------------------------------------------------------------------
    def save_graph(self):
        n = len(self.rewards_per_episode)
        if n == 0:
            return

        # Data is per-round (one entry per round, already aggregated across agents).
        roll_window   = 100
        action_window = 100
        rounds = np.arange(1, n + 1)  # 1-based round numbers for x-axis

        def rolling_mean(data, window=100):
            arr = np.array(data, dtype=float)
            if len(arr) == 0:
                return arr
            w = min(window, len(arr))
            cumsum = np.cumsum(np.insert(arr, 0, 0))
            result = (cumsum[w:] - cumsum[:-w]) / w
            # Pad the head with expanding means so length matches input.
            head = np.array([arr[:i+1].mean() for i in range(min(w - 1, len(arr)))])
            return np.concatenate([head, result])

        def _plot_history(ax, data, color, label=None, linestyle='-'):
            if not data:
                return
            arr = np.array(data, dtype=float)
            x = rounds if len(arr) == n else np.arange(1, len(arr) + 1)
            ax.plot(x, arr, alpha=0.3, color=color, linewidth=0.8, label=label)
            ax.plot(x, rolling_mean(arr, roll_window), color=color, linestyle=linestyle)

        fig, axes = plt.subplots(3, 3, figsize=(21, 13))
        fig.suptitle(
            f'PPO Diagnostics — {self.hyperparameter_set}  '
            f'(round {n} | {self.num_agents} agents)',
            fontsize=13)

        # ── (0,0) Reward + Curriculum ──────────────────────────────────────────
        ax = axes[0, 0]
        _plot_history(ax, self.rewards_per_episode, 'tab:blue', label=f'{roll_window}-round mean')
        ax.set_ylabel('Mean Reward', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax.set_xlabel('Round')
        ax.set_title('Reward + Curriculum')
        if self.goal_time_limit_history:
            ax2 = ax.twinx()
            ax2.plot(rounds, self.goal_time_limit_history, color='tab:orange', alpha=0.6)
            ax2.set_ylabel('Time Limit (s)', color='tab:orange', labelpad=10)
            ax2.tick_params(axis='y', labelcolor='tab:orange')
            if self._env is not None:
                cfg = self._env.get_config_data()
                ax2.set_ylim(cfg.get('min_goal_search_seconds', 10.0),
                             cfg.get('max_goal_search_seconds', 60.0))
            else:
                ax2.set_ylim(10.0, 60.0)

        # ── (0,1) Episode Length ────────────────────────────────────────────────
        ax = axes[0, 1]
        _plot_history(ax, self.episode_length_history, 'tab:purple')
        ax.set_ylabel('Steps / Episode')
        ax.set_xlabel('Round')
        ax.set_title('Episode Length')

        # ── (0,2) Goals Reached ─────────────────────────────────────────────────
        ax = axes[0, 2]
        _plot_history(ax, self.goal_reach_history, 'tab:green')
        ax.set_ylabel('Goals / Agent')
        ax.set_xlabel('Round')
        ax.set_title('Goal-Find Rate (per agent)')

        # ── (1,0) Policy Gradient Loss ──────────────────────────────────────────
        ax = axes[1, 0]
        _plot_history(ax, self.pg_loss_history, 'tab:red')
        ax.set_ylabel('PG Loss')
        ax.set_xlabel('Round')
        ax.set_title('Policy Gradient Loss')
        ax.autoscale(axis='y')

        # ── (1,1) Value Function Loss ───────────────────────────────────────────
        ax = axes[1, 1]
        _plot_history(ax, self.value_loss_history, 'tab:blue')
        ax.set_ylabel('Value Loss')
        ax.set_xlabel('Round')
        ax.set_title('Value Function Loss')

        # ── (1,2) Gradient Norm ─────────────────────────────────────────────────
        ax = axes[1, 2]
        if self.gradient_norm_history:
            _plot_history(ax, self.gradient_norm_history, 'tab:brown', label='Pre-clip')
        if self.gradient_norm_post_clip_history:
            _plot_history(ax, self.gradient_norm_post_clip_history, 'tab:olive', label='Post-clip', linestyle='--')
        ax.set_ylabel('Gradient L2 Norm')
        ax.set_xlabel('Round')
        ax.set_title('Gradient Norm (pre / post clip)')
        if self.gradient_norm_history or self.gradient_norm_post_clip_history:
            ax.legend(loc='upper left', fontsize='small')

        # ── (2,0) Entropy ───────────────────────────────────────────────────────
        ax = axes[2, 0]
        _plot_history(ax, self.entropy_history, 'tab:cyan')
        ax.set_ylabel('Policy Entropy')
        ax.set_xlabel('Round')
        ax.set_title('Entropy (exploration)')

        # ── (2,1) Clip Fraction + Approx KL ────────────────────────────────────
        ax = axes[2, 1]
        if self.clip_frac_history:
            cf  = np.array(self.clip_frac_history, dtype=float)
            kl  = np.array(self.approx_kl_history, dtype=float)
            x_cf = rounds if len(cf) == n else np.arange(1, len(cf) + 1)
            x_kl = rounds if len(kl) == n else np.arange(1, len(kl) + 1)
            ax.plot(x_cf, cf, alpha=0.3, color='tab:orange', linewidth=0.8)
            ax.plot(x_cf, rolling_mean(cf,  roll_window), color='tab:orange', label='Clip frac')
            ax2 = ax.twinx()
            # Only plot raw KL clipped to 3× the rolling-mean range to suppress early spikes.
            kl_smooth = rolling_mean(kl, roll_window)
            kl_cap = max(kl_smooth.max() * 3.0, 0.01)
            ax2.plot(x_kl, np.clip(kl, 0, kl_cap), alpha=0.3, color='tab:pink', linewidth=0.8)
            ax2.plot(x_kl, kl_smooth, color='tab:pink',   label='Approx KL', linestyle='--')
            ax.set_ylabel('Clip Fraction', color='tab:orange')
            ax2.set_ylabel('Approx KL',    color='tab:pink')
            ax.tick_params(axis='y', labelcolor='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:pink')
        ax.set_xlabel('Round')
        ax.set_title('Clip Fraction + Approx KL')

        # ── (2,2) Action Distribution ───────────────────────────────────────────
        ax = axes[2, 2]
        if self.action_dist_history:
            recent     = np.array(self.action_dist_history[-action_window:], dtype=float)
            avg_counts = recent.mean(axis=0)
            total      = avg_counts.sum()
            pct        = (avg_counts / total * 100) if total > 0 else avg_counts
            num_a      = len(pct)
            labels     = ACTION_NAMES[:num_a] if num_a <= len(ACTION_NAMES) else [str(i) for i in range(num_a)]
            colors     = plt.cm.tab10(np.linspace(0, 0.9, num_a))
            bars       = ax.bar(labels, pct, color=colors)
            for bar, val in zip(bars, pct):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                        f'{val:.1f}%', ha='center', va='bottom', fontsize=7)
            ax.set_ylabel('Usage (%)')
            ax.set_ylim(0, 100)
        ax.set_title(f'Action Distribution (last {action_window} rounds)')

        plt.tight_layout(rect=[0, 0, 0.97, 1])
        fig.savefig(self.GRAPH_FILE, dpi=100)
        plt.close(fig)


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('hyperparameters', help='Hyperparameter set name (key in hyperparameters.yml)')
    parser.add_argument('--train', action='store_true', help='Training mode')
    args = parser.parse_args()

    agent = Agent(hyperparameter_set=args.hyperparameters)
    if args.train:
        agent.run(is_training=True)
    else:
        agent.run(is_training=False, render=True)
