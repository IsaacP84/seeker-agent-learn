import sys
print(sys.prefix)
print(sys.path)

import os
import shutil
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

ACTION_NAMES = ['Forward', 'Backward', 'Strafe L', 'Strafe R', 'Turn L', 'Turn R', 'Jump']

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


def find_latest_checkpoint(model_file=None):
    pt_files = [
        os.path.join(RUNS_DIR, f)
        for f in os.listdir(RUNS_DIR)
        if f.endswith('.pt')
    ] if os.path.isdir(RUNS_DIR) else []
    if pt_files:
        return max(pt_files, key=os.path.getmtime)
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

        raw_hidden = hyperparameters.get("hidden_dims", 256)
        if isinstance(raw_hidden, int):
            raw_hidden = [raw_hidden]
        self.hidden_dims = raw_hidden

        self.env_make_params = hyperparameters.get("env_make_params", {})

        self.optimizer = None
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    # ------------------------------------------------------------------
    def setup(self, num_states, num_actions, is_training=True, load_model_file=None):
        self.num_states  = num_states
        self.num_actions = num_actions
        self.is_training = is_training

        self.actor  = Actor( num_states, num_actions, self.hidden_dims).to(device)
        self.critic = Critic(num_states,              self.hidden_dims).to(device)

        self.optimizer = torch.optim.Adam(
            list(self.actor.parameters()) + list(self.critic.parameters()),
            lr=self.learning_rate_a,
        )

        if self.lr_decay_episodes > 0:
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.lr_decay_episodes,
                eta_min=self.lr_min,
            )
        else:
            self.scheduler = None

        self.buffer = RolloutBuffer(self.n_steps, device=device)

        self.best_reward             = float('-inf')
        self.rewards_per_episode     = []
        self.goal_time_limit_history = []
        self._episode_reward         = 0.0
        self._env                    = None
        self._loaded_env_data        = None

        self._last_log_prob = 0.0
        self._last_value    = 0.0
        self._last_obs      = None

        # Per-episode accumulators
        self._episode_steps        = 0
        self._action_counts        = [0] * num_actions
        self._goal_count           = 0
        self._episode_pg_losses    = []
        self._episode_value_losses = []
        self._episode_entropies    = []
        self._episode_clip_fracs   = []
        self._episode_approx_kls   = []
        self._gradient_norms       = []
        self._obs_stats_samples    = []

        # Persistent diagnostic histories
        self.action_dist_history    = []
        self.pg_loss_history        = []
        self.value_loss_history     = []
        self.entropy_history        = []
        self.clip_frac_history      = []
        self.approx_kl_history      = []
        self.episode_length_history = []
        self.gradient_norm_history  = []
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
                checkpoint = torch.load(latest_checkpoint, map_location=device)

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
                    if is_training:
                        if 'optimizer' in checkpoint:
                            self.optimizer.load_state_dict(checkpoint['optimizer'])
                        if 'scheduler' in checkpoint and self.scheduler is not None:
                            self.scheduler.load_state_dict(checkpoint['scheduler'])
                        for key in ('rewards_per_episode', 'goal_time_limit_history',
                                    'action_dist_history', 'pg_loss_history',
                                    'value_loss_history', 'entropy_history',
                                    'clip_frac_history', 'approx_kl_history',
                                    'episode_length_history', 'gradient_norm_history',
                                    'goal_reach_history'):
                            if key in checkpoint:
                                setattr(self, key, checkpoint[key])
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
    def step(self, obs):
        """Select an action. Training: stochastic sample. Inference: greedy."""
        if not hasattr(self, 'actor'):
            raise RuntimeError('Agent not initialized. Call setup() before step().')

        state = torch.tensor(obs, dtype=torch.float32, device=device)

        with torch.no_grad():
            if self.is_training:
                action, log_prob, _ = self.actor.get_action(state.unsqueeze(0))
                value = self.critic(state.unsqueeze(0)).squeeze()
                self._last_log_prob = log_prob.item()
                self._last_value    = value.item()
            else:
                logits = self.actor(state.unsqueeze(0)).squeeze()
                action = logits.argmax()
                self._last_log_prob = 0.0
                self._last_value    = 0.0

        self._last_obs = obs
        return int(action.item())

    # ------------------------------------------------------------------
    def observe(self, state, action, reward, next_state, done):
        """Record a transition; trigger PPO update when the rollout buffer is ready."""
        if not hasattr(self, 'buffer'):
            raise RuntimeError('Agent not initialized. Call setup() before observe().')

        reward_f = float(reward)
        done_b   = bool(done)

        # Per-step diagnostics
        self._episode_reward += reward_f
        self._episode_steps  += 1
        self._action_counts[int(action)] += 1
        if reward_f > 9.0:
            self._goal_count += 1
        if self._episode_steps % 10 == 0:
            obs_arr = np.asarray(state, dtype=np.float32)
            self._obs_stats_samples.append(
                (float(obs_arr.mean()), float(obs_arr.std()),
                 float(obs_arr.min()),  float(obs_arr.max()))
            )
            if obs_arr.std() < 1e-6:
                print(f'[OBS WARNING] Step {self._episode_steps}: obs near-constant (std={obs_arr.std():.2e})')

        if not self.is_training:
            return

        self.buffer.add(
            obs      = state,
            action   = action,
            reward   = reward_f,
            done     = done_b,
            log_prob = self._last_log_prob,
            value    = self._last_value,
        )

        # Flush when buffer is full (rollout complete — truncated bootstrap)
        _flushed_full = False
        if self.buffer.is_full():
            with torch.no_grad():
                ns_t     = torch.tensor(next_state, dtype=torch.float32, device=device)
                last_val = self.critic(ns_t.unsqueeze(0)).squeeze().item()
            self._run_ppo_update(last_val)
            _flushed_full = True

        # Also flush on episode end if buffer has partial data (but not if we
        # just did a full-rollout flush above — avoids a double update).
        if done_b and not _flushed_full and self.buffer._size > 0:
            self._run_ppo_update(last_value=0.0)

        if done_b:
            self.rewards_per_episode.append(self._episode_reward)
            self._episode_reward = 0.0

            current_limit = (
                self._env.get_env_data()['current_goal_time_limit']
                if self._env is not None else 0.0
            )
            self.goal_time_limit_history.append(current_limit)

            self.action_dist_history.append(list(self._action_counts))
            self.episode_length_history.append(self._episode_steps)
            self.goal_reach_history.append(self._goal_count)
            self.pg_loss_history.append(
                float(np.mean(self._episode_pg_losses))    if self._episode_pg_losses    else 0.0)
            self.value_loss_history.append(
                float(np.mean(self._episode_value_losses)) if self._episode_value_losses else 0.0)
            self.entropy_history.append(
                float(np.mean(self._episode_entropies))    if self._episode_entropies    else 0.0)
            self.clip_frac_history.append(
                float(np.mean(self._episode_clip_fracs))   if self._episode_clip_fracs   else 0.0)
            self.approx_kl_history.append(
                float(np.mean(self._episode_approx_kls))   if self._episode_approx_kls   else 0.0)
            mean_grad = float(np.mean(self._gradient_norms)) if self._gradient_norms else 0.0
            self.gradient_norm_history.append(mean_grad)

            episode = len(self.rewards_per_episode)
            msg = (
                f'Episode {episode:4d} | '
                f'reward={self.rewards_per_episode[-1]:8.2f} | '
                f'goal_limit={current_limit:.1f}s | '
                f'steps={self._episode_steps:4d} | '
                f'goals={self._goal_count} | '
                f'pg={self.pg_loss_history[-1]:.4f} | '
                f'vf={self.value_loss_history[-1]:.4f} | '
                f'ent={self.entropy_history[-1]:.4f} | '
                f'kl={self.approx_kl_history[-1]:.4f} | '
                f'clip={self.clip_frac_history[-1]:.3f} | '
                f'grad={mean_grad:.3f}'
            )
            log_message(self.LOG_FILE, msg)

            if self._obs_stats_samples:
                stds = [s[1] for s in self._obs_stats_samples]
                if np.mean(stds) < 1e-5:
                    log_message(self.LOG_FILE,
                        f'  [OBS WARNING] Episode {episode}: very low obs std ({np.mean(stds):.2e})')

            # Reset per-episode accumulators
            self._action_counts        = [0] * self.num_actions
            self._episode_steps        = 0
            self._goal_count           = 0
            self._episode_pg_losses    = []
            self._episode_value_losses = []
            self._episode_entropies    = []
            self._episode_clip_fracs   = []
            self._episode_approx_kls   = []
            self._gradient_norms       = []
            self._obs_stats_samples    = []

            if self.scheduler is not None:
                self.scheduler.step()

            if episode % 10 == 0:
                self.save_graph()
                self._save_checkpoint()

    # ------------------------------------------------------------------
    def _run_ppo_update(self, last_value):
        """Run PPO update epochs over the current rollout buffer."""
        adv, ret = self.buffer.prepare(last_value, self.discount_factor_g, self.gae_lambda)

        # Normalize advantages
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        for _ in range(self.n_epochs):
            for batch in self.buffer.get_batches(self.mini_batch_size):
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

                value_loss   = torch.nn.functional.mse_loss(value, ret_b)
                entropy_loss = -entropy.mean()

                loss = pg_loss + self.value_loss_coeff * value_loss + self.entropy_coeff * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.actor.parameters()) + list(self.critic.parameters()),
                    self.max_grad_norm
                )
                self.optimizer.step()

                with torch.no_grad():
                    clip_frac  = ((ratio - 1.0).abs() > self.clip_epsilon).float().mean().item()
                    approx_kl  = (old_lp_b - new_log_prob).mean().item()
                    total_norm = sum(
                        p.grad.data.norm(2).item() ** 2
                        for p in list(self.actor.parameters()) + list(self.critic.parameters())
                        if p.grad is not None
                    ) ** 0.5

                self._episode_pg_losses.append(pg_loss.item())
                self._episode_value_losses.append(value_loss.item())
                self._episode_entropies.append(-entropy_loss.item())
                self._episode_clip_fracs.append(clip_frac)
                self._episode_approx_kls.append(approx_kl)
                self._gradient_norms.append(total_norm)

        self.buffer.clear()

    # ------------------------------------------------------------------
    def _save_checkpoint(self):
        episode = len(self.rewards_per_episode)
        checkpoint_data = {
            'actor':                  self.actor.state_dict(),
            'critic':                 self.critic.state_dict(),
            'optimizer':              self.optimizer.state_dict(),
            **({'scheduler': self.scheduler.state_dict()} if self.scheduler is not None else {}),
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
            'goal_reach_history':     self.goal_reach_history,
        }
        if self._env is not None:
            checkpoint_data['env_data'] = self._env.get_env_data()

        torch.save(checkpoint_data, self.MODEL_FILE)

        if episode % 100 == 0:
            ckpt_file = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}_ep{episode:08d}.pt')
            torch.save(checkpoint_data, ckpt_file)
            print(f'Checkpoint saved: {ckpt_file}')

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

        def rolling_mean(data, window=100):
            arr = np.array(data, dtype=float)
            result = np.zeros(len(arr))
            for i in range(len(arr)):
                result[i] = arr[max(0, i - window + 1):i + 1].mean()
            return result

        fig, axes = plt.subplots(3, 3, figsize=(21, 13))
        fig.suptitle(f'PPO Diagnostics — {self.hyperparameter_set}  (ep {n})', fontsize=13)

        # ── (0,0) Reward + Curriculum ──────────────────────────────────────────
        ax = axes[0, 0]
        ax.plot(rolling_mean(self.rewards_per_episode), color='tab:blue', label='100-ep mean')
        ax.set_ylabel('Mean Reward', color='tab:blue')
        ax.tick_params(axis='y', labelcolor='tab:blue')
        ax.set_xlabel('Episode')
        ax.set_title('Reward + Curriculum')
        if self.goal_time_limit_history:
            ax2 = ax.twinx()
            offset = n - len(self.goal_time_limit_history)
            ax2.plot(range(offset, offset + len(self.goal_time_limit_history)),
                     self.goal_time_limit_history, color='tab:orange', alpha=0.6)
            ax2.set_ylabel('Goal Time Limit (s)', color='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:orange')

        # ── (0,1) Episode Length ────────────────────────────────────────────────
        ax = axes[0, 1]
        if self.episode_length_history:
            lengths = np.array(self.episode_length_history, dtype=float)
            ax.plot(lengths, alpha=0.3, color='tab:purple', linewidth=0.8)
            ax.plot(rolling_mean(lengths), color='tab:purple')
        ax.set_ylabel('Steps / Episode')
        ax.set_xlabel('Episode')
        ax.set_title('Episode Length')

        # ── (0,2) Goals Reached ─────────────────────────────────────────────────
        ax = axes[0, 2]
        if self.goal_reach_history:
            goals = np.array(self.goal_reach_history, dtype=float)
            ax.plot(goals, alpha=0.3, color='tab:green', linewidth=0.8)
            ax.plot(rolling_mean(goals), color='tab:green')
        ax.set_ylabel('Goals Reached')
        ax.set_xlabel('Episode')
        ax.set_title('Goal-Find Rate')

        # ── (1,0) Policy Gradient Loss ──────────────────────────────────────────
        ax = axes[1, 0]
        if self.pg_loss_history:
            pg = np.array(self.pg_loss_history, dtype=float)
            ax.plot(pg, alpha=0.3, color='tab:red', linewidth=0.8)
            ax.plot(rolling_mean(pg), color='tab:red')
        ax.set_ylabel('PG Loss')
        ax.set_xlabel('Episode')
        ax.set_title('Policy Gradient Loss')

        # ── (1,1) Value Function Loss ───────────────────────────────────────────
        ax = axes[1, 1]
        if self.value_loss_history:
            vf = np.array(self.value_loss_history, dtype=float)
            ax.plot(vf, alpha=0.3, color='tab:blue', linewidth=0.8)
            ax.plot(rolling_mean(vf), color='tab:blue')
        ax.set_ylabel('Value Loss')
        ax.set_xlabel('Episode')
        ax.set_title('Value Function Loss')

        # ── (1,2) Gradient Norm ─────────────────────────────────────────────────
        ax = axes[1, 2]
        if self.gradient_norm_history:
            gnorm = np.array(self.gradient_norm_history, dtype=float)
            ax.plot(gnorm, alpha=0.3, color='tab:brown', linewidth=0.8)
            ax.plot(rolling_mean(gnorm), color='tab:brown')
        ax.set_ylabel('Gradient L2 Norm')
        ax.set_xlabel('Episode')
        ax.set_title('Gradient Norm')

        # ── (2,0) Entropy ───────────────────────────────────────────────────────
        ax = axes[2, 0]
        if self.entropy_history:
            ent = np.array(self.entropy_history, dtype=float)
            ax.plot(ent, alpha=0.3, color='tab:cyan', linewidth=0.8)
            ax.plot(rolling_mean(ent), color='tab:cyan')
        ax.set_ylabel('Policy Entropy')
        ax.set_xlabel('Episode')
        ax.set_title('Entropy (exploration)')

        # ── (2,1) Clip Fraction + Approx KL ────────────────────────────────────
        ax = axes[2, 1]
        if self.clip_frac_history:
            cf  = np.array(self.clip_frac_history, dtype=float)
            kl  = np.array(self.approx_kl_history, dtype=float)
            ax.plot(rolling_mean(cf),  color='tab:orange', label='Clip frac')
            ax2 = ax.twinx()
            ax2.plot(rolling_mean(kl), color='tab:pink',   label='Approx KL', linestyle='--')
            ax.set_ylabel('Clip Fraction', color='tab:orange')
            ax2.set_ylabel('Approx KL',    color='tab:pink')
            ax.tick_params(axis='y', labelcolor='tab:orange')
            ax2.tick_params(axis='y', labelcolor='tab:pink')
        ax.set_xlabel('Episode')
        ax.set_title('Clip Fraction + Approx KL')

        # ── (2,2) Action Distribution ───────────────────────────────────────────
        ax = axes[2, 2]
        if self.action_dist_history:
            recent     = np.array(self.action_dist_history[-100:], dtype=float)
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
        ax.set_title('Action Distribution (last 100 eps)')

        plt.tight_layout()
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
