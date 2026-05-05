import torch
import numpy as np


class RolloutBuffer:
    """Fixed-size on-policy rollout buffer for PPO.

    Accumulates transitions step-by-step. When full (or when the caller
    signals end-of-episode with ``flush=True``), ``compute_advantages``
    is called to produce GAE-Lambda advantages and discounted returns,
    then ``get_batches`` yields shuffled mini-batches for the PPO update.

    Parameters
    ----------
    n_steps  : int   Maximum number of steps before forcing an update.
    device   : str   Torch device string ('cpu' or 'cuda').
    """

    def __init__(self, n_steps, device='cpu'):
        self.n_steps = n_steps
        self.device = device
        self._ptr = 0
        self._size = 0

        # Storage — pre-allocated after the first ``add`` call so we don't
        # need state_dim at construction time.
        self._obs = None
        self._actions = None
        self._rewards = None
        self._dones = None
        self._log_probs = None
        self._values = None
        self._agent_ids = None  # per-slot agent id; enables per-agent GAE

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, obs, action, reward, done, log_prob, value, agent_id=0):
        """Store a single transition.

        Parameters
        ----------
        obs      : array-like (state_dim,)
        action   : int
        reward   : float
        done     : bool  — True if the episode ended on this step
        log_prob : float — log π(a|s) at collection time
        value    : float — V(s) at collection time
        agent_id : int   — which agent this transition belongs to
        """
        if self._obs is None:
            state_dim = len(obs)
            self._obs       = torch.zeros((self.n_steps, state_dim), dtype=torch.float32, device=self.device)
            self._actions   = torch.zeros(self.n_steps, dtype=torch.long,    device=self.device)
            self._rewards   = torch.zeros(self.n_steps, dtype=torch.float32, device=self.device)
            self._dones     = torch.zeros(self.n_steps, dtype=torch.float32, device=self.device)
            self._log_probs = torch.zeros(self.n_steps, dtype=torch.float32, device=self.device)
            self._values    = torch.zeros(self.n_steps, dtype=torch.float32, device=self.device)
            self._agent_ids = torch.zeros(self.n_steps, dtype=torch.long,    device=self.device)

        i = self._ptr
        self._obs[i]        = torch.as_tensor(obs,      dtype=torch.float32, device=self.device)
        self._actions[i]    = int(action)
        self._rewards[i]    = float(reward)
        self._dones[i]      = float(done)
        self._log_probs[i]  = float(log_prob)
        self._values[i]     = float(value)
        self._agent_ids[i]  = int(agent_id)

        self._ptr  = (self._ptr + 1) % self.n_steps
        self._size = min(self._size + 1, self.n_steps)

    def is_full(self):
        return self._size >= self.n_steps

    def clear(self):
        self._ptr  = 0
        self._size = 0

    def snapshot(self):
        """Return a deep copy of this buffer's current data.

        The copy owns its own tensors so the worker thread can read it
        safely while the main thread continues writing to self.
        """
        snap = RolloutBuffer(self.n_steps, device=self.device)
        snap._ptr  = self._ptr
        snap._size = self._size
        if self._obs is not None:
            n = self._size
            snap._obs       = self._obs[:n].clone()
            snap._actions   = self._actions[:n].clone()
            snap._rewards   = self._rewards[:n].clone()
            snap._dones     = self._dones[:n].clone()
            snap._log_probs = self._log_probs[:n].clone()
            snap._values    = self._values[:n].clone()
            snap._agent_ids = self._agent_ids[:n].clone() if self._agent_ids is not None else None
            # Resize internal storage to match snapshot size so indexing is consistent.
            snap.n_steps = n
            snap._ptr    = 0   # snapshot is treated as full / read-only
        return snap

    def compute_advantages(self, last_value, gamma, gae_lambda):
        """Compute GAE-Lambda advantages and discounted returns.

        Parameters
        ----------
        last_value : float  V(s_{t+1}) for the step *after* the buffer ends.
                            Pass 0.0 when the episode ended with a terminal
                            state (e.g. fell off map); pass critic(next_obs)
                            when the episode was truncated by a step/time limit.
        gamma      : float  Discount factor (e.g. 0.999).
        gae_lambda : float  GAE smoothing (e.g. 0.95).

        Returns
        -------
        advantages : Tensor (n,)
        returns    : Tensor (n,)  — targets for the value function
        """
        n = self._size
        advantages = torch.zeros(n, dtype=torch.float32, device=self.device)

        if self._agent_ids is not None:
            # Multi-agent: compute GAE independently per agent so that V(s) from
            # one agent never contaminates another agent's advantage estimate.
            #
            # For interleaved buffers (e.g. [a0,a1,a2,a3,a0,a1,...]) same-agent
            # indices are spaced num_agents apart — NOT contiguous.  We must NOT
            # split on index gaps.  Instead we run one backward GAE pass over ALL
            # of the agent's buffer positions in order.  Episode boundaries within
            # the sequence are handled by the standard nnt=1-done mask (same as
            # single-agent) — nnt=0 zeroes the carry when an episode ends so the
            # next episode starts fresh.  V(s_{t+1}) for each step is the stored
            # critic value at the same agent's NEXT buffer slot, which is exactly
            # the observation the agent saw one frame later.
            agent_ids_np = self._agent_ids[:n].cpu().numpy()
            for agent_id in np.unique(agent_ids_np):
                idx = np.where(agent_ids_np == agent_id)[0]  # sorted positions
                K   = len(idx)

                # Bootstrap for the very last step of this agent in the buffer.
                t_last = int(idx[-1])
                if self._dones[t_last].item() > 0.5:
                    end_bootstrap = 0.0          # true terminal
                elif t_last == n - 1:
                    end_bootstrap = float(last_value)  # this agent triggered the flush
                else:
                    # Agent's last slot is mid-buffer; next same-agent step would
                    # be in the next rollout.  Use stored V(s) as approximation —
                    # this affects only 1 of ~(n/num_agents) steps per rollout.
                    end_bootstrap = self._values[t_last].item()

                # Single backward pass — gae carry propagates across the full
                # agent trajectory, resetting at episode boundaries via nnt=0.
                gae = 0.0
                for k in reversed(range(K)):
                    t      = int(idx[k])
                    nnt    = 1.0 - self._dones[t].item()
                    v_next = (self._values[int(idx[k + 1])].item()
                              if k < K - 1 else end_bootstrap)
                    delta  = (self._rewards[t].item()
                              + gamma * v_next * nnt
                              - self._values[t].item())
                    gae           = delta + gamma * gae_lambda * nnt * gae
                    advantages[t] = gae
        else:
            # Single-agent: original flat backward pass.
            gae      = 0.0
            next_val = float(last_value)
            for t in reversed(range(n)):
                nnt   = 1.0 - self._dones[t].item()
                delta = (self._rewards[t].item()
                         + gamma * next_val * nnt
                         - self._values[t].item())
                gae          = delta + gamma * gae_lambda * nnt * gae
                advantages[t] = gae
                next_val     = self._values[t].item()

        returns = advantages + self._values[:n]
        return advantages, returns

    def get_batches(self, mini_batch_size):
        """Yield shuffled mini-batches of the current rollout.

        Yields
        ------
        dict with keys: obs, actions, log_probs, values, advantages, returns
        Each value is a Tensor of shape (mini_batch_size, ...) — the last
        batch may be smaller if n is not divisible.
        """
        n = self._size
        indices = torch.randperm(n, device=self.device)

        # These are passed in from compute_advantages; store on the object
        # temporarily so get_batches can access them.
        adv = self._advantages
        ret = self._returns

        for start in range(0, n, mini_batch_size):
            idx = indices[start: start + mini_batch_size]
            yield {
                'obs':       self._obs[:n][idx],
                'actions':   self._actions[:n][idx],
                'log_probs': self._log_probs[:n][idx],
                'values':    self._values[:n][idx],
                'advantages':adv[idx],
                'returns':   ret[idx],
            }

    def prepare(self, last_value, gamma, gae_lambda):
        """Compute advantages/returns and store them ready for ``get_batches``.

        Call this once before iterating ``get_batches``.

        Returns
        -------
        advantages, returns  (also stored internally)
        """
        adv, ret = self.compute_advantages(last_value, gamma, gae_lambda)
        self._advantages = adv
        self._returns    = ret
        return adv, ret
