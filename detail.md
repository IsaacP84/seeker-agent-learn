# Seeker PPO — Hyperparameters & Training Timeline

## Algorithm: Proximal Policy Optimization (PPO)

DQN has been replaced with PPO. The C++↔Python interface (`step(obs) → int`, `feedback(prev_obs, action, reward, done, next_obs)`) is unchanged. The agent now uses separate Actor and Critic networks trained on fixed-length rollouts with GAE-Lambda advantage estimation.

**Key files:**
- `assets/scripts/learning/agent_ppo.py` — Agent class (PPO training loop)
- `assets/scripts/learning/ppo_network.py` — Actor and Critic network definitions
- `assets/scripts/learning/trajectory_buffer.py` — On-policy rollout buffer with GAE
- DQN backups: `agent_dqn.py`, `dqn_backup.py`, `experience_replay_dqn.py`

---

## Hyperparameters (seeker set)

| Parameter | Description | Rationale |
|---|---|---|
| `n_steps` | Rollout length before each PPO update | Captures multiple episodes per update |
| `n_epochs` | Number of optimization passes over each rollout batch | More epochs increase stability at some cost |
| `mini_batch_size` | Mini-batch size used within each epoch | Balances gradient variance and throughput |
| `clip_epsilon` | PPO clipping range | Limits policy update magnitude |
| `gae_lambda` | GAE smoothing factor | Balances bias vs. variance in advantage estimates |
| `discount_factor_g` | Discount factor for future rewards | Retains credit over long search horizons |
| `reward_scale` | Scalar applied to rewards | Keeps returns numerically stable |
| `entropy_coeff` | Entropy bonus weight | Keeps exploration alive without dominating loss |
| `value_loss_coeff` | Critic loss weight relative to actor loss | Balances value fitting with policy updates |
| `learning_rate_a` | Initial Adam learning rate | Starting step size for both networks |
| `lr_min` | Minimum learning rate | Floor for the annealing schedule |
| `lr_decay_episodes` | Length of the LR annealing schedule | Controls how quickly LR decays over training |
| `max_grad_norm` | Gradient clipping threshold | Prevents runaway updates |
| `stop_on_reward` | Episode reward threshold for stopping | Avoids premature termination of high-return episodes |
| `hidden_dims` | Hidden layer widths for both networks | Defines the MLP representation capacity |

---

## Network Architecture

Both Actor and Critic are **separate** MLPs (no shared trunk):

```
Input (234) → Linear(256) → ReLU → Linear(256) → ReLU → Linear(128) → ReLU → head
```

- **Actor head:** Linear(128 → 8) → Categorical distribution over 8 actions
- **Critic head:** Linear(128 → 1) → scalar value estimate V(s)
- Single shared Adam optimizer over both networks' parameters

---

## Training Loop

Each C++ physics step calls `step(obs)` then `feedback(...)` once. Internally:

1. `step()` samples from the Actor (stochastic in training, greedy in inference) and caches `log_prob` and `V(s)` for the next `observe()` call.
2. `observe()` appends the transition to the `RolloutBuffer`. When the buffer has accumulated a full PPO rollout:
   - Bootstrap the last value with `V(next_state)` from the Critic
   - Compute GAE-Lambda advantages and discounted returns over the rollout
   - Normalize advantages (zero mean, unit std)
   - Run multiple epochs of mini-batch PPO updates with shuffled batches
   - Clipped surrogate loss + value loss + entropy bonus
   - Clip gradient norm to the configured max threshold
   - Clear the buffer
3. On episode end (done=True), any partial buffer is also flushed with `last_value=0`.

---

## Checkpoint Format

Checkpoints (`runs/seeker.pt`) contain:
- `actor`, `critic` — network state dicts
- `optimizer` — Adam state (momentum, lr)
- All diagnostic histories: `rewards_per_episode`, `pg_loss_history`, `value_loss_history`, `entropy_history`, `clip_frac_history`, `approx_kl_history`, `episode_length_history`, `gradient_norm_history`, `goal_reach_history`, `goal_time_limit_history`, `action_dist_history`
- `env_data` — curriculum state (current goal time limit)

On load, shape mismatches and DQN checkpoints (missing `'actor'` key) are detected and skipped gracefully, starting fresh.

---

| Indices | Content |
|---|---|
| 0–12 | Ray hit fractions (13 horizontal rays, 120° FOV, 10° spacing) |
| 13–25 | Sighting interest per ray: −0.1 = wall hit this frame, 0.9 = goal visible on this ray, 0.0 = nothing |
| 26–181 | Sighting ring buffer: 13 rays × 3 history × (offset_x, offset_y, offset_z, age/max_age) = 156 floats |
| 182–186 | Ground / edge-detection ray fractions (5 rays, 45° downward pitch, 120° FOV) — 1.0 = no ground = edge ahead |
| 187–188 | Last-known goal direction in seeker-local space (normalised; holds last seen position when occluded) |
| 189 | Seeker linear velocity X, normalised by max_speed (10 m/s) |
| 190 | Seeker linear velocity Z, normalised by max_speed |
| 191 | Seeker linear velocity Y, normalised by max_speed |
| 192 | Seeker angular velocity Y (yaw rate), normalised by max_angular_speed (10 rad/s) |
| 193 | sin(looking_angle) — encodes heading without ±180° discontinuity |
| 194 | cos(looking_angle) |
| 195 | Distance to last-known goal, normalised by map diagonal |
| 196–197 | Last-known goal world position (x, z), normalised to [0, 1] |
| 198 | Goal currently visible: 1.0 = clear LOS, 0.0 = occluded by wall |
| 199 | Goal sighting staleness normalised [0, 1] over 5 s — 0 = just seen, 1 = not seen for 5+ s |
| 200 | `in_air` flag (1.0 = airborne, 0.0 = grounded) |
| 201 | `is_jumping` flag (1.0 = jump impulse active) |
| 202–233 | Previous action history one-hot (4 steps × 8 actions; all zero at episode start) |

**Actions (8):**

| Index | Action |
|---|---|
| 0 | MOVE_FORWARD |
| 1 | MOVE_BACKWARD |
| 2 | STRAFE_LEFT |
| 3 | STRAFE_RIGHT |
| 4 | TURN_LEFT |
| 5 | TURN_RIGHT |
| 6 | JUMP |
| 7 | IDLE |

---

## Reward Function

| Signal | Value | Notes |
|---|---|---|
| Goal reached | +10.0 | Goal relocates randomly; episode continues |
| Goal speed bonus | +2.0 | Added to goal reward if goal reached within 15 s |
| Delta distance | shaping | `prev_dist − curr_dist` per step; positive when moving closer |
| Look alignment | +0.003 × cos(angle_diff) | Encourages facing the goal |
| Forward reward | +0.002 / step | Bonus per step for MOVE_FORWARD while grounded |
| Action penalty | −0.01 / step | Constant cost; incentivises efficient routes |
| Strafe penalty | −0.02 / step | Applied on top of action penalty when strafing |
| Stuck penalty | −0.05 / step | Applied when moving but speed < 0.5 m/s (stuck against wall) |
| Edge danger | up to −0.3 / step | Linear ramp over 3 world units from boundary; stacks with goal rewards |
| Fall off map | −50 − remaining_steps × 0.01 | Early termination is strictly worse than late; forfeits future action penalties |

| Condition | Value |
|---|---|
| Max steps | 100,000 |
| Max episode time | 300 s |
| Goal search timeout | Starts at the configured maximum and decays toward a lower floor each episode; resets on every goal reach |

The curriculum shrinks the time limit each episode by the configured fall rate, forcing the agent to find goals faster as it improves. The timer resets on every goal reach, so a well-trained agent chains goals as long as each one is found within the current limit.

With **3 simultaneous goals** (`NUM_GOALS = 3`) spread across the map, the agent always has a nearby target visible. A well-trained agent scoring 15–30 goals/episode earns 150–300+ reward.

---

## Estimated Training Timeline

> The logic thread runs **free (unlocked)** during training — steps accumulate as fast as the CPU allows. Estimates are in steps only, not wall-clock time.
> PPO updates every configured rollout length, so "updates" = total steps ÷ rollout length.

| Milestone | Total Steps | Approx. Updates | Approx. Episodes |
|---|---|---|---|
| Agent learns basic movement direction | ~50,000–150,000 | ~25–75 | ~100–300 |
| Consistent turning toward visible goal | ~300,000–600,000 | ~150–300 | ~500–1,000 |
| Goal-finding through walls (LOS-aware search) | ~1,000,000–2,000,000 | ~500–1,000 | ~1,500–3,000 |
| Chaining 5+ goals per episode | ~3,000,000–6,000,000 | ~1,500–3,000 | ~4,000–7,000 |

**Key factors:**
- Free-running simulation: steps accumulate far faster than real-time — the bottleneck is the PyTorch backward pass, with multiple epochs and moderate-sized mini-batches per rollout.
- PPO is generally more sample-efficient than DQN for this task because on-policy updates avoid the experience-staleness problem entirely.
- **234-input observation space** as of current implementation. Key additions over legacy DQN layout: sin/cos heading encoding, normalised 3-axis velocity + angular velocity, `in_air`/`is_jumping` flags, previous action history one-hot (4 steps × 8 actions), and sighting ring buffer.
- **Reward scaling:** all rewards are multiplied by `reward_scale = 0.1` before entering the buffer, keeping discounted returns in a unit-ish range without per-batch normalisation.
- **LR schedule:** cosine annealing from the configured initial LR to the configured minimum LR over the scheduled decay window.
- Ground rays give an explicit edge signal, eliminating falling-off behaviour early.
- **Goal LOS gating:** when the goal is behind a wall, interest bias is zeroed and goal inputs show last-known position. The agent must actively search when it loses sight of the goal.
- Reward shaping (delta-distance + look-alignment) provides a dense gradient signal every step, not just on goal reach.
- `gamma` is set close to 1 so the agent can credit actions taken long before a goal reach over the episode horizon.
- 5 walls (tall, 6 units) significantly increase the exploration requirement.
- Map size is **100×100** (navigable area ±20 in world space); edge-falling is rare once the agent learns edge detection.
- **3 simultaneous goals** placed randomly each episode. On goal reach, only the reached goal relocates — the others remain, giving an immediate nearby target.
- **Curriculum timeout:** goal search limit starts at the configured maximum and decreases by the configured fall rate each episode. Resets on every goal reach.

**Note:** Track **total steps**, not episode count. Early episodes are short (timeout-terminated); later episodes grow longer as the policy improves.

---

## Diagnostic Graph Panels (3×3)

| Position | Panel |
|---|---|
| (0,0) | Reward + Curriculum (100-ep mean reward, twin axis: goal time limit) |
| (0,1) | Episode Length |
| (0,2) | Goals Reached per Episode |
| (1,0) | Policy Gradient (PG) Loss |
| (1,1) | Value Function Loss |
| (1,2) | Gradient Norm |
| (2,0) | Policy Entropy |
| (2,1) | Clip Fraction + Approx KL (dual axis) |
| (2,2) | Action Distribution (last 100 episodes avg) |
