# Seeker DQN — Hyperparameters & Training Timeline

## Hyperparameters (seeker set)

| Parameter | Value | Rationale |
|---|---|---|
| `replay_memory_size` | 150,000 | Fills in ~30 episodes; keeps experience fresh without waiting 90 episodes for diversity |
| `mini_batch_size` | 64 | More update steps per episode; better for sparse goal rewards |
| `epsilon_init` | 1.0 | Full exploration at start |
| `epsilon_decay` | 0.99998 | ~115k steps to floor |
| `epsilon_limit` | 0.1 | 10% random action floor to prevent total exploitation |
| `learning_rate_a` | 0.0003 | Adam's commonly optimal range; faster early learning without diverging |
| `discount_factor_g` | 0.999 | `0.99^300 ≈ 0.05` (goal credit dies); `0.999^300 ≈ 0.74` — agent can plan across the full search window |
| `stop_on_reward` | 999,999 | Multi-goal episodes can score 200+ when trained; don't stop prematurely |
| `hidden_dims` | [256, 256, 128] | Sufficient capacity for 199 inputs (raycast + ground rays + sighting history + positional + LOS features) |

---

## Observation Space Summary (199 floats)

| Indices | Content |
|---|---|
| 0–12 | Ray hit fractions (13 horizontal rays, 120° FOV, 10° spacing) |
| 13–25 | Interest bias toward goal per ray (cosine alignment; zeroed when goal occluded by wall) |
| 26–181 | Sighting ring buffer: 13 rays × 3 history × (offset_x, offset_y, offset_z, age) |
| 182–186 | Ground / edge-detection ray fractions (5 rays, 45° downward pitch, 120° FOV) — 1.0 = no ground = edge ahead |
| 187–188 | Last-known goal direction in seeker-local space (normalised; holds last seen position when occluded) |
| 189–190 | Seeker linear velocity (xz) |
| 191 | Looking angle normalised to [−1, 1] |
| 192 | Distance to last-known goal normalised |
| 193–194 | Seeker world position normalised |
| 195–196 | Last-known goal world position normalised |
| 197 | Goal currently visible: 1.0 = clear LOS, 0.0 = occluded by wall |
| 198 | Goal sighting staleness normalised [0, 1] over 5 s — 0 = just seen, 1 = not seen for 5+ s |

---

## Episode Termination Conditions

| Condition | Value |
|---|---|
| Max steps | 2,000 |
| Max episode time | 180 s |
| Goal search timeout | 60 s initially, decreasing by 0.003 s per episode down to a floor of 10 s (resets on every goal reach) |

The curriculum shrinks the time limit each episode (`SEARCH_TIME_FALL_RATE = 0.003`), forcing the agent to find goals faster as it improves. The timer resets on every goal reach, so a well-trained agent chains goals as long as each one is found within the current limit.

With **3 simultaneous goals** (`NUM_GOALS = 3`) spread across the map, the agent always has a nearby target visible. A well-trained agent scoring 15–30 goals/episode earns 150–300+ reward.

---

## Estimated Training Timeline

> The logic thread runs **free (unlocked)** during training — steps accumulate as fast as the CPU allows. Estimates are in steps only, not wall-clock time.

| Milestone | Total Steps | Approx. Episodes |
|---|---|---|
| Replay fills enough to learn | ~5,000 | ~8–15 |
| Epsilon hits floor (0.1) | ~115,000 | ~200–300 |
| Agent reliably turns toward visible goal | ~300,000–500,000 | ~500–800 |
| Consistent goal-finding through walls (LOS-aware search) | ~1,200,000–2,500,000 | ~1,500–3,500 |
| Chaining 5+ goals per episode | ~4,000,000–8,000,000 | ~4,000–8,000 |

**Key factors:**
- Free-running simulation: steps accumulate far faster than real-time — the bottleneck is the PyTorch backward pass, not the physics step.
- 199-input observation space is larger than before (added xyz sighting history, goal LOS flag, staleness, and last-known position); expect more steps to converge vs. the old 158-float layout.
- Ground rays give the agent an explicit edge signal, which should eliminate falling-off behaviour well before the agent is otherwise "optimal".
- **Goal LOS gating:** when the goal is behind a wall, interest bias is zeroed and goal inputs show last-known position. The agent must actively search when it loses sight of the goal — significantly increasing exploration difficulty vs. X-ray vision.
- Reward shaping (delta-distance + look-alignment) provides a gradient signal every step, not just on goal reach.
- `gamma=0.999`: agent can credit actions taken ~300 steps before a goal reach.
- 5 walls (tall, 6 units) in the environment significantly increase the exploration requirement — milestones are ~3–5× later than a flat open map.
- Map size is **100×100** (navigable area ±20 in world space before walls); ground is large enough that edge-falling is rare once the agent learns edge detection.
- **3 simultaneous goals** are placed randomly each episode. On goal reach, only the reached goal relocates — the other two remain in place, giving the agent an immediate nearby target and reducing time spent searching after each reach.
- **Curriculum timeout:** goal search limit starts at 60 s and decreases by 0.003 s per episode to a minimum of 10 s. This prevents the agent from settling into lazy slow-search behaviour early in training while still being reachable by a beginner policy.
- `epsilon_decay=0.99998` is intentionally slow; rapid decay caused policy collapse when the environment changed mid-training.

**Note:** Track **total steps**, not episode count. Early episodes are short (timeout-terminated); later episodes grow longer as the policy improves.

---

## Replay Buffer Analysis

| Stat | Value |
|---|---|
| Buffer size | 150,000 transitions |
| Avg. steps/episode (30s timeout) | ~600–900 |
| Episodes of history in buffer | ~200 |
| Steps when epsilon hits floor | ~115,000 (~130–190 episodes) |

**Buffer size is well-matched to the current config.** The buffer fills just as exploitation begins, giving the network a full, diverse batch to learn from at the moment it matters most.

**Why not increase it?**  
A much larger buffer means early random-walk transitions (from epsilon=1.0) persist into the exploitation phase. Those transitions have no structure and add noise to Bellman targets — known as the *experience staleness* problem. 150k already stores ~200 episodes, which is generous.

**What would help more:** Prioritized Experience Replay (PER) — sample transitions with high TD-error more frequently, so rare wall-navigation recoveries get replayed more often. This is a code change to `experience_replay.py`.
