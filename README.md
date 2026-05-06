# seeker-agent-learn

A reinforcement learning agent that learns navigation using the [Magic-Engine](https://github.com/IsaacP84/Magic-Engine) runtime.

## Structure

| Path | Description |
|------|-------------|
| `src/` | App entry point, scenes, entities, navigation env, Python bindings |
| `src/scenes/LearnScene.*` | Main RL training scene |
| `src/navigation_env.*` | Navigation environment (step/reset/reward logic) |
| `src/entities/` | Seeker entity definition |
| `assets/scripts/learning/` | Python training scripts |
| `assets/hyperparameters.yml` | Hyperparameter config |
| `monitor_training.py` | Live training monitor |
| `download-engine.ps1` | Downloads prebuilt engine binaries from GitHub release |
| `engine/` | Prebuilt engine binaries (not in git — see Setup below) |

## Setup

The `engine/` directory contains prebuilt binaries for Magic-Engine.

## Build

Requires CMake 3.20+, a MinGW/UCRT64 toolchain, and the `engine/` folder populated via the script above.

```powershell
cmake -S . -B build --preset <your-preset>
# or manually:
cmake -S . -B build
cmake --build build
```

The engine package is resolved automatically from `engine/lib/cmake/Magic`.

## Training

```powershell
python monitor_training.py
```

## Diagnostic Chart (`runs/seeker.png`)

The chart is saved every round. Each subplot is indexed by **round number** (1-based), where one round = all agents completing one episode.

### Row 1 — Performance

| Panel | What it shows | What to look for |
|-------|---------------|-----------------|
| **Reward + Curriculum** | Mean episode reward across agents (blue, left axis). Per-goal time limit in seconds (orange, right axis). | Reward trending upward. Time limit decaying toward floor (10 s) as the agent improves. Sudden drops = policy regression. |
| **Episode Length** | Mean steps per episode across agents. | Should grow as agents find more goals and survive longer. A plateau followed by decline = policy degradation. |
| **Goal-Find Rate** | Mean goals collected per agent per episode. | Primary success metric. Rising = improving navigation. |

### Row 2 — Training Health

| Panel | What it shows | What to look for |
|-------|---------------|-----------------|
| **Policy Gradient Loss** | PPO clipped surrogate loss per mini-batch, averaged over the round. | Should be small (near 0). Large spikes = noisy gradient or bad batch. Sustained negative = entropy-dominated loss. |
| **Value Function Loss** | MSE between critic predictions and GAE returns. | Should decrease over time. Large values mean the critic is miscalibrated; advantages will be noisy. |
| **Gradient Norm** | Pre-clip L2 norm of the combined actor+critic gradient. `max_grad_norm = 1.0`. | Values consistently above 1.0 = clipping is always active; LR may be too high. A rising trend after a performance peak is a warning sign. |

### Row 3 — Exploration & Distribution

| Panel | What it shows | What to look for |
|-------|---------------|-----------------|
| **Entropy** | Mean policy entropy (nats). Higher = more exploratory. | Should start high (~2.1 for 8 actions = ln(8)) and decay gradually. A sudden collapse below ~1.0 means the policy has become near-deterministic prematurely. |
| **Clip Fraction + Approx KL** | Orange (left): fraction of mini-batch samples that hit the PPO clip boundary. Pink dashed (right): approximate KL divergence from old policy. | Clip fraction above 0.3 and KL above 0.02 = updates are too large; consider reducing LR or `clip_epsilon`. |
| **Action Distribution** | Average action usage % over the last 100 rounds. 8 actions: Forward, Backward, Strafe L, Strafe R, Turn L, Turn R, Jump, Idle. | Jump or Idle dominating (>25%) usually indicates reward exploitation. Expect Forward/Turn to dominate in a healthy navigation policy. |

### Key hyperparameters (see `assets/hyperparameters.yml`)

| Parameter | Value | Effect |
|-----------|-------|--------|
| `n_steps` | 4096 | Shared rollout buffer size before a PPO update |
| `n_epochs` | 10 | Gradient passes per rollout |
| `mini_batch_size` | 256 | Samples per gradient step |
| `clip_epsilon` | 0.1 | PPO trust-region clip |
| `gae_lambda` | 0.95 | GAE smoothing |
| `discount_factor_g` | 0.999 | Long-horizon discount |
| `learning_rate_a` | 0.00005 | Initial Adam LR |
| `lr_min` | 0.00003 | Cosine annealing floor |
| `lr_decay_episodes` | 5000 | Episodes over which LR decays |
| `entropy_coeff` | 0.01 | Entropy bonus (auto-boosted if entropy collapses) |
| `value_loss_coeff` | 1.0 | Critic loss weight |
| `max_grad_norm` | 1.0 | Gradient clip ceiling |
