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

The `engine/` directory contains prebuilt binaries for Magic-Engine and is **not stored in this repository**. Download it before building:

```powershell
.\download-engine.ps1
```

This downloads `engine.zip` from the latest GitHub release and extracts it locally.
To force a re-download: `.\download-engine.ps1 -Force`

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
