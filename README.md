# seeker-agent-learn

This repository tracks the learning agent and navigation environment extracted from `Magic-Engine`.

## Contents

- `apps/navigation_env.h` / `apps/navigation_env.cpp`
- `apps/scenes/LearnScene.hpp` / `apps/scenes/LearnScene.cpp`
- `apps/python/navigation_env.cpp`
- `apps/assets/scripts/scenes/learn_scene.py`
- `apps/assets/scripts/learning/*`
- `apps/assets/hyperparameters.yml`
- `apps/assets/learning/config.json`
- `monitor_training.py`
- `detail.md`

## Purpose

This repo is intended as an independent learning-agent workspace that can be versioned separately from the main engine repository.

## Build

The learning agent code depends on `Magic-Engine` for engine headers and runtime integration. Use a CMake variable such as `MagicEngine_DIR` to point to the engine source tree when configuring this repository.

Example:

```sh
cmake -S . -B build -DMagicEngine_DIR="C:/Github/Magic-Engine/apps"
```
