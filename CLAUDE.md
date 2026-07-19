# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A cart-pole reinforcement-learning demo built directly on the Gazebo Sim (gz-sim 8 / Harmonic) Python API, trained with Stable Baselines3 PPO. Despite the repo name, no ROS 2 APIs are used — the sim is driven through `gz.sim8` bindings and `gz.transport13` topics. The whole project is one script, `cart_pole_env.py`, plus the world definition `cart_pole.sdf`.

## Environment & Running

Dependencies are split across two places, and both are required:

- **uv-managed venv** (`.venv`, Python 3.12, no system site-packages): `stable-baselines3[extra]`, gymnasium, torch. Install with `uv sync`.
- **System dist-packages** (`/usr/lib/python3/dist-packages`): the Gazebo Python bindings (`gz.sim8`, `gz.common5`, `gz.math7`, `gz.transport13`, `gz.msgs10`), installed via the `gz-harmonic` apt packages on Ubuntu Noble. `.vscode/settings.json` adds this path for Pylance only — it is **not** on the venv's import path at runtime.

So running the script requires exposing the system bindings to the venv:

```bash
PYTHONPATH=/usr/lib/python3/dist-packages uv run cart_pole_env.py
```

Importing the script has side effects: it immediately trains PPO for 25,000 timesteps (headless), saves `cart_pole_ppo.zip`, then launches a Gazebo server + GUI and runs inference until Ctrl+C. There are no tests or linters configured.

`pyproject.toml` overrides opencv to `opencv-python-headless` (via `[tool.uv] override-dependencies`) so that `stable-baselines3[extra]` doesn't pull in GUI opencv — keep that override when touching dependencies.

## Architecture

`cart_pole_env.py` has two distinct sim-interaction paths that must not be confused:

1. **Training (headless, in-process)** — `GzRewardScorer` loads `cart_pole.sdf` through `gz.sim8.TestFixture` and hooks `on_pre_update` (applies ±2000 N world force to the chassis based on the pending action) and `on_post_update` (reads pole pitch / cart position from the ECM, computes reward and termination). Each Gym `step()` runs the server for 5 blocking sim iterations (physics step is 1 ms, so one env step = 5 ms sim time). `CustomCartPole` wraps this in the Gymnasium API (Discrete(2) actions, 4-dim Box observation: cart x, cart velocity, pole pitch, pole angular velocity). Termination: |pitch| > 0.48 rad or |cart x| > 4.8 m.

2. **Inference (out-of-process, with GUI)** — after training, the script spawns `gz sim -s -r` and `gz sim -g` as subprocesses and talks to them over Gazebo transport: it publishes `EntityWrench` forces to `/world/cart_pole/wrench` and reads state from `/world/cart_pole/dynamic_pose/info`. This path depends on the `ApplyLinkWrench` and `SceneBroadcaster` plugins declared in `cart_pole.sdf`. Note the observation here is degraded — cart/pole velocities are never populated by the pose callback, only positions.

Load-bearing details:

- The `ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)` at the very top of the script must run **before** any `gz.*` import, or symbol resolution fails.
- Entity names are hard-coded strings shared between the SDF and Python: model `vehicle_green`, links `pole` and `chassis`, world `cart_pole` (which also appears in the transport topic names). Renaming anything in `cart_pole.sdf` requires matching edits in `cart_pole_env.py`.
- After `reset()`, entity handles are re-looked-up lazily via `_ensure_initialized` on the next update; `enable_velocity_checks` must be re-enabled each pre-update or velocity reads return None.
