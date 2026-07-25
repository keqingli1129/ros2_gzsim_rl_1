# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

A cart-pole reinforcement-learning demo built directly on the Gazebo Sim (gz-sim 8 / Harmonic) Python API, trained with Stable Baselines3 PPO. Despite the repo name, no ROS 2 APIs are used — the sim is driven through `gz.sim8` bindings and `gz.transport13` topics. The project lives in `cart_pole/` — one script, `cart_pole/cart_pole_env.py`, plus the world definition `cart_pole/cart_pole.sdf` — under a repo root shared across environments (single `.venv`, `pyproject.toml`, `uv.lock`). Other envs (e.g. a future bipedal one) are expected to live in sibling top-level folders following the same self-contained layout.

## Environment & Running

Dependencies are split across two places, and both are required:

- **uv-managed venv** (`.venv`, Python 3.12, no system site-packages): `stable-baselines3[extra]`, gymnasium, torch. Install with `uv sync`.
- **System dist-packages** (`/usr/lib/python3/dist-packages`): the Gazebo Python bindings (`gz.sim8`, `gz.common5`, `gz.math7`, `gz.transport13`, `gz.msgs10`), installed via the `gz-harmonic` apt packages on Ubuntu Noble. `.vscode/settings.json` adds this path for Pylance only — it is **not** on the venv's import path at runtime.

So running the script requires exposing the system bindings to the venv:

```bash
PYTHONPATH=/usr/lib/python3/dist-packages uv run cart_pole/cart_pole_env.py
```

Importing the script has side effects: it immediately trains PPO for 100,000 timesteps (headless), saves `cart_pole_ppo.zip`, then launches a Gazebo server + GUI and runs inference until Ctrl+C. Pass `--infer-only` to skip training and load the saved `cart_pole_ppo.zip` straight into `run_inference`. There are no tests or linters configured — verification for changes to the inference path is done with scratch scripts run against a live `gz sim` server, not pytest.

`pyproject.toml` overrides opencv to `opencv-python-headless` (via `[tool.uv] override-dependencies`) so that `stable-baselines3[extra]` doesn't pull in GUI opencv — keep that override when touching dependencies.

The repo root also contains a `dqn/` directory (a from-scratch DQN implementation training on gym's classic `CartPole-v1` and `FlappyBird-v0`, with its own `hyperparameters.yml`). It is unrelated to the gz-sim cart-pole project — don't conflate the two when a task mentions "cart pole" or "DQN".

## Architecture

`cart_pole_env.py` has two distinct sim-interaction paths that must not be confused:

1. **Training (headless, in-process)** — `GzRewardScorer` loads `cart_pole.sdf` through `gz.sim8.TestFixture` and hooks `on_pre_update` (applies ±2000 N world force to the chassis based on the pending action) and `on_post_update` (reads pole pitch / cart position from the ECM, computes reward and termination). Cart/pole velocities are estimated by finite difference across the 5ms step (`(pose - prev_pose) / 0.005`), deliberately matching the estimator used at inference rather than reading the physics engine's true instantaneous velocity — training on the same estimator the deployed policy will actually see avoids a train/inference distribution mismatch (empirically ~76% relative error between the two). Each Gym `step()` runs the server for 5 blocking sim iterations (physics step is 1 ms, so one env step = 5 ms sim time). `CustomCartPole` wraps this in the Gymnasium API (Discrete(2) actions, 4-dim Box observation: cart x, cart velocity, pole pitch, pole angular velocity). Termination: |pitch| > 0.48 rad or |cart x| > 4.8 m.

2. **Inference (out-of-process, with GUI)** — after training, the script spawns `gz sim -s -r` and `gz sim -g` as subprocesses and talks to them over Gazebo transport. Forces are published as `EntityWrench` messages to `/world/cart_pole/wrench/persistent` (not the plain `/wrench` topic — that only holds a force for one 1ms physics tick before dropping it, a fifth of training's per-action authority). State is read via synchronous request/response calls to the `/world/cart_pole/state` service (`_query_world_state`), not the async `dynamic_pose/info` subscription — the service returns a full ECS snapshot on demand, decoded by replicating gz-sim's internal FNV-1a component-type hash (`_gz_component_hash`) to identify `Name`/`Pose` components, then composing model+link local poses into world frame (`_world_frame_pose`). Velocities are finite-differenced against the response's own `sim_time`. This path depends on the `ApplyLinkWrench` and `SceneBroadcaster` plugins declared in `cart_pole.sdf`.

Load-bearing details:

- The `ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)` at the very top of the script must run **before** any `gz.*` import, or symbol resolution fails.
- Entity names are hard-coded strings shared between the SDF and Python: model `vehicle_green`, links `pole` and `chassis`, world `cart_pole` (which also appears in the transport topic names). Renaming anything in `cart_pole.sdf` requires matching edits in `cart_pole_env.py`.
- After `reset()`, entity handles are re-looked-up lazily via `_ensure_initialized` on the next update.
- `/wrench/persistent` entries can't be cleared in this gz-sim build (`OnWrenchClear`'s entity match never succeeds) and survive a world reset untouched, so `run_inference` never clears — it tracks the net force already applied (`net_force_x`) and publishes only the delta needed to reach the new target force (including zeroing it before a reset).
- `GzRewardScorer.reset()` rebuilds the `TestFixture`/server from scratch rather than calling `server.reset_all()` — the latter desyncs the physics engine from the ECM (force application and velocity reads silently stop working) while leaving entity IDs unchanged, so it isn't detectable as a stale-handle problem.

## ros2_ws

A separate, mostly-independent port of the same cart-pole concept onto the standard ROS 2 stack — a colcon workspace (ROS 2 Jazzy) with four packages under `ros2_ws/src/`: `robot_description` (URDF/xacro + STL/DAE meshes), `robot_control` (`robot_state_publisher` launch), `robot_launch` (top-level launch: spawns Gazebo, the `ros_gz_bridge`, and the robot), and `commander` (a from-scratch PyTorch DQN node, `dqn_learning.py`). It uses `ros2_control`-adjacent Gazebo plugins (`ApplyJointForce`, `JointStatePublisher`) and ROS 2 topics/services rather than `cart_pole/`'s direct `gz.sim8`/`gz.transport13` calls — the two are unrelated code paths that happen to model the same robot, not a shared implementation.

**Building this workspace requires the opposite Python environment from `cart_pole/`:** `colcon build` must run with this repo's `.venv` **not** shadowing `python3` — ROS Jazzy's `ament_cmake` package processing needs `catkin_pkg`, which lives in system dist-packages, not the isolated venv. If `.venv/bin` is ahead of it on `PATH` (e.g. the venv is active), CMake picks up the venv's `python3`, `catkin_pkg` import fails, and the build errors out (also make sure to `rm -rf build install log` first if a prior attempt already ran under the wrong interpreter, since CMake caches the interpreter path). Build with:

```bash
cd ros2_ws
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv' | paste -sd:) VIRTUAL_ENV= colcon build --symlink-install
```
