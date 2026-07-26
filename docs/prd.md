# PRD: Headless Gazebo-native training for the ros2_ws cart-pole

## Background

`ros2_ws/src/` currently trains its cart-pole via `commander/commander/dqn_learning.py`: a from-scratch PyTorch DQN that runs **live**, in real time, over ROS 2 topics/services against a running `gz sim` GUI + `ros_gz_bridge`. This is slow (wall-clock-bound, `time_interval=0.02` per step) compared to the sibling `cart_pole/` project at the repo root, which trains PPO (Stable-Baselines3) by driving `gz.sim8.TestFixture` directly in-process, headless, fast-forwarding through 100,000 timesteps with no real-time pacing and no ROS involvement at all.

This feature ports that second approach — headless, in-process, SB3-driven training — into `ros2_ws`, without disturbing the existing `commander` DQN path or the live ROS launch stack.

## Goal

A standalone training script under `ros2_ws/src/` that trains a policy for the ros2_ws cart-pole robot (the URDF-defined one: prismatic `cart_joint` + revolute `pole_joint`, described in `robot_description/robot/cart_pole.urdf.xacro`) headlessly via `TestFixture`, saving a policy file, the same way `cart_pole/cart_pole_env.py` does for the (unrelated) toy vehicle model at the repo root.

## Out of scope

- Any change to `commander/dqn_learning.py` or the live ROS launch stack (`robot_launch`, `robot_control`, the `ros_gz_bridge` topic mapping) — untouched.
- Installing SB3/gymnasium into system `dist-packages`, or otherwise making this buildable/runnable via `colcon`/`ros2 run`.

> **Amendment (follow-up feature):** out-of-process/GUI inference, originally listed above as a deferred follow-up, is now in scope — see Requirement 8 and `docs/plan.md`'s Task 6.

## Requirements

1. **No ROS integration.** Plain Python script using `gz.sim8.TestFixture` directly (no `rclpy`), run via `PYTHONPATH=/usr/lib/python3/dist-packages uv run ros2_ws/src/cart_pole_gz_train/train_cart_pole.py` — same invocation style as the root project. Requires the same `ctypes.CDLL(".../libgz-sim8.so", RTLD_GLOBAL)` preload before any `gz.*` import (verified: importing `gz.sim8` without it fails with an undefined-symbol error under system Python).

2. **Independent, not a colcon package.** Lives in a new top-level folder `ros2_ws/src/cart_pole_gz_train/` with no `package.xml`/`CMakeLists.txt`, so `colcon build` ignores it entirely — it never becomes a ROS package.

3. **Joint-based ECM access, not wrench-based.** Unlike the root project (which has no joint on its chassis and so needed a finite-difference velocity estimator, `/wrench/persistent`, and net-force delta tracking to work around gz-sim's broken wrench-clear), this robot has real joints. The trainer applies force to `cart_joint` directly via its `JointForceCmd` component and reads `cart_joint`/`pole_joint` position and velocity directly from their `Joint` components — no wrench plugin, no finite-difference estimation, no reset-time force bookkeeping needed.

4. **World SDF is auto-generated from the real xacro at script start, not hand-copied.** `TestFixture` needs a full `<world>`-wrapped SDF, but `cart_pole.urdf.xacro` only produces a bare `<robot>`. The script:
   - Runs `xacro cart_pole.urdf.xacro` as a subprocess with `PATH` sanitized to strip `.venv/bin` — verified empirically that `xacro` (an `ament_index_python`-based ROS tool) fails under the venv's `python3` with `PackageNotFoundError` for `robot_description`, and succeeds once the venv is stripped from `PATH`. This is the same class of environment gotcha the root `CLAUDE.md` already documents for `colcon build`.
   - Converts the resulting URDF to SDF via `gz sdf -p` (verified working).
   - Post-processes the result: strips all `<visual>` blocks, replaces each `<collision><geometry><mesh>` with a small hardcoded primitive (box for `base_footprint`/`cart_link`, cylinder for `pole_link`) sized to approximate the real mesh, and drops `tip_link`/`tip_joint` (mass 0.0001, physically negligible).
   - Wraps the processed model in a fresh `<world>` with a 1ms-step `Physics` plugin plus `UserCommands`/`SceneBroadcaster`, and a ground plane. No `ApplyLinkWrench` plugin (not needed — force goes through the joint, not a wrench).
   - This keeps the numbers that actually matter for RL dynamics (masses, inertias, joint types/limits) always in sync with the real xacro, while collision geometry (which barely affects cart-pole dynamics) stays hand-simplified.
   - On `xacro`/`gz sdf` subprocess failure, raise immediately with stderr attached — no silent fallback to a stale cached SDF. The generated SDF is written to disk each run (not just held in memory) so a bad conversion can be inspected directly.

5. **Gym wrapper.** `Discrete(2)` action space, 4-dim `Box` observation (cart x, cart velocity, pole pitch, pole angular velocity) — matching the root project's shape. Reward/termination thresholds reused from the root project (`|pitch| > 0.48 rad`, `|cart x| > 4.8 m` → terminal) since these are generic bounds, not tied to the root project's specific geometry. Force magnitude for the two discrete actions is tuned empirically during implementation against this model's actual `cart_joint` effort limit (30N) rather than reused from the root project's (2000N, a different-scale unrelated model).

6. **Reset** rebuilds the `TestFixture`/server from scratch (same as the root project) rather than calling `server.reset_all()`, which desyncs the physics engine from the ECM without changing entity IDs, making it look valid while silently breaking force application and state reads. No net-force tracking is needed on reset here (unlike the root project) since there's no persistent wrench to zero out.

7. **Output**: saved policy `cart_pole_gz_train_ppo.zip`, written into `ros2_ws/src/cart_pole_gz_train/` alongside the script and the generated SDF.

8. **Live GUI inference.** A standalone `run_inference.py`, mirroring `cart_pole_env.py`'s `run_inference()`: spawn `gz sim -s -r`/`-g` against the generated training world and drive the saved policy over Gazebo transport (not `TestFixture`) until Ctrl+C. Because this robot has real joints, the transport interface is simpler than the root project's wrench-based one — command `cart_joint` via the `ApplyJointForce` plugin's `/model/cart_pole/joint/cart_joint/cmd_force` topic (`gz.msgs.Double`, holds its last value with no clear/decay bug, unlike `/wrench/persistent`) and read true joint position/velocity (no finite-difference estimation needed) off the `JointStatePublisher` plugin's `/world/cart_pole_train/model/cart_pole/joint_state` topic (`gz.msgs.Model`, `joint[].axis1.position`/`.velocity`, keyed by joint name) via plain pub/sub, rather than the root project's synchronous FNV-hashed ECS-state decoding. Observations must be normalized with the same `VecNormalize` statistics (`vecnormalize.pkl`) used in training before calling `model.predict` — done via a `gym.Env` stub carrying only the right `observation_space`/`action_space` (no `gz` calls), wrapped in `DummyVecEnv` + `VecNormalize.load(...)`, calling `.normalize_obs()` as a pure array op so a second live `GzCartPoleScorer` never double-registers on the training world's transport name. Cart-joint force magnitude is parsed from the generated `cart_pole_train.sdf`'s `<limit><effort>` rather than hardcoded. Episode reset on falling (`|cart_pos| > CART_POSITION_LIMIT` or `|pole_pos| > POLE_PITCH_LIMIT`, imported from `gz_scorer.py`) uses `WorldControl.reset.model_only = True` on `/world/cart_pole_train/control` — **not** `reset.all`, which `ros2_ws/src/commander`'s notes document as permanently killing `JointStatePublisher`'s topic advertisement after the first reset.

   > **Amendment (post-implementation):** `reset.model_only` above was superseded. Direct measurement while implementing `run_inference.py` showed `reset.model_only` is a complete no-op on this project's generated world — position and velocity keep evolving through the "reset" with zero effect, so a real out-of-bounds episode could never recover. The shipped `run_inference.py` uses `reset.all` instead, confirmed by direct measurement to actually reset position/velocity to ~0. This does **not** hit the `JointStatePublisher`-killing teardown bug documented above for the unrelated `commander`/`robomaster_rale` world — on this project's own generated SDF, `reset.all` leaves `JointStatePublisher` publishing normally afterward. See `docs/plan.md`'s Task 6 amendment and `run_inference.py`'s `_reset_world` docstring for the full explanation.

9. **Testing**: no pytest, matching this repo's existing convention in both `cart_pole/` and `ros2_ws/`. Verification is a scratch script that steps the env a handful of times and prints applied force vs. resulting `cart_joint` velocity change, confirming the force→dynamics link works before committing to a full training run. Training success is judged by episode reward trending upward.

## Key trade-offs decided during design

- **Standalone script vs. real ROS node**: standalone chosen — SB3/gymnasium only exist in the repo's `.venv`, while `colcon build` requires the venv *not* be on `PATH` (needs system `catkin_pkg`). Making this a real `ros2 run`-able node would mean either installing SB3 system-wide or building a re-exec shim; not worth it for a script that has no reason to integrate with ROS topics/services in the first place.
- **New independent folder vs. tucked inside `commander`**: independent folder chosen, mirroring how `cart_pole/` sits self-contained at the repo root.
- **Joint-based ECM access vs. replicating the root project's wrench-based approach**: joint-based chosen — this robot has real joints the root project's model doesn't, so the wrench-based workarounds (finite-difference velocity, wrench-clear bug, net-force tracking) don't apply and shouldn't be needlessly replicated.
- **Primitive collision geometry vs. real meshes**: primitives chosen — headless training never renders, and meshes would require resolving `GZ_SIM_RESOURCE_PATH` inside a script that's otherwise self-contained, plus slower mesh-based collision across 100k timesteps.
- **Auto-generate world SDF from xacro vs. hand-author once**: auto-generate chosen, to avoid a second, manually-synced source of truth for physical parameters (mass/inertia/joint limits) that could silently drift from the real robot description. This reintroduced the mesh-geometry conflict with the primitives decision above, resolved by generating physics parameters from the xacro but still hand-simplifying collision geometry in a post-processing step.
