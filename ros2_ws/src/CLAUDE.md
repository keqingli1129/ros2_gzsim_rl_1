# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

This covers the four packages inside `ros2_ws/src/`, plus the non-package `cart_pole_gz_train/` folder that also lives there. For the workspace-level build quirk (venv `PATH` shadowing `catkin_pkg`) and how this workspace relates to the top-level `cart_pole/` project, see the root `CLAUDE.md`'s "ros2_ws" section — that context isn't repeated here.

## Running

Build first (from `ros2_ws/`, per the root CLAUDE.md), then source and launch:

```bash
source install/setup.bash
ros2 launch robot_launch launch_simulation.launch.py
```

There are no unit tests in any of the four packages (no `test/` directories, despite `commander/setup.py` listing `tests_require=['pytest']`).

## Packages and launch order

`robot_launch` is the top-level package; its `launch_simulation.launch.py` wires the other three together in a specific dependency order, not just a flat list of nodes:

1. Starts `gz_sim.launch.py` (from `ros_gz_sim`) against `robot_launch/worlds/robomaster_rale.world` — the world name `robomaster_rale` is hard-coded into `robot_control_launch.py`'s and `commander`'s topic/service names below, so renaming the world means updating those too.
2. Runs `xacro` on `robot_description/robot/cart_pole.urdf.xacro` (via `Command(['xacro ', ...])`) to produce the `robot_description` launch argument, and publishes it through **two separate** `robot_state_publisher` nodes — one included from `robot_control`'s own launch file, one instantiated directly in `robot_launch`. Both exist because `robot_control_launch.py` was written to be includable standalone; `robot_launch` duplicates the node rather than relying on the include alone.
3. Sets `GZ_SIM_RESOURCE_PATH` to the parent of `robot_description`'s share directory *before* spawning anything, because gz-sim resolves the URDF's `package://robot_description/meshes/...` URIs as `model://robot_description/meshes/...`, which is looked up relative to that env var, not the ROS package path.
4. Spawns the model as `cart_pole` at `z=1.225` via `ros_gz_sim`'s `create` executable, reading the URDF off the `robot_description` topic.
5. Starts `ros_gz_bridge`'s `parameter_bridge` with three bridged topics/services (see below).
6. Only *after* the spawn node exits (`RegisterEventHandler(OnProcessExit(...))`) does it launch `commander` — the DQN node must not start publishing forces before the model exists in the world.

### Bridge topic/service mapping (`robot_launch`)

| Gazebo side | ROS side | Note |
|---|---|---|
| `/model/cart_pole/joint/cart_joint/cmd_force` | `/cart_controller/command` | `Float64` |
| `/world/robomaster_rale/model/cart_pole/joint_state` | `/joint_states` | `JointState`; note the topic is nested under the **world** name, not the plain `/model/<model>/...` form the `JointStatePublisher`/`ApplyJointForce` docs describe — confirmed empirically via `gz topic -l`/`gz topic -i` against a running sim, for a model spawned into an already-running world |
| `/world/robomaster_rale/control` | (same) | `ros_gz_interfaces/srv/ControlWorld`, used by `commander` to reset between episodes |

### `robot_description`

URDF/xacro + mesh (DAE for visuals, STL for collision) description of the cart-pole, built from four xacro macros (`base`, `cart`, `pole`, `tip`) chained as a `base_footprint → base_link → cart_link → pole_link → tip_link` joint tree in `robot/cart_pole.urdf.xacro`. The two gz-sim plugins that make the robot controllable/observable are declared directly in this top-level xacro file (not per-link):

- `gz::sim::systems::ApplyJointForce` bound to `cart_joint` — the actuation path.
- `gz::sim::systems::JointStatePublisher` — the state-feedback path (see the bridge table above for why its topic is world-namespaced).

`cart_trans_v0` declares a `transmission_interface/SimpleTransmission` on `cart_joint`, but nothing in this workspace consumes `ros2_control` — it's vestigial from an earlier `ros2_control`-based design and has no effect on the gz-sim plugin path actually in use.

### `robot_control`

Single-purpose package: its launch file starts one parameterized `robot_state_publisher` node, taking `robot_description` as a launch argument rather than reading a file itself. Exists so `robot_launch` (or anything else) can bring up state publishing via one include rather than duplicating the node config — although `robot_launch` currently also stands up its own second `robot_state_publisher` instance alongside this include (see step 2 above).

### `commander`

`dqn_learning.py` is a from-scratch DQN (`QNet`/`Brain`/`Agent`), not Stable-Baselines3 (unlike the unrelated `cart_pole/` project at the repo root). `DQNSimulationNode`:

- Reads cart position/velocity and pole angular velocity off `/joint_states` by name-matching `cart_joint`/`pole_joint` in the message, and integrates pole yaw angle itself from angular velocity (`yaw_angle += y_angular * time_interval`) — the joint state message carries no angle field for `pole_joint` directly usable here.
- Publishes a scalar force to `/cart_controller/command`, computed from a discretized action index (`num_actions=10`) via `force = action * 16 / 9 - 8`, mapping the discrete action space onto a continuous force range.
- Resets between episodes by calling `/world/robomaster_rale/control` with `reset.model_only = True`, deliberately **not** `reset.all` — confirmed via live testing that `reset.all` tears down and recreates every entity, which permanently stops `JointStatePublisher` from advertising its gz-transport topic after the first reset (silently freezing `/joint_states` for the rest of training). `model_only` resets poses/velocities without that teardown.
- Runs training synchronously inside the main thread (`node.simulate(...)` in a loop up to `num_episodes=1000`), with `rclpy.spin` on a separate daemon thread purely to service the `/joint_states` subscription and the reset service client in the background.

## `cart_pole_gz_train` (not a package)

`ros2_ws/src/cart_pole_gz_train/` is deliberately **not** a colcon package — it has no `package.xml`/`CMakeLists.txt`, so `colcon build` skips it entirely. It trains a Stable-Baselines3 PPO policy for the same cart-pole robot headlessly and in-process via `gz.sim8.TestFixture`, with no `rclpy`, no `ros_gz_bridge`, and no real-time pacing — the fast alternative to `commander`'s live over-ROS DQN, ported from the root `cart_pole/` project's approach but driving the real `cart_joint`/`pole_joint` through the ECM instead of publishing wrenches.

Run it from the **repo root** (not from `ros2_ws/`):

```bash
PYTHONPATH=/usr/lib/python3/dist-packages uv run ros2_ws/src/cart_pole_gz_train/train_cart_pole.py
```

Two environment prerequisites, both non-obvious:

- **`ros2_ws` must already be `colcon build`-ed.** `world_builder.py` regenerates the training world from `robot_description/robot/cart_pole.urdf.xacro` on every run by shelling out to `xacro` and `gz sdf -p`, and it sources `ros2_ws/install/setup.bash` (with `.venv` stripped from `PATH`) to do so. Without a built workspace, `xacro` fails with `PackageNotFoundError: robot_description`.
- The generated `cart_pole_train.sdf` / `_generated.urdf` are **outputs, not sources** — gitignored, rewritten once per process. Don't hand-edit them; edit the xacro or `world_builder.py`'s postprocessing.

**Any consumer of the saved policy must also load `vecnormalize.pkl`.** `train_cart_pole.py` writes two files side by side: `cart_pole_gz_train_ppo.zip` and `vecnormalize.pkl`. The policy was trained on `VecNormalize`-normalized observations (this env's velocity dimensions run 10-30x larger than its position dimensions, which stalled learning outright until normalization was added), so the running mean/std are part of the policy's expected input contract. Loading the `.zip` alone and feeding it raw observations does not degrade gracefully — it silently collapses to random-baseline performance. Load it as `VecNormalize.load(path, venv)` with `venv.training = False` (see `evaluate_policy.py`, which hard-errors rather than proceeding if the stats are missing).

Verification here follows the repo's no-pytest convention: `verify_world_builder.py`, `verify_dynamics.py`, `verify_scorer.py` are scratch scripts run the same way as the trainer. `verify_dynamics.py`/`verify_scorer.py` specifically assert the robot spawns *grounded and at rest* (base at z=0.3, pole undisturbed) and that max effort yields ~`effort_limit/cart_mass` acceleration — regression guards for a bug where the model spawned airborne and landed with its pole collision cylinder speared through the ground plane.

`run_inference.py` is the live-GUI counterpart to `train_cart_pole.py`: it loads `cart_pole_gz_train_ppo.zip` + `vecnormalize.pkl`, spawns `gz sim -s -r`/`gz sim -g` as subprocesses, and drives `cart_joint` over `/model/cart_pole/joint/cart_joint/cmd_force` while reading state from `/world/cart_pole_train/model/cart_pole/joint_state`. `verify_reset_preserves_joint_state.py` is its regression guard for the episode-reset path. Both use `WorldControl.reset.all`, **not** `reset.model_only` — the opposite of `commander`'s choice above, and deliberately so: `reset.model_only` was measured to be a complete no-op against this project's SDF-declared world (position/velocity never change, despite the RPC reporting success), whereas `reset.all` was measured to work and to *not* hit the `JointStatePublisher`-killing teardown bug documented for `commander`'s dynamically-spawned `robomaster_rale` world. That teardown bug is real but specific to how a model gets into the world (spawned at runtime via `ros_gz_sim`'s `create`, vs. declared whole in the SDF `gz sim -s -r` loads directly) — don't generalize either reset type's safety from one world to the other without re-measuring.

`nudge.py` is a manual testing helper, not part of the training/inference pipeline: run it in a second terminal while `run_inference.py` is already up, to publish a short (default 0.3s), fast (500Hz) force burst to `cmd_force` and watch the policy recover from a real disturbance. A single one-off `gz topic -p` publish isn't enough to test this — `ApplyJointForce` just holds the last commanded value, so a lone publish gets overwritten within ~5ms by `run_inference.py`'s own next action, and the `gz topic` CLI's per-call connection overhead is too slow to sustain a burst at that rate. `nudge.py` holds one `gz.transport13` `Node`/publisher open instead, matching the cadence needed to actually override the policy.

`world_builder.py`'s generated world also carries a `<gui>` block (camera pose, per-link visual materials, a sun light, `WorldControl`/`WorldStats`/`EntityTree` plugins) purely for `run_inference.py`'s benefit — headless training never renders, so none of it affects `train_cart_pole.py`. Note a `<gui>` element in the SDF *replaces* gz-sim's default GUI config rather than extending it: every plugin the GUI should show has to be declared explicitly, or it silently disappears (this cost an iteration — an early version of the block declared only the 3D-view plugin and quietly dropped World Control/World Stats/Entity Tree).
