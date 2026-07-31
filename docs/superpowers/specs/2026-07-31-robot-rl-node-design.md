# Design: `robot_rl_node` — ROS2-node conversion of `cart_pole_gz_train`

## Purpose

`ros2_ws/src/cart_pole_gz_train/` is a deliberately non-colcon folder (no
`package.xml`/`CMakeLists.txt`) that trains and runs an SB3 PPO cart-pole
policy headlessly and fast, via direct `gz.sim8.TestFixture`/ECM access —
bypassing `rclpy`/ROS2 topics entirely for speed (unlike `commander`'s
DQN, which does go over ROS2 topics and is the slow path by comparison).

This converts that utility into a real colcon package, `robot_rl_node`,
so its scripts are discoverable and runnable via `ros2 run` / launch
files and show up in the ROS2 graph as nodes — without touching the
underlying training/inference mechanics that make it fast.

`cart_pole_gz_train/` is left in place, unmodified. This is a
copy-and-convert into a new package, not a refactor of shared code; the
two are independent from this point on.

## Non-goals

- Does **not** rework training/inference to drive the robot over real
  ROS2 topics/services (the way `commander/dqn_learning.py` does against
  `/cart_controller/command`, `/joint_states`,
  `/world/.../control`). That would reintroduce DDS/`ros_gz_bridge`
  round-trip overhead per env step, which the original design explicitly
  avoided. All `gz.sim8` ECM access and `gz.transport13` topic use inside
  the training/inference loops is preserved exactly as-is.
- Does not modify `cart_pole_gz_train/` in any way, or remove it.
- Does not change the trained policy's observation/action contract,
  reward, termination thresholds, or `VecNormalize` requirement.

## Package layout

New ament_python package, structured like the existing `commander`
package:

```
ros2_ws/src/robot_rl_node/
  package.xml
  setup.py
  setup.cfg
  resource/robot_rl_node
  robot_rl_node/
    __init__.py
    gz_scorer.py                              # support module, ported as-is
    world_builder.py                          # support module, REPO_ROOT path fixed (see below)
    train_cart_pole.py                        # rclpy.Node wrapper; entry point `train_cart_pole`
    run_inference.py                          # rclpy.Node wrapper; entry point `run_inference`
    evaluate_policy.py                        # rclpy.Node wrapper; entry point `evaluate_policy`
    nudge.py                                  # rclpy.Node wrapper; entry point `nudge`
    verify_dynamics.py                        # rclpy.Node wrapper + main(); entry point `verify_dynamics`
    verify_scorer.py                          # rclpy.Node wrapper + main(); entry point `verify_scorer`
    verify_world_builder.py                   # rclpy.Node wrapper + main(); entry point `verify_world_builder`
    verify_reset_preserves_joint_state.py     # rclpy.Node wrapper + main(); entry point `verify_reset_preserves_joint_state`
```

`package.xml` declares `ament_python` build type and a `rclpy` exec/build
dependency, matching `commander/package.xml`'s pattern.

`setup.py` registers all 8 as `console_scripts`, matching
`commander/setup.py`'s pattern:

```python
entry_points={
    'console_scripts': [
        'train_cart_pole = robot_rl_node.train_cart_pole:main',
        'run_inference = robot_rl_node.run_inference:main',
        'evaluate_policy = robot_rl_node.evaluate_policy:main',
        'nudge = robot_rl_node.nudge:main',
        'verify_dynamics = robot_rl_node.verify_dynamics:main',
        'verify_scorer = robot_rl_node.verify_scorer:main',
        'verify_world_builder = robot_rl_node.verify_world_builder:main',
        'verify_reset_preserves_joint_state = robot_rl_node.verify_reset_preserves_joint_state:main',
    ],
},
```

## Node wrapping (all 8 scripts)

Every one of the 8 scripts gets a trivial `rclpy.Node` wrapper around its
existing, unmodified logic:

```python
def main():
    rclpy.init()
    node = rclpy.create_node('cart_pole_rl_<script_name>')
    try:
        <existing script logic, unchanged>
    finally:
        node.destroy_node()
        rclpy.shutdown()
```

This gives every script ROS2-graph presence (`ros2 node list`) and a
`ros2 run robot_rl_node <name>` invocation, but no script gains ROS2
publishers/subscribers/services internally — the node object exists for
graph presence and (optionally) `get_logger()`, not for communication.
This preserves the original design's speed rationale in full.

The 4 `verify_*.py` scripts currently run as unguarded top-level module
code (no `main()`, and in some cases no `if __name__ == "__main__":`
guard at all). Converting them requires wrapping their existing logic in
a `main()` function each — a mechanical wrap of existing code, not a
logic change.

## Import style

The current scripts use a flat-folder import style:
`sys.path.insert(0, FILE_DIR)` followed by bare imports
(`from gz_scorer import ...`, `from world_builder import ...`,
`from train_cart_pole import CustomCartPoleGzTrain`). Since these modules
now live inside an actual installed Python package, this is replaced with
real package-relative imports (`from robot_rl_node.gz_scorer import ...`,
`from robot_rl_node.world_builder import ...`, etc.), and the
`sys.path.insert` hack is dropped.

The `ctypes.CDLL(".../libgz-sim8.so", ctypes.RTLD_GLOBAL)` call at the
top of each script that touches `gz.sim8` — which must run before any
`gz.*` import or symbol resolution fails — is preserved unchanged, at
the top of each relevant module.

## `world_builder.py` path fix

`world_builder.py`'s `REPO_ROOT` is computed as
`os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))`
— 3 `..` hops, correct for `cart_pole_gz_train/world_builder.py`'s depth
(`cart_pole_gz_train/ → src/ → ros2_ws/ → repo root`). In the new
location, `robot_rl_node/robot_rl_node/world_builder.py` sits one
directory deeper (the ament_python package adds an extra nesting level:
`<package>/<package>/`), so this needs a 4th `..` hop to still resolve to
the actual repo root. This is a straightforward constant fix, verified by
asserting the resolved path against the known repo root.

## Artifacts (trained model, VecNormalize stats, generated world)

Each script continues to save/load its outputs next to its own module
file (`FILE_DIR`-relative), exactly as today — e.g.
`train_cart_pole.py` still writes `cart_pole_gz_train_ppo.zip` and
`vecnormalize.pkl` beside itself, `gz_scorer.py`/`run_inference.py` still
write/read the generated `cart_pole_train.sdf` beside themselves.

Because this workspace's established build convention is
`colcon build --symlink-install` (per the root `CLAUDE.md`), the
installed module remains a symlink back to its source file, so these
writes land in `ros2_ws/src/robot_rl_node/robot_rl_node/` in the repo
tree, not somewhere under `install/`.

`.gitignore` gains entries for the new package's generated files,
mirroring the existing `cart_pole_gz_train` entries:

```
ros2_ws/src/robot_rl_node/robot_rl_node/cart_pole_train.sdf
ros2_ws/src/robot_rl_node/robot_rl_node/_generated.urdf
ros2_ws/src/robot_rl_node/robot_rl_node/cart_pole_gz_train_ppo.zip
ros2_ws/src/robot_rl_node/robot_rl_node/vecnormalize.pkl
```

## Documentation

Add a `robot_rl_node` section to `ros2_ws/src/CLAUDE.md`, describing it
as the ROS2-node-wrapped counterpart to `cart_pole_gz_train` (same
mechanics, packaged for `ros2 run`/graph presence). The existing
`cart_pole_gz_train` section is left as-is, since nothing about that
folder changes.

## Testing / verification

Matches this repo's no-pytest convention. After
`colcon build --symlink-install` (with the venv-stripped-`PATH` build
invocation documented in the root `CLAUDE.md`):

- `ros2 run robot_rl_node verify_world_builder`,
  `ros2 run robot_rl_node verify_scorer`,
  `ros2 run robot_rl_node verify_dynamics`,
  `ros2 run robot_rl_node verify_reset_preserves_joint_state` all run and
  pass their existing assertions unchanged.
- `ros2 run robot_rl_node train_cart_pole` runs a (short, for
  verification purposes) training loop and appears in `ros2 node list`
  while running.
- `ros2 run robot_rl_node run_inference` launches the GUI and appears in
  `ros2 node list` while running; `ros2 run robot_rl_node nudge` in a
  second terminal still produces a visible disturbance.
- `ros2 run robot_rl_node evaluate_policy trained` reports the same kind
  of episode-length statistics as the original script.
