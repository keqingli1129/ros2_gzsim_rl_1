# Headless gz-sim Cart-Pole Training — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A standalone script under `ros2_ws/src/cart_pole_gz_train/` that trains a PPO policy for the ros2_ws cart-pole (real `cart_pole.urdf.xacro` joints/masses) headlessly via `gz.sim8.TestFixture`, driving `cart_joint` directly through the ECM — no ROS, no wrench workarounds, no `colcon` involvement.

**Architecture:** `world_builder.py` auto-generates a training-only world SDF from the real xacro (xacro → `gz sdf -p` → strip visuals/replace mesh collision with primitives → wrap in a `<world>`). `gz_scorer.py` hosts `GzCartPoleScorer`, a `TestFixture`-driven reward/state scorer using joint position/velocity/force directly (no finite-difference estimation). `train_cart_pole.py` wraps it in a Gymnasium env and runs SB3 PPO.

**Tech Stack:** `gz.sim8`/`gz.common5` (system `dist-packages`), Stable-Baselines3 + gymnasium + torch (`.venv`), `xacro` + `gz sdf` (ROS/gz-sim CLI tools, invoked as subprocesses).

## Global Constraints

- No `package.xml`/`CMakeLists.txt` in `ros2_ws/src/cart_pole_gz_train/` — must never become a colcon package (verified: `colcon build` only builds directories containing a manifest).
- Every `gz.*` import must be preceded by `ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)` — verified: omitting this makes `import gz.sim8` fail with an undefined-symbol error.
- Any subprocess that shells out to `xacro` or `gz sdf` must (a) source `ros2_ws/install/setup.bash` and (b) strip any `.venv` entry from `PATH` and unset `VIRTUAL_ENV` first — verified empirically: `xacro` (an `ament_index_python` tool) throws `PackageNotFoundError: robot_description` under the venv's `python3`, and separately fails to resolve the `robot_description` package at all unless `ros2_ws/install/setup.bash` has been sourced (i.e. **`ros2_ws` must already be `colcon build`'t** before this script can run).
- Run the finished script the same way as the root project: `PYTHONPATH=/usr/lib/python3/dist-packages uv run ros2_ws/src/cart_pole_gz_train/train_cart_pole.py` from the repo root.
- No pytest anywhere in this repo (`cart_pole/` or `ros2_ws/`) — verification is scratch-script based, run against real `TestFixture`/`gz sdf` behavior, matching existing convention.
- `cart_joint`'s declared effort limit (30N) is a **hard actuator clamp enforced by the physics engine**, not just metadata — verified: commanding 1,000,000N produces the same realized acceleration as 30N. Action force magnitude must be read from `Joint.effort_limits(ecm)[0]` at runtime, not hardcoded.
- `cart_joint`'s declared position limit (±1m) is a **hard mechanical stop** — verified: sustained max force drives `position(ecm)` to exactly `1.0` and it stays pinned there (velocity collapses to ~0). The out-of-bounds termination threshold must be well inside this (e.g. `0.9`), **not** the root project's unrelated `4.8`, which this joint physically cannot reach.
- The generated world's root model must be given an initial `<pose>` lifting it clear of the ground plane (verified root cause of a serious bug — see Task 2) — at `z=0` the primitive collision geometry interpenetrates the ground plane on load, and the resulting contact-resolution forces dominate the tiny prismatic joint's dynamics, making force application look completely broken (a 1,000,000N test force produced ~0.02 m/s after 200ms) even though `set_force` was working correctly the whole time.

---

### Task 1: World generation — xacro → SDF conversion

**Files:**
- Create: `ros2_ws/src/cart_pole_gz_train/world_builder.py`
- Test: `ros2_ws/src/cart_pole_gz_train/verify_world_builder.py` (scratch script, not pytest)

**Interfaces:**
- Produces: `run_xacro(xacro_path: str) -> str` (returns URDF XML text), `convert_urdf_to_sdf(urdf_text: str, scratch_dir: str) -> str` (returns bare `<model>` SDF XML text)

- [ ] **Step 1: Write `world_builder.py` with the ROS-env subprocess helpers**

```python
import os
import subprocess

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
XACRO_REL_PATH = "ros2_ws/src/robot_description/robot/cart_pole.urdf.xacro"


def _run_in_ros_env(cmd: str) -> str:
    """Run a shell command with a colcon-built ros2_ws sourced and the venv
    stripped from PATH. Required because xacro (ament_index_python-based)
    fails under the venv's python3 and can't resolve the robot_description
    package unless ros2_ws/install/setup.bash has been sourced."""
    script = (
        'PATH=$(echo "$PATH" | tr ":" "\\n" | grep -v "\\.venv" | paste -sd:); '
        'unset VIRTUAL_ENV; '
        f'source {REPO_ROOT}/ros2_ws/install/setup.bash; '
        + cmd
    )
    result = subprocess.run(["bash", "-c", script], cwd=REPO_ROOT,
                             capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"command failed: {cmd!r}\n{result.stderr}")
    return result.stdout


def run_xacro() -> str:
    return _run_in_ros_env(f"xacro {XACRO_REL_PATH}")


def convert_urdf_to_sdf(urdf_text: str, scratch_dir: str) -> str:
    tmp_urdf = os.path.join(scratch_dir, "_generated.urdf")
    with open(tmp_urdf, "w") as f:
        f.write(urdf_text)
    return _run_in_ros_env(f"gz sdf -p {tmp_urdf}")
```

- [ ] **Step 2: Write the verification script**

```python
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from world_builder import run_xacro, convert_urdf_to_sdf

scratch_dir = os.path.dirname(__file__)
urdf = run_xacro()
assert "<robot" in urdf, "xacro did not produce a URDF"
model_sdf = convert_urdf_to_sdf(urdf, scratch_dir)
assert "cart_joint" in model_sdf and "pole_joint" in model_sdf, \
    "converted SDF is missing expected joints"
os.remove(os.path.join(scratch_dir, "_generated.urdf"))
print("PASS: xacro -> SDF conversion produced cart_joint and pole_joint")
```

- [ ] **Step 3: Run it and confirm it passes**

Run (from repo root):
```bash
PYTHONPATH=/usr/lib/python3/dist-packages uv run ros2_ws/src/cart_pole_gz_train/verify_world_builder.py
```
Expected: `PASS: xacro -> SDF conversion produced cart_joint and pole_joint`

If it instead fails with `PackageNotFoundError: robot_description`, `ros2_ws` has not been `colcon build`'t — build it first per `ros2_ws/src/CLAUDE.md` (`PATH=... VIRTUAL_ENV= colcon build --symlink-install` from `ros2_ws/`), then re-run.

- [ ] **Step 4: Commit**

```bash
git add ros2_ws/src/cart_pole_gz_train/world_builder.py ros2_ws/src/cart_pole_gz_train/verify_world_builder.py
git commit -m "feat(cart_pole_gz_train): add xacro-to-SDF conversion pipeline"
```

---

### Task 2: World generation — postprocessing and world wrapping

**Files:**
- Modify: `ros2_ws/src/cart_pole_gz_train/world_builder.py`
- Modify: `ros2_ws/src/cart_pole_gz_train/verify_world_builder.py`

**Interfaces:**
- Consumes: `convert_urdf_to_sdf`'s output (bare `<model>` SDF XML text) from Task 1
- Produces: `postprocess_model_sdf(model_sdf_text: str) -> str`, `wrap_in_world(model_sdf_text: str) -> str`, `generate_training_world(output_path: str) -> str` (writes the final world SDF to `output_path` and returns its text)

- [ ] **Step 1: Add postprocessing and world-wrap functions to `world_builder.py`**

```python
import xml.etree.ElementTree as ET

_PRIMITIVE_GEOMETRY = {
    "base_footprint": ("box", "0.4 0.4 0.6"),
    "cart_link": ("box", "0.3 0.3 0.15"),
    "pole_link": ("cylinder", None),  # handled specially below
}


def postprocess_model_sdf(model_sdf_text: str) -> str:
    """Strip visuals and replace mesh collision with primitives sized to
    roughly match the real robot_description meshes, and drop tip_link
    (mass 0.0001, physically negligible mount point) - keeps the physics
    parameters (mass/inertia/joint limits) sourced live from the xacro
    while collision shape stays hand-simplified, since headless training
    never renders and shouldn't pay for mesh-based collision."""
    root = ET.fromstring(model_sdf_text)
    model = root if root.tag == "model" else root.find("model")

    for link in list(model.findall("link")):
        name = link.get("name")
        for visual in link.findall("visual"):
            link.remove(visual)
        for collision in link.findall("collision"):
            geometry = collision.find("geometry")
            mesh = geometry.find("mesh")
            if mesh is None:
                continue
            geometry.remove(mesh)
            if name == "pole_link":
                cylinder = ET.SubElement(geometry, "cylinder")
                ET.SubElement(cylinder, "radius").text = "0.02"
                ET.SubElement(cylinder, "length").text = "1.0"
            else:
                shape, size = _PRIMITIVE_GEOMETRY.get(name, ("box", "0.2 0.2 0.2"))
                box = ET.SubElement(geometry, shape)
                ET.SubElement(box, "size").text = size

    for joint in list(model.findall("joint")):
        if joint.get("name") == "tip_joint":
            model.remove(joint)
    for link in list(model.findall("link")):
        if link.get("name") == "tip_link":
            model.remove(link)

    return ET.tostring(model, encoding="unicode")


def wrap_in_world(model_sdf_text: str) -> str:
    """Wrap the processed <model> in a full <world> TestFixture can load.

    The model gets an initial pose lifting it to z=2, clear of the ground
    plane - verified necessary: at z=0 the primitive collision boxes
    interpenetrate the ground plane on load, and the resulting contact
    forces dominate cart_joint's tiny prismatic motion, making commanded
    force look like it has no effect even though it's being applied
    correctly (confirmed by lifting the model and re-testing: a 1,000,000N
    force went from producing ~0.02 m/s after 200ms to producing the
    expected effort-limit-clamped ~11 m/s^2 acceleration).
    """
    model_sdf_text = model_sdf_text.replace(
        '<model name="cart_pole">',
        '<model name="cart_pole"><pose>0 0 2 0 0 0</pose>',
    )
    return f"""<?xml version="1.0" ?>
<sdf version="1.10">
  <world name="cart_pole_train">
    <physics name="1ms" type="ignored">
      <max_step_size>0.001</max_step_size>
      <real_time_factor>1.0</real_time_factor>
    </physics>
    <plugin filename="gz-sim-physics-system" name="gz::sim::systems::Physics"></plugin>
    <plugin filename="gz-sim-user-commands-system" name="gz::sim::systems::UserCommands"></plugin>
    <plugin filename="gz-sim-scene-broadcaster-system" name="gz::sim::systems::SceneBroadcaster"></plugin>
    <model name="ground_plane">
      <static>true</static>
      <link name="link">
        <collision name="collision">
          <geometry><plane><normal>0 0 1</normal><size>100 100</size></plane></geometry>
        </collision>
      </link>
    </model>
    {model_sdf_text}
  </world>
</sdf>
"""


def generate_training_world(output_path: str) -> str:
    scratch_dir = os.path.dirname(output_path)
    urdf = run_xacro()
    model_sdf = convert_urdf_to_sdf(urdf, scratch_dir)
    processed = postprocess_model_sdf(model_sdf)
    world = wrap_in_world(processed)
    with open(output_path, "w") as f:
        f.write(world)
    return world
```

- [ ] **Step 2: Extend the verification script to check postprocessing and validity**

```python
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from world_builder import generate_training_world

scratch_dir = os.path.dirname(__file__)
output_path = os.path.join(scratch_dir, "cart_pole_train.sdf")
world_text = generate_training_world(output_path)

assert "<mesh>" not in world_text, "primitive replacement left a mesh behind"
assert "tip_link" not in world_text, "tip_link should have been dropped"
assert 'cart_joint' in world_text and 'pole_joint' in world_text
assert '<world name="cart_pole_train">' in world_text
print("PASS: generated world has no meshes, no tip_link, correct joints")
```

- [ ] **Step 3: Run it, then validate the SDF with `gz sdf -k`**

```bash
PYTHONPATH=/usr/lib/python3/dist-packages uv run ros2_ws/src/cart_pole_gz_train/verify_world_builder.py
PATH=$(echo "$PATH" | tr ':' '\n' | grep -v '\.venv' | paste -sd:) VIRTUAL_ENV= gz sdf -k ros2_ws/src/cart_pole_gz_train/cart_pole_train.sdf
```
Expected: `PASS: generated world has no meshes, no tip_link, correct joints` then `Valid.`

- [ ] **Step 4: Commit**

```bash
git add ros2_ws/src/cart_pole_gz_train/world_builder.py ros2_ws/src/cart_pole_gz_train/verify_world_builder.py
git commit -m "feat(cart_pole_gz_train): postprocess SDF to primitives and wrap in a trainable world"
```

---

### Task 3: Verify joint-based force/state access against the real generated world

**Files:**
- Create: `ros2_ws/src/cart_pole_gz_train/verify_dynamics.py` (scratch script, not pytest)

**Interfaces:**
- Consumes: `generate_training_world` from Task 2

This task exists specifically to catch the ground-interpenetration failure mode from Task 2 in isolation, before any RL code is built on top of it — confirming force actually moves the cart the way `GzCartPoleScorer` (Task 4) will depend on.

- [ ] **Step 1: Write the verification script**

```python
import os
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)

import sys
sys.path.insert(0, os.path.dirname(__file__))
from world_builder import generate_training_world
from gz.sim8 import TestFixture, World, world_entity, Model, Joint

scratch_dir = os.path.dirname(__file__)
sdf_path = os.path.join(scratch_dir, "cart_pole_train.sdf")
generate_training_world(sdf_path)

state = {}


def ensure_init(ecm):
    if state:
        return
    world = World(world_entity(ecm))
    model = Model(world.model_by_name(ecm, "cart_pole"))
    cart_joint = Joint(model.joint_by_name(ecm, "cart_joint"))
    cart_joint.enable_position_check(ecm, True)
    cart_joint.enable_velocity_check(ecm, True)
    state["cart_joint"] = cart_joint
    state["max_force"] = cart_joint.effort_limits(ecm)[0]


def on_pre_update(info, ecm):
    if info.paused:
        return
    ensure_init(ecm)
    state["cart_joint"].set_force(ecm, [state["max_force"]])


def on_post_update(info, ecm):
    if info.paused:
        return
    ensure_init(ecm)
    state["last_pos"] = state["cart_joint"].position(ecm)[0]
    state["last_vel"] = state["cart_joint"].velocity(ecm)[0]


fixture = TestFixture(sdf_path)
fixture.on_pre_update(on_pre_update)
fixture.on_post_update(on_post_update)
fixture.finalize()
server = fixture.server()
server.run(True, 100, False)  # 100ms at max effort

expected_min_vel = 0.5  # m/s - well below the ~1.1 m/s (11 m/s^2 * 0.1s) expected
                        # at max clamped force, generous margin for engine/friction slop
assert state["last_vel"] > expected_min_vel, (
    f"cart barely moved (vel={state['last_vel']}) - check for ground-plane "
    f"interpenetration (Task 2's wrap_in_world lift) before assuming a code bug"
)
print(f"PASS: cart_joint reached vel={state['last_vel']:.3f} m/s "
      f"(pos={state['last_pos']:.3f}) after 100ms at max effort "
      f"({state['max_force']}N)")
```

- [ ] **Step 2: Run it**

```bash
PYTHONPATH=/usr/lib/python3/dist-packages uv run ros2_ws/src/cart_pole_gz_train/verify_dynamics.py
```
Expected: `PASS: cart_joint reached vel=... m/s (pos=...) after 100ms at max effort (30.0N)`

- [ ] **Step 3: Commit**

```bash
git add ros2_ws/src/cart_pole_gz_train/verify_dynamics.py
git commit -m "test(cart_pole_gz_train): verify joint force application against the generated world"
```

---

### Task 4: `GzCartPoleScorer` — TestFixture-driven reward/state scorer

**Files:**
- Create: `ros2_ws/src/cart_pole_gz_train/gz_scorer.py`

**Interfaces:**
- Consumes: `generate_training_world` from Task 2
- Produces: `GzCartPoleScorer` class with `step(action) -> (obs, reward, terminated, truncated, info)`, `reset() -> (obs, info)`, `close() -> None`. `obs` is a 4-element `np.float32` array `[cart_x, cart_vel, pole_pitch, pole_angular_vel]`.

- [ ] **Step 1: Write `gz_scorer.py`**

```python
import os
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)

import numpy as np
from gz.sim8 import TestFixture, World, world_entity, Model, Joint

from world_builder import generate_training_world

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
SDF_PATH = os.path.join(FILE_DIR, "cart_pole_train.sdf")

# Hard mechanical stop on cart_joint is +/-1m (verified: sustained max force
# pins position(ecm) at exactly 1.0). Terminate well inside that, not at the
# root project's unrelated 4.8m (this joint can never reach it).
CART_POSITION_LIMIT = 0.9
# pole_joint's declared limit is +/-1.7 rad, well outside this - reused from
# the root project since it's a generic "fallen over" bound, not tied to
# that project's specific geometry.
POLE_PITCH_LIMIT = 0.48


class GzCartPoleScorer:
    """Gazebo System that scores the world via joint-based ECM access -
    reads cart_joint/pole_joint position and velocity directly from their
    Joint components (real physics-engine velocity, no finite-difference
    estimation needed, unlike the root cart_pole/ project's wrench-based
    model which had no joint to read from)."""

    def __init__(self):
        self.command = None
        self._build_fixture()
        self.terminated = False
        self._initialized = False
        self.state = np.zeros(4, dtype=np.float32)
        self.reward = 0.0

    def _build_fixture(self):
        """Rebuild TestFixture/server from scratch on reset rather than
        calling server.reset_all() - same gz-sim8 bug as the root project:
        reset_all() desyncs the physics engine from the ECM while leaving
        entity IDs unchanged, silently breaking force application and state
        reads without looking like a stale-handle problem."""
        if not os.path.exists(SDF_PATH):
            generate_training_world(SDF_PATH)
        self.server = None
        self.fixture = None
        self.fixture = TestFixture(SDF_PATH)
        self.fixture.on_pre_update(self.on_pre_update)
        self.fixture.on_post_update(self.on_post_update)
        self.fixture.finalize()
        self.server = self.fixture.server()

    def _ensure_initialized(self, ecm):
        if self._initialized:
            return
        world = World(world_entity(ecm))
        model = Model(world.model_by_name(ecm, "cart_pole"))
        self.cart_joint = Joint(model.joint_by_name(ecm, "cart_joint"))
        self.pole_joint = Joint(model.joint_by_name(ecm, "pole_joint"))
        self.cart_joint.enable_position_check(ecm, True)
        self.cart_joint.enable_velocity_check(ecm, True)
        self.pole_joint.enable_position_check(ecm, True)
        self.pole_joint.enable_velocity_check(ecm, True)
        # cart_joint's effort limit is a hard actuator clamp enforced by the
        # physics engine (verified: 1,000,000N produces the same realized
        # acceleration as 30N) - read it live rather than hardcoding, so a
        # future xacro edit changing the limit doesn't silently desync this.
        self.max_force = self.cart_joint.effort_limits(ecm)[0]
        self._initialized = True

    def on_pre_update(self, info, ecm):
        if info.paused:
            return
        self._ensure_initialized(ecm)
        if self.command == 1:
            self.cart_joint.set_force(ecm, [self.max_force])
        elif self.command == 0:
            self.cart_joint.set_force(ecm, [-self.max_force])

    def on_post_update(self, info, ecm):
        if info.paused:
            return
        self._ensure_initialized(ecm)
        cart_pos = self.cart_joint.position(ecm)[0]
        cart_vel = self.cart_joint.velocity(ecm)[0]
        pole_pos = self.pole_joint.position(ecm)[0]
        pole_vel = self.pole_joint.velocity(ecm)[0]
        self.state = np.array([cart_pos, cart_vel, pole_pos, pole_vel], dtype=np.float32)
        if not self.terminated:
            self.terminated = (
                abs(pole_pos) > POLE_PITCH_LIMIT or abs(cart_pos) > CART_POSITION_LIMIT
            )
        self.reward = 0.0 if self.terminated else 1.0

    def step(self, action):
        self.command = action
        self.server.run(True, 5, False)
        return self.state, self.reward, self.terminated, False, {}

    def reset(self):
        self._build_fixture()
        self.command = None
        self.terminated = False
        self._initialized = False
        obs, _reward, _term, _trunc, _info = self.step(None)
        return obs, {}

    def close(self):
        self.server = None
        self.fixture = None
```

- [ ] **Step 2: Write a scratch script exercising `step`/`reset` and run it**

```python
import os
import sys
sys.path.insert(0, os.path.dirname(__file__))
from gz_scorer import GzCartPoleScorer

scorer = GzCartPoleScorer()
obs, _info = scorer.reset()
assert obs.shape == (4,)

for _ in range(20):
    obs, reward, terminated, truncated, _info = scorer.step(1)
    assert not terminated, "should not fall over in 20 steps of 5ms each"

assert obs[1] > 0.0, f"cart should have positive velocity after pushing right, got {obs[1]}"
scorer.close()
print(f"PASS: after 20 steps of action=1, cart_vel={obs[1]:.3f} (positive as expected)")
```

Run:
```bash
PYTHONPATH=/usr/lib/python3/dist-packages uv run ros2_ws/src/cart_pole_gz_train/verify_scorer.py
```
Expected: `PASS: after 20 steps of action=1, cart_vel=... (positive as expected)`

- [ ] **Step 3: Commit**

```bash
git add ros2_ws/src/cart_pole_gz_train/gz_scorer.py ros2_ws/src/cart_pole_gz_train/verify_scorer.py
git commit -m "feat(cart_pole_gz_train): add GzCartPoleScorer with joint-based ECM access"
```

---

### Task 5: Gymnasium wrapper and PPO training entrypoint

**Files:**
- Create: `ros2_ws/src/cart_pole_gz_train/train_cart_pole.py`

**Interfaces:**
- Consumes: `GzCartPoleScorer` from Task 4 (`step`/`reset`/`close`, `CART_POSITION_LIMIT`, `POLE_PITCH_LIMIT`)
- Produces: `CustomCartPoleGzTrain(gym.Env)`, `main()` — saves `cart_pole_gz_train_ppo.zip` in the same directory

- [ ] **Step 1: Write `train_cart_pole.py`**

```python
import os
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO

from gz_scorer import GzCartPoleScorer, CART_POSITION_LIMIT, POLE_PITCH_LIMIT

FILE_DIR = os.path.dirname(os.path.realpath(__file__))


class CustomCartPoleGzTrain(gym.Env):
    """Wraps GzCartPoleScorer for Gymnasium/SB3."""

    def __init__(self, env_config=None):
        self.env = GzCartPoleScorer()
        self.action_space = gym.spaces.Discrete(2)
        # Bounds reflect this robot's real joint limits (cart_joint +/-1m,
        # pole_joint +/-1.7 rad declared in the xacro), not the root
        # project's arbitrary bounds tuned for its unrelated model.
        self.observation_space = gym.spaces.Box(
            np.array([-1.0, -np.inf, -1.7, -np.inf], dtype=np.float32),
            np.array([1.0, np.inf, 1.7, np.inf], dtype=np.float32),
            (4,), np.float32,
        )

    def reset(self, seed=None, options=None):
        return self.env.reset()

    def step(self, action):
        return self.env.step(action)

    def close(self):
        self.env.close()


def main():
    env = CustomCartPoleGzTrain()
    model = PPO("MlpPolicy", env, verbose=1, device="auto")
    model.learn(total_timesteps=100_000)
    model_path = os.path.join(FILE_DIR, "cart_pole_gz_train_ppo")
    model.save(model_path)
    env.close()
    print(f"Training complete. Saved model to {model_path}.zip")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run a short smoke-test (reduced timesteps) before committing to the full run**

```bash
PYTHONPATH=/usr/lib/python3/dist-packages uv run python -c "
import sys, os
sys.path.insert(0, 'ros2_ws/src/cart_pole_gz_train')
os.chdir('ros2_ws/src/cart_pole_gz_train')
from train_cart_pole import CustomCartPoleGzTrain
from stable_baselines3 import PPO
env = CustomCartPoleGzTrain()
model = PPO('MlpPolicy', env, verbose=0, device='auto')
model.learn(total_timesteps=2000)
env.close()
print('PASS: 2000-timestep smoke test completed without error')
"
```
Expected: `PASS: 2000-timestep smoke test completed without error` (no assertion of learned quality yet — this only confirms the SB3/env plumbing doesn't crash)

- [ ] **Step 3: Run the full training script**

```bash
PYTHONPATH=/usr/lib/python3/dist-packages uv run ros2_ws/src/cart_pole_gz_train/train_cart_pole.py
```
Expected: SB3 logs episode reward trending upward over the run, ending with `Training complete. Saved model to .../cart_pole_gz_train_ppo.zip`

- [ ] **Step 4: Commit**

```bash
git add ros2_ws/src/cart_pole_gz_train/train_cart_pole.py
git commit -m "feat(cart_pole_gz_train): add PPO training entrypoint"
```

---

## Notes for the implementer

- `cart_pole_train.sdf` is regenerated by `GzCartPoleScorer._build_fixture()` only if it doesn't already exist at `SDF_PATH` — delete it manually to force regeneration after editing the xacro.
- If Task 3 or Task 4's verification shows near-zero velocity again despite Task 2's lift fix, re-check the model's pose offset didn't get lost — `wrap_in_world`'s string replace on `'<model name="cart_pole">'` is exact-match and silently no-ops if the tag's whitespace/attribute order changes upstream (e.g. after a `gz sdf` version bump reformats its output).
- `ros2_ws` must be `colcon build`'t before this script's first run (needed for `xacro`'s `ament_index_python` package resolution) — this is an existing artifact of the workspace, not something this feature needs to build itself.
