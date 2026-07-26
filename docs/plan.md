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
- The generated world's root model must be given an initial `<pose>` lifting it clear of the ground plane (verified root cause of a serious bug — see Task 2) — at `z=0` the primitive collision geometry interpenetrates the ground plane on load, and the resulting contact-resolution forces dominate the tiny prismatic joint's dynamics, making force application look completely broken (a 1,000,000N test force produced ~0.02 m/s after 200ms) even though `set_force` was working correctly the whole time. **(Amended — see Task 2's amendment: the lift is real but must be exactly `z=0.3`, half of `base_footprint`'s collision-box height, and it is only half the story; the pole's collision cylinder also needs a `-length/2` pose offset.)**

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

> **Amendment (post-implementation, from final whole-branch review):** the
> `<pose>0 0 2 0 0 0</pose>` lift in the `wrap_in_world` code below was a
> **misdiagnosis** that masked a geometry bug rather than fixing it.
> `postprocess_model_sdf` emitted the pole's collision cylinder with no
> `<pose>`, so it sat centred on `pole_link`'s origin (pole-frame z ∈
> [-0.5, +0.5]) while the real pole occupies z ∈ [-1, 0] (the xacro puts
> `tip_link` at `0 0 -1` and the CoM at `-0.489554` relative to
> `pole_link`; `pole_joint`'s 180°-about-y rotation then points that
> direction up in the world). Half the cylinder therefore hung *below* the
> joint, and on landing it speared straight through the ground plane.
> Measured (zero applied force, free settle, then 30N for 200ms):
>
> | spawn z | pole collision | rest `base_z` | rest `pole_pitch` | cart accel |
> | --- | --- | --- | --- | --- |
> | none | offset (fixed) | 0.006 | 0.000 | 0.10 m/s² |
> | 2.0 | centred (buggy) | 0.300 | **1.700 (at joint limit)** | 0.03 m/s² |
> | 2.0 | offset (fixed) | 0.300 | 0.000 | 10.27 m/s² |
> | **0.3** | **offset (fixed)** | **0.300** | **0.000** | **10.27 m/s²** |
>
> So: (a) a lift *is* genuinely needed — with none, the 88kg base spawns
> half-buried and the contact solver never frees it (`base_z` crawls from
> 0.002 to 0.006 over 6 simulated seconds) leaving `cart_joint` at ~1% of
> its proper authority; but (b) `z=2` costs a ~590ms (~118 env step)
> free-fall at the start of *every* episode, and with the buggy cylinder
> the landing dragged `pole_joint` to its ±1.7rad limit and pinned the
> cart — i.e. all prior training happened either mid-air or in a
> contact-jammed regime, never in real grounded cart-pole dynamics.
>
> Fix: give the pole collision `<pose>0 0 -0.5 0 0 0</pose>` (derived as
> `-length/2` from the cylinder's own declared length, not hardcoded), and
> set the spawn lift to `SPAWN_Z = 0.3`, derived as half of
> `base_footprint`'s 0.6m collision-box height so the box lands flush.
> With both, `base_z` reads exactly 0.3000 at t=1ms and never moves, and
> the cart accelerates at 10.27 m/s² against an ideal
> `effort_limit/cart_mass` of 30/2.7 = 11.11 m/s² (the shortfall is joint
> damping/friction plus the pole's reaction). `wrap_in_world` now also
> raises instead of silently no-op'ing if its exact-match anchor is
> missing, and `verify_world_builder.py` asserts both invariants
> structurally (parsed from the XML, tied to the geometry sizes) rather
> than string-matching a literal pose.

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

> **Amendment (post-implementation, human-approved):** the code below is the
> plan as originally written — vanilla PPO with no observation normalization.
> Building and training it exactly as written produced a policy that was
> statistically indistinguishable from a random policy (verified empirically:
> 15-episode random baseline mean 123.2/range 88-154 vs. the trained
> deterministic policy's identical 118-step episode every time). Root cause,
> confirmed by directly measuring raw observation magnitudes: `cart_vel` and
> especially `pole_vel` (the pole link's low rotational inertia) run roughly
> 10-30x larger in magnitude than `cart_pos`/`pole_pitch`, and SB3's PPO does
> not normalize observations by default — a freshly-initialized `MlpPolicy`
> expects roughly unit-scale inputs, so the velocity dimensions dominated the
> learning signal. Fix: wrap the env in `DummyVecEnv` + `VecNormalize`
> (`norm_obs=True`, `norm_reward=False` — reward is untouched, still the flat
> 0/1 from Tasks 2/4) before constructing `PPO`, and persist the running
> normalization statistics (`vecnormalize.pkl`) alongside the model, since any
> code that later feeds this policy observations (evaluation, future
> inference) must normalize them the same way or the fix silently reverts.
> Retrained for the full 100,000 timesteps under this change: the resulting
> deterministic policy now survives 453/453/453... steps across 15 evaluation
> episodes, vs. a freshly-reproduced random baseline of mean 121.2 (range
> 112-157) — see `.superpowers/sdd/task-5-tuning-report.md` for full
> measurements, the causality-control run (same model, without the
> normalization stats, collapses back to ~126 steps — random-baseline range),
> and self-review. A companion scratch script, `evaluate_policy.py`, was added
> to run this before/after comparison (random vs. trained episode-length,
> matching the repo's no-pytest convention).
>
> **Amendment 2 (final whole-branch review): the "453 vs 121" numbers above
> are superseded.** They were measured against the broken world described in
> Task 2's amendment, where every episode began with a ~590ms free-fall and
> ended in a contact-jammed landing — so they measured falling and jamming,
> not balancing. After fixing the pole collision offset and the spawn height,
> and wrapping the env in `Monitor` (SB3 only auto-wraps `Monitor` when the
> env handed to `PPO()` is not already a `VecEnv`, so passing the pre-built
> `DummyVecEnv` had silently suppressed all `ep_rew_mean`/`ep_len_mean`
> logging), the model was retrained from scratch for the same 100,000
> timesteps (708s wall clock). Real numbers:
>
> - **Training curve** (now visible): `ep_len_mean` 143 → 175 → 211 → 254 →
>   304 → 360 → 386 → 432 at 45k timesteps, then flat at **438** from ~50k
>   onwards. `ep_rew_mean` tracks it exactly (flat +1/step reward).
> - **Random baseline**, 15 episodes: mean **154.1** steps, range 99-269,
>   std 53.6, all 15 terminating on the pole-angle limit.
> - **Trained deterministic policy**, 15 episodes: **≥2000 steps every
>   episode** — it never terminates. Runs were cut off at a 2000-step
>   (10s sim) cap; at the cap the cart sits at x=+0.108 (limit 0.9) with
>   pole_pitch=-0.0004 rad (limit 0.48), i.e. genuinely balanced, not
>   drifting or stuck. So the honest comparison is **154.1 → >2000 steps
>   (>13x, lower bound)**, not 121 → 453.
> - **Causality control re-run** on the retrained model: same policy, same
>   env, `vecnormalize.pkl` *not* loaded → collapses to exactly 246 steps
>   every episode, terminating on the cart-position limit. Still inside the
>   random baseline's 99-269 range, so the VecNormalize dependency is
>   unchanged and real. `evaluate_policy.py` now hard-errors on missing
>   stats instead of warning and continuing.
>
> Two artifacts of the env being fully deterministic (no initial-state
> randomization — `reset(seed=...)` is ignored) show up in these numbers and
> are worth knowing before reading too much into them: every trained episode
> is bit-identical, and the 438-step training plateau is the *stochastic*
> rollout policy's score (it occasionally samples the losing action),
> not the deterministic policy's, which is unbounded. Randomizing the
> initial state remains an open follow-up.
>
> ~~Note also that `evaluate_policy.py` has **no step cap**, so
> `evaluate_policy.py trained` now runs forever against this policy; the
> numbers above came from a capped scratch harness. Adding a cap is an
> open follow-up.~~ **Amended (final whole-branch review, round 2):** a step
> cap (`MAX_STEPS`/`--max-steps`, default 2000) has since landed in
> `evaluate_policy.py` — this is no longer an open follow-up. A subsequent
> review round found and fixed two bugs in that cap's implementation: the
> cap check ran *before* the `terminated`/`done` check, so a genuine fall on
> exactly the cap'th step was misreported as "reached step cap (still
> balancing)"; and the "still balancing" claim wasn't backed by printing the
> actual `cart_x`/`pole_pitch` at the cap. Both are fixed, and the cap is now
> a `--max-steps` CLI flag rather than a hardcoded constant.

**Files:**
- Create: `ros2_ws/src/cart_pole_gz_train/train_cart_pole.py`
- Create: `ros2_ws/src/cart_pole_gz_train/evaluate_policy.py` (scratch episode-length measurement script, added post-implementation)

**Interfaces:**
- Consumes: `GzCartPoleScorer` from Task 4 (`step`/`reset`/`close`, `CART_POSITION_LIMIT`, `POLE_PITCH_LIMIT`)
- Produces: `CustomCartPoleGzTrain(gym.Env)`, `main()` — saves `cart_pole_gz_train_ppo.zip` **and `vecnormalize.pkl`** in the same directory

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

### Task 6: Verify `reset.model_only` preserves `JointStatePublisher`'s topic

**Files:**
- Create: `ros2_ws/src/cart_pole_gz_train/verify_reset_preserves_joint_state.py` (scratch script, not pytest)

**Interfaces:**
- Consumes: `generate_training_world` from `world_builder.py` (Tasks 1-2)

This task exists to check a specific risk before Task 7 depends on it: `ros2_ws/src/CLAUDE.md`'s `commander` notes document that `WorldControl.reset.all = True` permanently stops `JointStatePublisher` from advertising its gz-transport topic after the first reset (entities get torn down and recreated). That finding was against a different world (a model dynamically spawned via `ros_gz_sim`'s `create`, under `ros_gz_bridge`) — this task confirms `reset.model_only = True` avoids the same failure against *this* world (declared whole in the generated `cart_pole_train.sdf`, loaded directly by `gz sim -s -r`, no dynamic spawn).

The following was already confirmed live (via `gz topic -l`/`-i`/`-e` against a running `cart_pole_train.sdf` server) and can be taken as given: the server exposes `/model/cart_pole/joint/cart_joint/cmd_force` (`gz.msgs.Double`, subscribed by `ApplyJointForce`), `/world/cart_pole_train/model/cart_pole/joint_state` (`gz.msgs.Model`, published by `JointStatePublisher`, each `joint` entry has `name` and `axis1.position`/`axis1.velocity`), and `/world/cart_pole_train/control` (`gz.msgs.WorldControl` → `gz.msgs.Boolean`). `gz.transport13`/`gz.msgs10` import fine without the `ctypes.CDLL(".../libgz-sim8.so", ...)` preload (confirmed empirically) — that preload is specifically for `gz.sim8`'s `TestFixture`/ECM bindings, which this script never touches.

- [ ] **Step 1: Write the verification script**

```python
import os
import sys
import subprocess
import time

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, FILE_DIR)

from world_builder import generate_training_world

from gz.transport13 import Node
from gz.msgs10.model_pb2 import Model
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean

sdf_path = os.path.join(FILE_DIR, "cart_pole_train.sdf")
generate_training_world(sdf_path)

gz_server = subprocess.Popen(["gz", "sim", "-s", "-r", sdf_path])
try:
    time.sleep(4)  # let the server come up and start publishing

    node = Node()
    counts = {"before": 0, "after": 0}
    phase = {"value": "before"}

    def on_joint_state(_msg):
        counts[phase["value"]] += 1

    node.subscribe(
        Model, "/world/cart_pole_train/model/cart_pole/joint_state",
        on_joint_state)

    time.sleep(2)
    assert counts["before"] > 0, (
        "no joint_state messages received before reset - is the server up "
        "and is JointStatePublisher declared in the generated SDF?"
    )

    request = WorldControl()
    request.reset.model_only = True
    ok, _resp = node.request(
        "/world/cart_pole_train/control", request, WorldControl, Boolean, 5000)
    assert ok, "reset.model_only request failed"

    phase["value"] = "after"
    time.sleep(2)

    assert counts["after"] > 0, (
        f"JointStatePublisher stopped publishing after reset.model_only=True "
        f"(before={counts['before']} msgs, after={counts['after']} msgs) - "
        f"the same bug ros2_ws/src/CLAUDE.md documents for reset.all in the "
        f"commander package. run_inference.py (Task 7) would need a "
        f"different reset strategy (e.g. re-subscribing after reset, or "
        f"driving joints back via set_pose_vector instead of "
        f"WorldControl.reset) - do not proceed to Task 7 until this is "
        f"resolved."
    )
    print(
        f"PASS: joint_state kept publishing after reset.model_only=True "
        f"(before={counts['before']} msgs, after={counts['after']} msgs)"
    )
finally:
    gz_server.terminate()
    gz_server.wait(timeout=10)
```

- [ ] **Step 2: Run it**

```bash
PYTHONPATH=/usr/lib/python3/dist-packages uv run ros2_ws/src/cart_pole_gz_train/verify_reset_preserves_joint_state.py
```
Expected: `PASS: joint_state kept publishing after reset.model_only=True (before=... msgs, after=... msgs)`

If it instead fails the `counts["after"] > 0` assertion, stop — Task 7 cannot use `WorldControl.reset` for episode resets as designed, and the reset strategy needs to be revisited before continuing (see the assertion message for alternatives to try).

- [ ] **Step 3: Commit**

```bash
git add ros2_ws/src/cart_pole_gz_train/verify_reset_preserves_joint_state.py
git commit -m "test(cart_pole_gz_train): verify reset.model_only keeps joint_state publishing"
```

---

### Task 7: `run_inference.py` — live-GUI inference over Gazebo transport

**Files:**
- Create: `ros2_ws/src/cart_pole_gz_train/run_inference.py`

**Interfaces:**
- Consumes: `SDF_PATH`, `CART_POSITION_LIMIT`, `POLE_PITCH_LIMIT` from `gz_scorer.py` (Task 4); `generate_training_world` from `world_builder.py` (Tasks 1-2); `cart_pole_gz_train_ppo.zip`/`vecnormalize.pkl` produced by `train_cart_pole.py` (Task 5); the transport interface and `reset.model_only` behavior confirmed by Task 6.
- Produces: `run_inference.py`, runnable the same way as the trainer: `PYTHONPATH=/usr/lib/python3/dist-packages uv run ros2_ws/src/cart_pole_gz_train/run_inference.py`.

This is the joint-based, `ros2_ws` analogue of `cart_pole/cart_pole_env.py`'s `run_inference()` — simpler in three ways because this robot has real joints instead of a bare chassis link: true physics velocity is available directly (no finite-difference estimator), `ApplyJointForce`'s `cmd_force` just holds its last value (no persistent-wrench delta/net-force bookkeeping), and joint state arrives via a plain pub/sub topic (no FNV-hashed ECS-state decoding).

- [ ] **Step 1: Write `run_inference.py`**

```python
import argparse
import os
import sys
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)

import subprocess
import time
import xml.etree.ElementTree as ET

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, FILE_DIR)

from world_builder import generate_training_world
from gz_scorer import SDF_PATH, CART_POSITION_LIMIT, POLE_PITCH_LIMIT

from gz.transport13 import Node
from gz.msgs10.double_pb2 import Double
from gz.msgs10.model_pb2 import Model
from gz.msgs10.world_control_pb2 import WorldControl
from gz.msgs10.boolean_pb2 import Boolean

WORLD_NAME = "cart_pole_train"
MODEL_NAME = "cart_pole"
STEP_PERIOD = 0.005  # matches training's 5 x 1ms action cadence
MAX_ITERATIONS = 50000  # ~250s, mirrors cart_pole_env.py's run_inference bound


class _ObsSpaceStub(gym.Env):
    """Carries only the observation/action space VecNormalize.load needs to
    shape-check against - never calls into gz, so constructing it can't
    double-register on the training world's transport name alongside the
    live inference server (unlike instantiating CustomCartPoleGzTrain,
    whose __init__ builds a real GzCartPoleScorer)."""

    def __init__(self):
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            np.array([-CART_POSITION_LIMIT, -np.inf, -POLE_PITCH_LIMIT, -np.inf], dtype=np.float32),
            np.array([CART_POSITION_LIMIT, np.inf, POLE_PITCH_LIMIT, np.inf], dtype=np.float32),
            (4,), np.float32,
        )

    def reset(self, seed=None, options=None):
        raise NotImplementedError("stub env is never actually reset/stepped")

    def step(self, action):
        raise NotImplementedError("stub env is never actually reset/stepped")


def _load_normalizer(vecnorm_path):
    if not os.path.exists(vecnorm_path):
        raise SystemExit(
            f"ERROR: VecNormalize stats not found at {vecnorm_path!r}.\n"
            "The trained policy expects observations normalized with the "
            "running statistics saved during training; running inference "
            "without them silently reproduces random-baseline performance. "
            "Re-run train_cart_pole.py (which writes vecnormalize.pkl next "
            "to the model) or pass --vecnorm explicitly."
        )
    venv = DummyVecEnv([lambda: _ObsSpaceStub()])
    venv = VecNormalize.load(vecnorm_path, venv)
    venv.training = False
    venv.norm_reward = False
    return venv


def _read_effort_limit(sdf_path, joint_name):
    root = ET.parse(sdf_path).getroot()
    effort_el = root.find(f".//joint[@name='{joint_name}']/axis/limit/effort")
    if effort_el is None:
        raise RuntimeError(
            f"could not find an effort limit for joint {joint_name!r} in "
            f"{sdf_path} - did the xacro or gz sdf conversion change its "
            f"structure?"
        )
    return float(effort_el.text)


def _kill_stale_gz_processes():
    """Terminate any gz sim server/GUI left over from a prior run - a
    leftover server registers on the same transport bus under the same
    world name as the one about to be launched, and the new GUI can attach
    to that stale instance instead of ours."""
    subprocess.run(["pkill", "-f", "gz sim"], check=False)
    time.sleep(1)


def _reset_world(node):
    """Reset via reset.model_only, not reset.all - see Task 6's
    verify_reset_preserves_joint_state.py: reset.all tears down and
    recreates entities, which permanently stops JointStatePublisher from
    advertising its topic after the first reset. model_only resets
    poses/velocities without that teardown."""
    request = WorldControl()
    request.reset.model_only = True
    ok, _resp = node.request(
        f"/world/{WORLD_NAME}/control", request, WorldControl, Boolean, 5000)
    if not ok:
        raise RuntimeError("world reset request failed")


def _wait_for_obs(latest, timeout=2.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if latest["obs"] is not None:
            return
        time.sleep(0.01)
    raise RuntimeError(
        "no joint_state message received - is JointStatePublisher declared "
        "in the generated SDF?"
    )


def run_inference(model, normalizer, effort_limit):
    _kill_stale_gz_processes()

    gz_server = None
    gz_gui = None
    try:
        print("Launching Gazebo server...")
        gz_server = subprocess.Popen(["gz", "sim", "-s", "-r", SDF_PATH])
        time.sleep(3)

        print("Launching Gazebo GUI...")
        gz_gui = subprocess.Popen(["gz", "sim", "-g"])
        time.sleep(5)  # wait for GUI to connect

        node = Node()
        force_pub = node.advertise(
            f"/model/{MODEL_NAME}/joint/cart_joint/cmd_force", Double)

        latest = {"obs": None}

        def on_joint_state(msg):
            positions = {j.name: j.axis1.position for j in msg.joint}
            velocities = {j.name: j.axis1.velocity for j in msg.joint}
            try:
                obs = np.array([
                    positions["cart_joint"], velocities["cart_joint"],
                    positions["pole_joint"], velocities["pole_joint"],
                ], dtype=np.float32)
            except KeyError:
                return  # mid-reset snapshot missing a joint; skip it
            latest["obs"] = obs

        node.subscribe(
            Model, f"/world/{WORLD_NAME}/model/{MODEL_NAME}/joint_state",
            on_joint_state)

        print("Waiting for first joint_state message...")
        _wait_for_obs(latest)

        print("Running inference with GUI... Press Ctrl+C to stop.")
        episode_start = time.monotonic()
        for _ in range(MAX_ITERATIONS):
            loop_start = time.monotonic()

            obs = latest["obs"]
            normalized = normalizer.normalize_obs(obs.reshape(1, -1))
            action, _state = model.predict(normalized, deterministic=True)
            action = int(action[0])

            force_msg = Double()
            force_msg.data = effort_limit if action == 1 else -effort_limit
            force_pub.publish(force_msg)

            cart_pos, _cart_vel, pole_pos, _pole_vel = obs
            if abs(cart_pos) > CART_POSITION_LIMIT or abs(pole_pos) > POLE_PITCH_LIMIT:
                episode_len = time.monotonic() - episode_start
                print(f"Cart-pole out of bounds after {episode_len:.2f}s, resetting world...")
                _reset_world(node)
                latest["obs"] = None
                time.sleep(0.5)  # let the reset propagate before next read
                _wait_for_obs(latest)
                episode_start = time.monotonic()

            elapsed = time.monotonic() - loop_start
            remaining = STEP_PERIOD - elapsed
            if remaining > 0:
                time.sleep(remaining)

    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        for proc in (gz_gui, gz_server):
            if proc is not None:
                proc.terminate()
        for proc in (gz_gui, gz_server):
            if proc is not None:
                proc.wait()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default=os.path.join(FILE_DIR, "cart_pole_gz_train_ppo"))
    parser.add_argument(
        "--vecnorm", default=os.path.join(FILE_DIR, "vecnormalize.pkl"))
    args = parser.parse_args()

    generate_training_world(SDF_PATH)
    effort_limit = _read_effort_limit(SDF_PATH, "cart_joint")
    print(f"Read cart_joint effort limit from generated SDF: {effort_limit}N")

    model = PPO.load(args.model)
    print(f"Loaded model from {args.model}.zip")
    normalizer = _load_normalizer(args.vecnorm)
    print(f"Loaded VecNormalize stats from {args.vecnorm}")

    run_inference(model, normalizer, effort_limit)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it and watch the GUI**

```bash
PYTHONPATH=/usr/lib/python3/dist-packages uv run ros2_ws/src/cart_pole_gz_train/run_inference.py
```
Expected console output: `Read cart_joint effort limit from generated SDF: 30.0N`, `Loaded model from .../cart_pole_gz_train_ppo.zip`, `Loaded VecNormalize stats from .../vecnormalize.pkl`, `Launching Gazebo server...`, `Launching Gazebo GUI...`, `Waiting for first joint_state message...`, `Running inference with GUI... Press Ctrl+C to stop.`

In the GUI window: the cart-pole should visibly balance (small, roughly-symmetric back-and-forth cart motion, pole staying near-vertical), matching the deterministic policy's training-time behavior (Task 5's amendment: ≥2000 steps/episode, never falling). If the pole falls over, the console should print `Cart-pole out of bounds after ...s, resetting world...` and the model should visibly reset to its upright starting pose and keep balancing — confirming Task 6's `reset.model_only` finding holds inside the full script too. Let it run for at least 15-20 seconds, then press Ctrl+C; expect `Stopping...` and a clean exit with no leftover `gz sim` processes (check with `ps aux | grep "gz sim"` — matching against just the substring `gz sim`, not the literal command run in this check, so the check itself never self-matches).

- [ ] **Step 3: Commit**

```bash
git add ros2_ws/src/cart_pole_gz_train/run_inference.py
git commit -m "feat(cart_pole_gz_train): add live-GUI inference over Gazebo transport"
```

---

## Notes for the implementer

- ~~`cart_pole_train.sdf` is regenerated by `GzCartPoleScorer._build_fixture()` only if it doesn't already exist at `SDF_PATH` — delete it manually to force regeneration after editing the xacro.~~ **Amended:** existence-gating meant a leftover file from any previous process (possibly a different checkout) was silently reused forever, contradicting the PRD's "no silent fallback to a stale cached SDF". `gz_scorer.ensure_world_generated()` now regenerates it exactly once per process, from `GzCartPoleScorer.__init__` — not on every `reset()` (thousands per run, and the world is fixed for a run's lifetime) and not conditional on the file existing.
- If Task 3 or Task 4's verification shows near-zero velocity again despite Task 2's lift fix, re-check the model's pose offset didn't get lost — `wrap_in_world`'s string replace on `'<model name="cart_pole">'` is exact-match and silently no-ops if the tag's whitespace/attribute order changes upstream (e.g. after a `gz sdf` version bump reformats its output). **Amended:** it now raises instead of silently no-op'ing, and `verify_dynamics.py` asserts the resting height directly.
- `ros2_ws` must be `colcon build`'t before this script's first run (needed for `xacro`'s `ament_index_python` package resolution) — this is an existing artifact of the workspace, not something this feature needs to build itself.
