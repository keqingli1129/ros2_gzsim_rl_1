# True 5ms-Synced Inference Observations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace `run_inference()`'s async `dynamic_pose/info` subscription (capped at ~17ms/message) with synchronous, on-demand `/world/cart_pole/state` requests, so inference reads a genuinely fresh observation every ~5ms action — matching training's actual step cadence instead of reusing a stale cached pose across 3+ actions.

**Architecture:** `/world/cart_pole/state` is a request/response service (`Empty` → `SerializedStepMap`) that returns the full ECS snapshot on demand, unthrottled by any fixed publish rate. Each entity's components are serialized generically as `(type_id: int64, bytes)` pairs, where `type_id` is `gz::common::hash64()` of the component's registered type-name string. We replicate that hash in Python to identify the `Name` and `Pose` components, decode their bytes (plain UTF-8 text — `Name` is the raw string, `Pose` is `"x y z roll pitch yaw"` from `gz::math::Pose3d::operator<<`), compose model+link local poses into world frame (reusing the existing quaternion helpers), and finite-difference velocity against the response's own `sim_time` (not wall-clock). The whole thing runs synchronously inside the main loop — no subscriber thread, no shared-state lock.

**Tech Stack:** `gz.transport13.Node.request`, `gz.msgs10.{empty_pb2,serialized_map_pb2,world_stats_pb2}`, pure-Python FNV-1a 64 hashing, existing quaternion helpers in `cart_pole_env.py`.

## Global Constraints

- All changes go in `cart_pole_env.py` — this is a deliberate single-file project (see CLAUDE.md); do not split into new modules.
- Do not touch `GzRewardScorer` / `CustomCartPole` / training path (`cart_pole_env.py:38-167`) — this plan is inference-only.
- Keep the existing out-of-bounds `_reset_world` crash fix and its termination bounds (`|pitch| > 0.48`, `|cart_x| > 4.8`) exactly as-is.
- This repo has no test framework (`pyproject.toml` / CLAUDE.md confirm no pytest, no linter configured). "Tests" below are standalone verification scripts run against a live server, exactly like the manual verification already used to build this plan — not pytest.
- Every gz sim server used for verification must be launched with `PYTHONPATH=/usr/lib/python3/dist-packages` (system Gazebo bindings, per CLAUDE.md) and torn down afterward.
- **Footgun already hit this session:** `pkill -f "gz sim"` matches the literal substring "gz sim" in its own invoking shell command line and kills its own wrapper process. Never invoke cleanup with the literal substring "gz sim" in the pattern — use `pkill -f "g[z] sim"` (regex character class breaks the self-match) or track PIDs explicitly and `kill <pid>`.
- Every new function/constant introduced here must actually be wired into `run_inference` by Task 4 — no orphaned helpers left unused at the end of this plan.

---

## File Structure

Single file, `cart_pole_env.py`. New pieces, in the order they'll be added (all in the "Inference-time" section, roughly `cart_pole_env.py:169-304` today):

- New imports: `gz.msgs10.empty_pb2.Empty`, `gz.msgs10.serialized_map_pb2.SerializedStepMap`.
- `_gz_component_hash(type_name)` + `_NAME_COMPONENT_ID` / `_POSE_COMPONENT_ID` constants — component-type ID resolution (Task 1).
- `_query_world_state(node)` — request + raw decode into `(sim_time, names_by_id, pose_text_by_id)` (Task 2).
- `_euler_to_quat(roll, pitch, yaw)` + `_resolve_target_entities(node)` + `_world_frame_pose(names_by_id, pose_text_by_id, entity_ids)` — entity resolution and world-frame composition (Task 3).
- `run_inference` rewritten to poll synchronously instead of subscribing; `pose_cb`, `_state_lock`, `_latest_state`, `_prev_pose_state`, the `Pose_V` import, and the `threading` import are all deleted as dead code once nothing references them (Task 4).
- Final validation script confirming reset frequency and loop timing (Task 5, not committed to the repo — a scratch script per its step).

---

### Task 1: Component-type ID resolution (FNV-1a 64 hash)

**Files:**
- Modify: `cart_pole_env.py` — add after `_pitch_from_quat` (currently ending at `cart_pole_env.py:215`)

**Interfaces:**
- Produces: `_gz_component_hash(type_name: str) -> int` (signed 64-bit, matches protobuf map key sign convention), `_NAME_COMPONENT_ID: int`, `_POSE_COMPONENT_ID: int` — consumed by Task 2's decode loop.

- [ ] **Step 1: Add the hash function and constants**

Insert immediately after `_pitch_from_quat` (before `def pose_cb(msg):`):

```python
def _gz_component_hash(type_name):
    """Replicate gz::common::hash64() (FNV-1a, 64-bit), used by gz-sim's
    component Factory (components/Factory.hh) to assign each registered
    component type a runtime ComponentTypeId. SerializedComponent.type on
    the /world/<world>/state service carries this same value, so decoding
    that service's response requires reproducing the hash here rather than
    hardcoding IDs (they're derived from the type name string, not stable
    across gz-sim versions/builds otherwise).
    """
    prime = 0x100000001b3
    h = 0xcbf29ce484222325
    mask = (1 << 64) - 1
    for byte in type_name.encode("utf-8"):
        h ^= byte
        h = (h * prime) & mask
    if h >= (1 << 63):
        # SerializedComponent.type is a signed int64 field on the wire;
        # values with the high bit set decode as negative.
        h -= (1 << 64)
    return h

_NAME_COMPONENT_ID = _gz_component_hash("gz_sim_components.Name")
_POSE_COMPONENT_ID = _gz_component_hash("gz_sim_components.Pose")
```

- [ ] **Step 2: Verify the hash against known-good values**

Run:

```bash
PYTHONPATH=/usr/lib/python3/dist-packages python3 -c "
import cart_pole_env as c
print(c._NAME_COMPONENT_ID)
print(c._POSE_COMPONENT_ID)
"
```

Expected output (verified live against a running `cart_pole.sdf` server during this plan's research — these are the actual on-wire component-type IDs for gz-sim 8 / Harmonic):

```
-998690179357215250
-7527930132038368260
```

Note: importing `cart_pole_env` at module scope will attempt to `ctypes.CDLL(...)` and define `GzRewardScorer` etc., but does **not** execute training/inference (those are gated behind `if __name__ == "__main__":`), so this import is safe to run standalone.

- [ ] **Step 3: Commit**

```bash
git add cart_pole_env.py
git commit -m "Add gz-sim component-type hash for decoding raw ECS state"
```

---

### Task 2: Synchronous world-state query + raw decode

**Files:**
- Modify: `cart_pole_env.py` — add new imports near the existing "Inference-time imports" block (`cart_pole_env.py:19-27`); add `_query_world_state` after Task 1's constants.

**Interfaces:**
- Consumes: `_NAME_COMPONENT_ID`, `_POSE_COMPONENT_ID` (Task 1).
- Produces: `_query_world_state(node) -> (sim_time: float, names_by_id: dict[int, str], pose_text_by_id: dict[int, str])` — consumed by Task 3.

- [ ] **Step 1: Add the two new imports**

In the existing import block, right after `from gz.msgs10.boolean_pb2 import Boolean`:

```python
from gz.msgs10.empty_pb2 import Empty
from gz.msgs10.serialized_map_pb2 import SerializedStepMap
```

- [ ] **Step 2: Add `_query_world_state`**

Insert after the `_POSE_COMPONENT_ID = ...` line from Task 1:

```python
def _query_world_state(node):
    """Synchronously fetch the full ECS snapshot from /world/cart_pole/state.

    Unlike the dynamic_pose/info topic (published on a fixed ~17ms timer by
    the SceneBroadcaster plugin, independent of when we actually need an
    observation), this is a request/response service we can call whenever we
    want a fresh reading - so the caller's own loop cadence becomes the
    observation rate, not the publisher's.
    """
    ok, resp = node.request(
        "/world/cart_pole/state", Empty(), Empty, SerializedStepMap, 2000)
    if not ok:
        return None, {}, {}

    sim_time = resp.stats.sim_time.sec + resp.stats.sim_time.nsec * 1e-9
    names_by_id = {}
    pose_text_by_id = {}
    for entity_id, entity_map in resp.state.entities.items():
        for comp_id, comp in entity_map.components.items():
            if comp_id == _NAME_COMPONENT_ID:
                names_by_id[entity_id] = comp.component.decode("utf-8")
            elif comp_id == _POSE_COMPONENT_ID:
                pose_text_by_id[entity_id] = comp.component.decode("utf-8")
    return sim_time, names_by_id, pose_text_by_id
```

- [ ] **Step 3: Verify against a live server**

Terminal A:

```bash
cd /home/keqing-li/Documents/ros2_gzsim_rl_1
PYTHONPATH=/usr/lib/python3/dist-packages gz sim -s -r cart_pole.sdf
```

Terminal B (after waiting ~3s for the server to come up):

```bash
PYTHONPATH=/usr/lib/python3/dist-packages python3 -c "
import cart_pole_env as c
from gz.transport13 import Node
n = Node()
sim_time, names, poses = c._query_world_state(n)
print('sim_time', sim_time)
for eid, nm in names.items():
    if nm in ('vehicle_green', 'chassis', 'pole'):
        print(eid, nm, '->', poses.get(eid))
"
```

Expected: three lines print, one each for `vehicle_green`, `chassis`, `pole`, each followed by 6 space-separated floats (`x y z roll pitch yaw`) — e.g.:

```
sim_time 1.062
8 vehicle_green -> -9.21137e-06 -1.48505e-12 0.324999 -2.21281e-14 6.65084e-06 1.63826e-11
15 chassis -> -0.151424 -1.22408e-12 0.175 7.90466e-15 -6.65756e-06 3.32532e-12
9 pole -> -0.151427 0 1.5 0 0 0
```

(Exact numbers will differ run to run — physics has been running freely since server start — but three matching entities with 6-float pose text each is the pass condition.) Stop the Terminal A server with Ctrl+C afterward.

- [ ] **Step 4: Commit**

```bash
git add cart_pole_env.py
git commit -m "Add synchronous /world/cart_pole/state query and decode"
```

---

### Task 3: Entity resolution + world-frame pose composition

**Files:**
- Modify: `cart_pole_env.py` — add after `_query_world_state`.

**Interfaces:**
- Consumes: `_query_world_state` (Task 2), existing `_quat_mult`, `_quat_rotate`, `_pitch_from_quat` (`cart_pole_env.py:186-215`, unchanged).
- Produces: `_euler_to_quat(roll, pitch, yaw) -> (w, x, y, z)`, `_resolve_target_entities(node) -> dict[str, int]` (keys `"vehicle_green"`, `"chassis"`, `"pole"`), `_world_frame_pose(names_by_id, pose_text_by_id, entity_ids) -> (cart_pose: float, pole_pose: float) | None` — consumed by Task 4.

- [ ] **Step 1: Add Euler-to-quaternion conversion**

The `Pose` component's text is `roll pitch yaw` (Euler angles, not a quaternion), because `gz::math::Pose3d::operator<<` prints `Quaternion::Euler()`. Composing model + link rotations needs quaternions (Euler composition isn't just addition), so convert back using gz-math's exact `Quaternion::SetFromEuler` formula (`gz/math/Quaternion.hh`) to avoid any convention mismatch:

```python
def _euler_to_quat(roll, pitch, yaw):
    """Convert roll/pitch/yaw (radians) to a (w, x, y, z) quaternion.

    Mirrors gz::math::Quaternion<T>::SetFromEuler exactly (see
    gz/math7/gz/math/Quaternion.hh) so composing poses decoded from the ECS
    Pose component's Euler-angle text matches what Link.world_pose() would
    have produced from the same underlying quaternion during training.
    """
    phi, the, psi = roll / 2.0, pitch / 2.0, yaw / 2.0
    cphi, sphi = math.cos(phi), math.sin(phi)
    cthe, sthe = math.cos(the), math.sin(the)
    cpsi, spsi = math.cos(psi), math.sin(psi)
    w = cphi * cthe * cpsi + sphi * sthe * spsi
    x = sphi * cthe * cpsi - cphi * sthe * spsi
    y = cphi * sthe * cpsi + sphi * cthe * spsi
    z = cphi * cthe * spsi - sphi * sthe * cpsi
    return (w, x, y, z)
```

- [ ] **Step 2: Add entity-ID resolution**

Entity IDs are stable for the lifetime of a running server and across `WorldControl.reset.all` (verified live: same IDs before/after a reset call), so resolve once by name and reuse:

```python
def _resolve_target_entities(node):
    """Look up entity IDs for vehicle_green/chassis/pole by name, once.

    Entity IDs are assigned at world-load time from the SDF and stay fixed
    across a reset (confirmed live: same IDs before/after
    WorldControl.reset.all=True), so this only needs to run once at
    startup, not after every reset.
    """
    _, names_by_id, _ = _query_world_state(node)
    wanted = {"vehicle_green", "chassis", "pole"}
    ids = {name: eid for eid, name in names_by_id.items() if name in wanted}
    missing = wanted - ids.keys()
    if missing:
        raise RuntimeError(f"Could not resolve entities: {missing}")
    return ids
```

- [ ] **Step 3: Add world-frame pose composition**

Mirrors the frame composition the deleted `pose_cb` used to do (model pose in world frame; every link pose relative to the model), just fed from decoded ECS text instead of `Pose_V`:

```python
def _parse_pose_text(text):
    x, y, z, roll, pitch, yaw = (float(v) for v in text.split())
    return (x, y, z), _euler_to_quat(roll, pitch, yaw)


def _world_frame_pose(names_by_id, pose_text_by_id, entity_ids):
    """Compose (cart_pose, pole_pose) in world frame from one state query.

    Returns None if any of the three tracked entities' pose text is
    missing from this particular response (e.g. mid-reset).
    """
    try:
        model_pos, model_quat = _parse_pose_text(
            pose_text_by_id[entity_ids["vehicle_green"]])
        chassis_local_pos, _ = _parse_pose_text(
            pose_text_by_id[entity_ids["chassis"]])
        _, pole_local_quat = _parse_pose_text(
            pose_text_by_id[entity_ids["pole"]])
    except KeyError:
        return None

    cart_pose = model_pos[0] + _quat_rotate(model_quat, chassis_local_pos)[0]
    pole_pose = _pitch_from_quat(_quat_mult(model_quat, pole_local_quat))
    return cart_pose, pole_pose
```

- [ ] **Step 4: Verify against a live server**

Terminal A (same as Task 2):

```bash
cd /home/keqing-li/Documents/ros2_gzsim_rl_1
PYTHONPATH=/usr/lib/python3/dist-packages gz sim -s -r cart_pole.sdf
```

Terminal B:

```bash
PYTHONPATH=/usr/lib/python3/dist-packages python3 -c "
import cart_pole_env as c
from gz.transport13 import Node
n = Node()
ids = c._resolve_target_entities(n)
print('ids', ids)
sim_time, names, poses = c._query_world_state(n)
print('cart_pose, pole_pose =', c._world_frame_pose(names, poses, ids))
"
```

Expected: `ids` prints a dict with keys `vehicle_green`, `chassis`, `pole` mapped to integers; the second line prints a tuple of two small floats close to `(-0.15, 0.0)` (chassis starts ~0.15m behind the model origin per `cart_pole.sdf:143`, pole starts upright at ~0 pitch).

- [ ] **Step 5: Commit**

```bash
git add cart_pole_env.py
git commit -m "Add entity resolution and world-frame pose composition for ECS state"
```

---

### Task 4: Wire synchronous polling into `run_inference`, delete dead subscription code

**Files:**
- Modify: `cart_pole_env.py:169-184` (delete `_state_lock`/`_latest_state`/`_prev_pose_state`), `cart_pole_env.py:217-267` (delete `pose_cb`), `cart_pole_env.py:279-304` (`_reset_world`), `cart_pole_env.py:306-380` (`run_inference`), `cart_pole_env.py:17` (`import threading`), `cart_pole_env.py:25` (`Pose_V` import).

**Interfaces:**
- Consumes: `_query_world_state`, `_resolve_target_entities`, `_world_frame_pose` (Tasks 2-3).

- [ ] **Step 1: Delete the now-unused `threading` import**

Remove line `cart_pole_env.py:17`:

```python
import threading
```

- [ ] **Step 2: Delete the `Pose_V` import**

Remove line `cart_pole_env.py:25`:

```python
from gz.msgs10.pose_v_pb2 import Pose_V
```

- [ ] **Step 3: Delete the module-level state dicts and lock**

Remove this whole block (`cart_pole_env.py:169-184`):

```python
# State shared between the subscriber callback and the main loop
_state_lock = threading.Lock()
_latest_state = {
    "cart_pose": 0.0,
    "cart_vel": 0.0,
    "pole_pose": 0.0,
    "pole_angular_vel": 0.0,
    "ready": False,
}
# Previous reading per entity, used to estimate velocity by finite difference
# since Pose_V only carries positions.
_prev_pose_state = {
    "cart_pose": None,
    "pole_pose": None,
    "time": None,
}
```

Nothing replaces this - the equivalent tracking becomes local variables inside `run_inference` in Step 6, since there's no longer a background callback thread that needs shared, lock-protected state.

- [ ] **Step 4: Delete `pose_cb`**

Remove the entire function (originally `cart_pole_env.py:217-267`, right after `_pitch_from_quat` and before `_kill_stale_gz_processes`):

```python
def pose_cb(msg):
    ...
```

(the whole function body from `def pose_cb(msg):` through the last `_prev_pose_state["time"] = now` line).

- [ ] **Step 5: Simplify `_reset_world`**

The old version cleared `_latest_state`/`_prev_pose_state` because a background subscriber thread could otherwise race the reset with a stale read. With no subscriber thread, `run_inference`'s own loop fully controls when it re-reads state after a reset, so `_reset_world` goes back to only doing the actual reset call:

Replace (`cart_pole_env.py:279-303`):

```python
def _reset_world(node):
    """Reset the running world via its control service.

    Without this, a fallen/out-of-bounds cart-pole keeps getting shoved by
    the policy's forces indefinitely: positions grow unbounded until they
    exceed what the physics engine's collision AABB math can represent,
    crashing the server (ODE assertion "aabbBound ... dMaxIntExact").
    """
    request = WorldControl()
    request.reset.all = True
    node.request(
        "/world/cart_pole/control", request, WorldControl, Boolean, 5000)
    with _state_lock:
        # Clear stale (out-of-bounds) readings too, not just the velocity
        # trackers - otherwise the next loop iteration reads last episode's
        # values before pose_cb delivers a post-reset message, and the
        # bounds check fires again immediately.
        _latest_state["cart_pose"] = 0.0
        _latest_state["cart_vel"] = 0.0
        _latest_state["pole_pose"] = 0.0
        _latest_state["pole_angular_vel"] = 0.0
        _latest_state["ready"] = False
        _prev_pose_state["cart_pose"] = None
        _prev_pose_state["pole_pose"] = None
        _prev_pose_state["time"] = None
```

with:

```python
def _reset_world(node):
    """Reset the running world via its control service.

    Without this, a fallen/out-of-bounds cart-pole keeps getting shoved by
    the policy's forces indefinitely: positions grow unbounded until they
    exceed what the physics engine's collision AABB math can represent,
    crashing the server (ODE assertion "aabbBound ... dMaxIntExact").
    """
    request = WorldControl()
    request.reset.all = True
    node.request(
        "/world/cart_pole/control", request, WorldControl, Boolean, 5000)
```

- [ ] **Step 6: Rewrite `run_inference`'s polling loop**

Replace the whole function (`cart_pole_env.py:306-380`):

```python
def run_inference(model):
    """
    Launch a Gazebo server + GUI and drive the trained model over Gazebo
    transport until Ctrl+C.
    """
    _kill_stale_gz_processes()

    sdf_path = os.path.join(file_path, "cart_pole.sdf")
    gz_server = None
    gz_gui = None
    try:
        print("Launching Gazebo server...")
        gz_server = subprocess.Popen(["gz", "sim", "-s", "-r", sdf_path])
        time.sleep(3)

        print("Launching Gazebo GUI...")
        gz_gui = subprocess.Popen(["gz", "sim", "-g"])
        time.sleep(5)  # Wait for GUI to connect

        node = Node()

        # Advertise on the wrench topic
        wrench_pub = node.advertise("/world/cart_pole/wrench", EntityWrench)
        time.sleep(1)

        entity_ids = _resolve_target_entities(node)
        _query_world_state(node)  # warm-up call; first request has ~200ms
                                   # one-time connection setup cost that
                                   # would otherwise skew the first loop
                                   # iteration's timing measurement below.

        print("Running inference with GUI... Press Ctrl+C to stop.")
        obs = np.zeros(4, dtype=np.float32)
        prev_cart_pose = None
        prev_pole_pose = None
        prev_sim_time = None
        target_period = 0.005  # match training's 5ms (5 x 1ms) action cadence
        for _ in range(50000):
            loop_start = time.monotonic()
            action, _s = model.predict(obs, deterministic=True)

            # Apply force to the chassis link directly, matching training.
            # Entity name must be unscoped ("chassis", not
            # "vehicle_green::chassis") - this world's transport topics only
            # match against links' bare names, confirmed via pose_cb.
            msg = EntityWrench()
            msg.entity.name = "chassis"
            msg.entity.type = 3  # LINK type
            force_x = 2000.0 if action == 1 else -2000.0
            msg.wrench.force.x = force_x
            msg.wrench.force.y = 0.0
            msg.wrench.force.z = 0.0
            wrench_pub.publish(msg)

            sim_time, names_by_id, pose_text_by_id = _query_world_state(node)
            frame = _world_frame_pose(names_by_id, pose_text_by_id, entity_ids)

            if frame is not None and sim_time is not None:
                cart_pose, pole_pose = frame
                dt = (sim_time - prev_sim_time
                      if prev_sim_time is not None else None)
                if dt is not None and dt <= 0:
                    # Sim-time reset (world reset) or a duplicate reading.
                    dt = None

                cart_vel = ((cart_pose - prev_cart_pose) / dt
                            if dt and prev_cart_pose is not None
                            else obs[1])
                pole_angular_vel = ((pole_pose - prev_pole_pose) / dt
                                     if dt and prev_pole_pose is not None
                                     else obs[3])

                obs = np.array(
                    [cart_pose, cart_vel, pole_pose, pole_angular_vel],
                    dtype=np.float32)
                prev_cart_pose, prev_pole_pose, prev_sim_time = (
                    cart_pose, pole_pose, sim_time)

            # Same bounds training uses to end an episode. Inference has no
            # episode boundary of its own, so without this the policy keeps
            # applying force to an already-fallen/out-of-bounds cart forever.
            cart_pose, pole_pose = obs[0], obs[2]
            if pole_pose > 0.48 or pole_pose < -0.48 or cart_pose > 4.8 or cart_pose < -4.8:
                print("Cart-pole out of bounds, resetting world...")
                _reset_world(node)
                time.sleep(0.5)  # let the reset propagate before next query
                obs = np.zeros(4, dtype=np.float32)
                prev_cart_pose = None
                prev_pole_pose = None
                prev_sim_time = None

            elapsed = time.monotonic() - loop_start
            remaining = target_period - elapsed
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
```

Note the `obs[1]`/`obs[3]` fallback when `dt` isn't available (first iteration, or right after a reset before two readings have landed) — this keeps the previous velocity estimate instead of forcing a spurious `0.0`, matching how the original `pose_cb` behaved (it simply left `_latest_state["cart_vel"]` unchanged until a valid `dt` existed).

- [ ] **Step 7: Compile check**

```bash
cd /home/keqing-li/Documents/ros2_gzsim_rl_1
python3 -m py_compile cart_pole_env.py && echo OK
```

Expected: `OK`, no syntax errors.

- [ ] **Step 8: Confirm no leftover references to deleted names**

```bash
grep -n "pose_cb\|_state_lock\|_latest_state\|_prev_pose_state\|Pose_V\|threading" cart_pole_env.py
```

Expected: no output (empty) — every deleted name is fully gone, not just unused.

- [ ] **Step 9: Commit**

```bash
git add cart_pole_env.py
git commit -m "Drive inference observations from synchronous ECS state queries

Replaces the dynamic_pose/info subscription (fixed ~17ms publish rate,
independent of when an observation is actually needed) with on-demand
/world/cart_pole/state requests, so each action reads a state no older
than this loop iteration - eliminating the 3+ actions per stale reading
gap the async subscription had."
```

---

### Task 5: End-to-end validation

**Files:** none modified — this is a verification-only task.

**Interfaces:**
- Consumes: the fully wired `run_inference` from Task 4.

- [ ] **Step 1: Confirm no stray gz sim processes before starting**

```bash
pgrep -af "g[z] sim" || echo clean
```

Expected: `clean`. If not, kill the listed PIDs individually with `kill <pid>` (do not use `pkill -f "gz sim"` — see Global Constraints footgun note).

- [ ] **Step 2: Run a timed 90-second inference session**

```bash
cd /home/keqing-li/Documents/ros2_gzsim_rl_1
PYTHONPATH=/usr/lib/python3/dist-packages timeout 90 uv run python -u cart_pole_env.py --infer-only > /tmp/plan_validate.log 2>&1
echo "exit: $?"
```

Expected: `exit: 124` (timeout reached, i.e. it ran the full 90s without crashing — matches the pattern used throughout this session's manual testing).

- [ ] **Step 3: Check for crashes**

```bash
grep -i -E "error|assert|abort|traceback" /tmp/plan_validate.log
```

Expected: no output. Any ODE assertion or Python traceback here means Task 4 has a bug — return to Task 4, do not proceed.

- [ ] **Step 4: Compare reset frequency against the recorded baselines**

```bash
grep -c "resetting world" /tmp/plan_validate.log
```

This session's two prior baselines (both using the fixed-5ms-sleep + `dynamic_pose/info` subscription approach) were **102-104 resets in 90s** (~470ms average episode). Record the new count. A meaningfully lower count (fewer, longer episodes) confirms fresher/more-frequent observations are helping the policy hold balance longer. If the count is roughly the same or higher, that's a real (not cosmetic) result — it means the ~17ms-vs-5ms observation staleness wasn't the dominant remaining factor, and the next investigation target should shift elsewhere (e.g. re-examine training itself, or the reward/termination shaping) rather than assuming this task failed silently.

- [ ] **Step 5: Sanity-check the achieved loop cadence**

Add a temporary print inside the loop in `run_inference` (`elapsed` variable from Task 4 Step 6) to confirm the 5ms target is actually achievable:

```bash
cd /home/keqing-li/Documents/ros2_gzsim_rl_1
PYTHONPATH=/usr/lib/python3/dist-packages python3 -c "
import cart_pole_env as c
from gz.transport13 import Node
import time
n = Node()
ids = c._resolve_target_entities(n)
c._query_world_state(n)  # warm-up
times = []
for _ in range(200):
    t0 = time.monotonic()
    c._query_world_state(n)
    times.append(time.monotonic() - t0)
times.sort()
print('median ms:', times[100] * 1000)
print('p95 ms:', times[190] * 1000)
"
```

(Run against a live server started the same way as Task 2/3's verification.) Expected: median well under 5ms (measured ~3-4ms live during this plan's research). If p95 exceeds 5ms, the loop's `remaining > 0` guard in Task 4 already handles it gracefully (no sleep, loop just runs at whatever rate the query allows) — no code change needed, just note the actual achieved rate instead of assuming 5ms.

- [ ] **Step 6: Clean up**

```bash
pgrep -af "g[z] sim" || echo clean
```

Kill any stragglers by PID (not `pkill -f "gz sim"`) before finishing.

---

## Self-Review Notes

- **Spec coverage:** Task 1 (component ID resolution) → Task 2 (raw state decode) → Task 3 (entity resolution + world-frame composition) → Task 4 (wiring + dead-code removal) → Task 5 (validation against real baselines) covers the full path from "decode raw ECS data" to "prove it actually helps."
- **No placeholders:** every step has runnable code and a concrete expected output captured from this session's live research (hash values, entity IDs, pose text format, RTT numbers) rather than assumed/guessed values.
- **Type/name consistency:** `_query_world_state` → `(sim_time, names_by_id, pose_text_by_id)` is used identically in Tasks 2, 3, and 4. `_resolve_target_entities` → `dict[str, int]` keyed by the same three literal names (`vehicle_green`, `chassis`, `pole`) used in `_world_frame_pose`'s lookups. `_world_frame_pose` → `(cart_pose, pole_pose) | None` matches how Task 4's loop unpacks it (`frame is not None` check before unpacking).
