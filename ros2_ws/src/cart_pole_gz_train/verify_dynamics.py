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
