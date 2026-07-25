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
