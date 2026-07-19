import os
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)

import gymnasium as gym
import numpy as np

from gz.common5 import set_verbosity
from gz.sim8 import TestFixture, World, world_entity, Model, Link
from gz.math7 import Vector3d

from stable_baselines3 import PPO
import time
import subprocess
import threading

file_path = os.path.dirname(os.path.realpath(__file__))

def run_gui():
    """
    This function looks for your gz sim installation and looks for
    an instance of the gui client
    """
    subprocess.Popen(["gz", "sim", "-g"])

class GzRewardScorer:
    """
    This Gazebo System is used to introspect and score the world.
    """
    def __init__(self):
        """
        We initialize a TestFixture: This is a simple fixture that is used
        to load our gazebo world. We also inject the code to be executed
        on each run.
        """
        self.fixture = TestFixture(os.path.join(file_path, 'cart_pole.sdf'))
        self.fixture.on_pre_update(self.on_pre_update)
        self.fixture.on_post_update(self.on_post_update)
        self.command = None # This variable is used as a bridge between Gymnasium and gazebo
        self.fixture.finalize()
        self.server = self.fixture.server()
        self.terminated = False
        self._initialized = False
        self.state = np.zeros(4, dtype=np.float32)
        self.reward = 0.0

    def _ensure_initialized(self, ecm):
        """Look up entities if not yet initialized (or after a reset)."""
        if not self._initialized:
            world = World(world_entity(ecm))
            self.model = Model(world.model_by_name(ecm, "vehicle_green"))
            self.pole_entity = self.model.link_by_name(ecm, "pole")
            self.chassis_entity = self.model.link_by_name(ecm, "chassis")
            self.pole = Link(self.pole_entity)
            self.chassis = Link(self.chassis_entity)
            self._initialized = True

    def on_pre_update(self, info, ecm):
        """
        on_pre_update is used to command the model vehicle.
        """
        if info.paused:
            return
        self._ensure_initialized(ecm)
        self.pole.enable_velocity_checks(ecm)
        self.chassis.enable_velocity_checks(ecm)
        if self.command == 1:
            self.chassis.add_world_force(ecm, Vector3d(2000, 0, 0))
        elif self.command == 0:
            self.chassis.add_world_force(ecm, Vector3d(-2000, 0, 0))

    def on_post_update(self, info, ecm):
        """
        on_post_update is used to read the current state of the world. We write the
        state to a local field.
        """
        if info.paused:
            return
        self._ensure_initialized(ecm)
        pole_pose = self.pole.world_pose(ecm).rot().euler().y()
        ang_vel = self.pole.world_angular_velocity(ecm)
        pole_angular_vel = ang_vel.y() if ang_vel is not None else 0.0
        cart_pose = self.chassis.world_pose(ecm).pos().x()
        lin_vel = self.chassis.world_linear_velocity(ecm)
        cart_vel = lin_vel.x() if lin_vel is not None else 0.0
        # Write the state to the environment
        self.state = np.array([cart_pose, cart_vel, pole_pose, pole_angular_vel], dtype=np.float32)
        if not self.terminated:
            self.terminated = pole_pose > 0.48 or pole_pose < -0.48 or cart_pose > 4.8 or cart_pose < -4.8

        if self.terminated:
            self.reward = 0.0
        else:
            self.reward = 1.0

    def step(self, action, paused=False):
        """
        Execute the server.

        There is a bit of nuance in this instance,
        our environment has control over every 5 simulation steps.
        We block the server till those 5 steps are completed.
        """
        self.command = action
        self.server.run(True, 5, paused)
        obs = self.state
        reward = self.reward
        return obs, reward, self.terminated, False, {}

    def reset(self):
        """
        This function simply resets the server
        """
        self.server.reset_all()
        self.command = None
        self.terminated = False
        self._initialized = False
        obs, reward_, term_, tunc_, other_= self.step(None, paused=False)
        return obs, {}



class CustomCartPole(gym.Env):
    """
    Wrapper around GzRewardScorer that adapts the reward scorer to work with
    gymnasium.
    """
    def __init__(self, env_config):
        self.env = GzRewardScorer()
        self.action_space = gym.spaces.Discrete(2)
        self.observation_space = gym.spaces.Box(
            np.array([-10, float("-inf"), -0.418, -3.4028235e+38]),
            np.array([10, float("inf"), 0.418, 3.4028235e+38]),
            (4,), np.float32)

    def reset(self, seed=123):
        return self.env.reset()

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return  obs, reward, done, truncated, info

env = CustomCartPole({})
model = PPO("MlpPolicy", env, verbose=1, device="cpu")
model.learn(total_timesteps=25_000)
model.save(os.path.join(file_path, "cart_pole_ppo"))
print("Training complete. Saved model to cart_pole_ppo.zip")

# --- Inference with GUI via Gazebo transport ---
from gz.transport13 import Node
from gz.msgs10.entity_wrench_pb2 import EntityWrench
from gz.msgs10.wrench_pb2 import Wrench
from gz.msgs10.vector3d_pb2 import Vector3d as Vector3dMsg
from gz.msgs10.entity_pb2 import Entity
from gz.msgs10.pose_v_pb2 import Pose_V

# State shared between the subscriber callback and the main loop
_state_lock = threading.Lock()
_latest_state = {
    "cart_pose": 0.0,
    "cart_vel": 0.0,
    "pole_pose": 0.0,
    "pole_angular_vel": 0.0,
    "ready": False,
}

def pose_cb(msg):
    """Subscribe to dynamic pose updates to read cart and pole state."""
    with _state_lock:
        for pose in msg.pose:
            if pose.name == "vehicle_green::chassis":
                _latest_state["cart_pose"] = pose.position.x
            elif pose.name == "vehicle_green::pole":
                # Extract pitch (Y rotation) from quaternion
                q = pose.orientation
                # pitch from quaternion: atan2(2*(qw*qy - qz*qx), 1 - 2*(qx^2 + qy^2))
                import math
                sinp = 2.0 * (q.w * q.y - q.z * q.x)
                sinp = max(-1.0, min(1.0, sinp))
                _latest_state["pole_pose"] = math.asin(sinp)
                _latest_state["ready"] = True

# Launch gz sim with GUI
sdf_path = os.path.join(file_path, "cart_pole.sdf")
print("Launching Gazebo server...")
gz_server = subprocess.Popen(["gz", "sim", "-s", "-r", sdf_path])
time.sleep(3)

print("Launching Gazebo GUI...")
gz_gui = subprocess.Popen(["gz", "sim", "-g"])
time.sleep(5)  # Wait for GUI to connect

node = Node()

# Subscribe to dynamic pose info
node.subscribe(Pose_V, "/world/cart_pole/dynamic_pose/info", pose_cb)

# Advertise on the wrench topic
wrench_pub = node.advertise("/world/cart_pole/wrench", EntityWrench)
time.sleep(1)

print("Running inference with GUI... Press Ctrl+C to stop.")
try:
    obs = np.zeros(4, dtype=np.float32)
    for i in range(50000):
        action, _s = model.predict(obs, deterministic=True)

        # Apply force based on action
        msg = EntityWrench()
        msg.entity.name = "vehicle_green"
        msg.entity.type = 2  # MODEL type
        force_x = 2000.0 if action == 1 else -2000.0
        msg.wrench.force.x = force_x
        msg.wrench.force.y = 0.0
        msg.wrench.force.z = 0.0
        wrench_pub.publish(msg)

        time.sleep(0.005)  # ~5ms per step to match sim time

        with _state_lock:
            obs = np.array([
                _latest_state["cart_pose"],
                _latest_state["cart_vel"],
                _latest_state["pole_pose"],
                _latest_state["pole_angular_vel"],
            ], dtype=np.float32)

except KeyboardInterrupt:
    print("\nStopping...")
finally:
    gz_gui.terminate()
    gz_server.terminate()
    gz_gui.wait()
    gz_server.wait()