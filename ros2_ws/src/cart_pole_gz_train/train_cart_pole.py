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
