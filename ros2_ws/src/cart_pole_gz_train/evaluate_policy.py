"""Scratch measurement script (not pytest, matching repo convention):
runs either a random policy or a trained PPO model against
CustomCartPoleGzTrain for N episodes and reports episode-length
(steps-survived) statistics, plus a termination-cause breakdown
(pole-angle limit vs. cart-position limit) - the same methodology used
to originally diagnose the "trained == random" bug in Task 5.
"""
import argparse
import os
import sys
import ctypes
ctypes.CDLL("/usr/lib/x86_64-linux-gnu/libgz-sim8.so", ctypes.RTLD_GLOBAL)

FILE_DIR = os.path.dirname(os.path.realpath(__file__))
sys.path.insert(0, FILE_DIR)
os.chdir(FILE_DIR)

import numpy as np

from train_cart_pole import CustomCartPoleGzTrain
from gz_scorer import CART_POSITION_LIMIT, POLE_PITCH_LIMIT


def _termination_cause(obs):
    cart_x, _cart_v, pole_p, _pole_v = obs
    if abs(pole_p) > POLE_PITCH_LIMIT:
        return "pole_angle"
    if abs(cart_x) > CART_POSITION_LIMIT:
        return "cart_position"
    return "unknown"


def run_random(n_episodes):
    env = CustomCartPoleGzTrain()
    lengths = []
    causes = []
    for ep in range(n_episodes):
        obs, _info = env.reset()
        steps = 0
        terminated = False
        while not terminated:
            action = env.action_space.sample()
            obs, _reward, terminated, _truncated, _info = env.step(action)
            steps += 1
        lengths.append(steps)
        causes.append(_termination_cause(obs))
        print(f"  episode {ep + 1}: {steps} steps (terminated on {causes[-1]})")
    env.close()
    return lengths, causes


def run_trained(n_episodes, model_path, vecnorm_path):
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

    venv = DummyVecEnv([lambda: CustomCartPoleGzTrain()])
    if vecnorm_path and os.path.exists(vecnorm_path):
        venv = VecNormalize.load(vecnorm_path, venv)
        venv.training = False
        venv.norm_reward = False
        print(f"  loaded VecNormalize stats from {vecnorm_path}")
    else:
        print("  WARNING: no VecNormalize stats found, using raw observations")
    model = PPO.load(model_path)

    lengths = []
    causes = []
    for ep in range(n_episodes):
        obs = venv.reset()
        steps = 0
        done = False
        while not done:
            action, _state = model.predict(obs, deterministic=True)
            obs, _reward, done_arr, info = venv.step(action)
            done = bool(done_arr[0])
            steps += 1
        # DummyVecEnv stores the true pre-reset obs in info on episode end
        # (VecNormalize doesn't touch it, so it's raw/un-normalized) -
        # fall back to the (already-reset) obs if unavailable.
        final_obs = info[0].get("terminal_observation", obs[0])
        causes.append(_termination_cause(np.asarray(final_obs)))
        lengths.append(steps)
        print(f"  episode {ep + 1}: {steps} steps (terminated on {causes[-1]})")
    venv.close()
    return lengths, causes


def summarize(label, lengths, causes):
    arr = np.array(lengths)
    print(
        f"{label}: n={len(arr)} mean={arr.mean():.1f} min={arr.min()} "
        f"max={arr.max()} std={arr.std():.1f}"
    )
    from collections import Counter
    print(f"  termination causes: {dict(Counter(causes))}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["random", "trained"])
    parser.add_argument("--episodes", type=int, default=15)
    parser.add_argument(
        "--model", default=os.path.join(FILE_DIR, "cart_pole_gz_train_ppo")
    )
    parser.add_argument(
        "--vecnorm", default=os.path.join(FILE_DIR, "vecnormalize.pkl")
    )
    args = parser.parse_args()

    if args.mode == "random":
        lengths, causes = run_random(args.episodes)
        summarize("random policy", lengths, causes)
    else:
        lengths, causes = run_trained(args.episodes, args.model, args.vecnorm)
        summarize("trained policy", lengths, causes)
