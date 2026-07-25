#!/usr/bin/env python3

import threading
import time

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim

import rclpy
from rclpy.node import Node

from std_msgs.msg import Float64
from sensor_msgs.msg import JointState
from ros_gz_interfaces.srv import ControlWorld


class QNet(nn.Module):
    def __init__(self, num_states, dim_mid, num_actions):
        super().__init__()

        self.fc = nn.Sequential(
            nn.Linear(num_states, dim_mid),
            nn.ReLU(),
            nn.Linear(dim_mid, dim_mid),
            nn.ReLU(),
            nn.Linear(dim_mid, num_actions)
        )

    def forward(self, x):
        x = self.fc(x)
        return x


class Brain:
    def __init__(self, num_states, num_actions, gamma, r, lr):
        self.num_states = num_states
        self.num_actions = num_actions
        self.eps = 1.0
        self.gamma = gamma
        self.r = r

        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("self.device = ", self.device)
        self.q_net = QNet(num_states, 64, num_actions)
        self.q_net.to(self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

    def updateQnet(self, obs_numpy, action, reward, next_obs_numpy):
        obs_tensor = torch.from_numpy(obs_numpy).float()
        obs_tensor.unsqueeze_(0)
        obs_tensor = obs_tensor.to(self.device)

        next_obs_tensor = torch.from_numpy(next_obs_numpy).float()
        next_obs_tensor.unsqueeze_(0)
        next_obs_tensor = next_obs_tensor.to(self.device)

        self.optimizer.zero_grad()

        self.q_net.train()
        q = self.q_net(obs_tensor)

        with torch.no_grad():
            self.q_net.eval()
            label = self.q_net(obs_tensor)
            next_q = self.q_net(next_obs_tensor)

            label[:, action] = reward + self.gamma * np.max(next_q.cpu().detach().numpy(), axis=1)[0]

        loss = self.criterion(q, label)
        loss.backward()
        self.optimizer.step()

    def getAction(self, obs_numpy, is_training):
        if is_training and np.random.rand() < self.eps:
            action = np.random.randint(self.num_actions)
        else:
            obs_tensor = torch.from_numpy(obs_numpy).float()
            obs_tensor.unsqueeze_(0)
            obs_tensor = obs_tensor.to(self.device)
            with torch.no_grad():
                self.q_net.eval()
                q = self.q_net(obs_tensor)
                action = np.argmax(q.cpu().detach().numpy(), axis=1)[0]

        if is_training and self.eps > 0.1:
            self.eps *= self.r
        return action


class Agent:
    def __init__(self, num_states, num_actions, gamma, r, lr):
        self.brain = Brain(num_states, num_actions, gamma, r, lr)

    def updateQnet(self, obs, action, reward, next_obs):
        self.brain.updateQnet(obs, action, reward, next_obs)

    def getAction(self, obs, is_training):
        action = self.brain.getAction(obs, is_training)
        return action


class DQNSimulationNode(Node):
    def __init__(self):
        super().__init__('DQN_simulation')

        self.cart_pose_x = 0.0
        self.cart_vel_x = 0.0
        self.y_angular = 0.0

        self.pub_cart = self.create_publisher(Float64, '/cart_controller/command', 10)
        self.create_subscription(JointState, '/joint_states', self._joint_state_callback, 10)
        self.reset_client = self.create_client(ControlWorld, '/world/robomaster_rale/control')

        self.number_of_steps = []

    def _joint_state_callback(self, msg):
        if 'cart_joint' in msg.name:
            idx = msg.name.index('cart_joint')
            self.cart_pose_x = msg.position[idx]
            self.cart_vel_x = msg.velocity[idx]
        if 'pole_joint' in msg.name:
            idx = msg.name.index('pole_joint')
            self.y_angular = msg.velocity[idx]

    def reset_simulation(self):
        if not self.reset_client.wait_for_service(timeout_sec=5.0):
            raise RuntimeError('/world/robomaster_rale/control service unavailable')
        request = ControlWorld.Request()
        # NOTE: reset.all=True tears down and recreates every entity in the
        # world (confirmed via live testing: gz-sim's JointStatePublisher
        # stops advertising its gz-transport topic permanently after an
        # `all` reset, silently freezing /joint_states forever after episode
        # 0). reset.model_only=True resets model poses/velocities without
        # that teardown, so JointStatePublisher keeps publishing across
        # every episode.
        request.world_control.reset.model_only = True
        future = self.reset_client.call_async(request)
        deadline = time.time() + 5.0
        while not future.done():
            if time.time() > deadline:
                raise RuntimeError(
                    'reset_simulation: /world/robomaster_rale/control did not '
                    'respond within 5.0s')
            time.sleep(0.001)

    def simulate(self, episode, is_training, agent):
        reward = 0
        yaw_angle = 0
        time_interval = 0.02

        self.reset_simulation()

        obs = np.array([0, 0, 0, 0], dtype='float16')
        next_obs = np.array([0, 0, 0, 0], dtype='float16')
        max_step = 200
        is_done = False
        episode_reward = 0

        for step in range(max_step):
            time1 = time.time()
            yaw_angle += self.y_angular * time_interval

            next_obs[0] = self.cart_pose_x
            next_obs[1] = self.cart_vel_x
            next_obs[2] = yaw_angle
            next_obs[3] = self.y_angular

            if step == 0:
                next_obs[0] = 0
                next_obs[1] = 0
                next_obs[2] = 0
                next_obs[3] = 0

            action = agent.getAction(obs, is_training)

            if abs(yaw_angle) > 0.6 or step == max_step - 1:
                is_done = True

            if is_done:
                if step < max_step - 1:
                    reward = -200
                else:
                    reward = episode_reward
            else:
                reward = 6 - abs(yaw_angle) * 10

            episode_reward += reward

            force = action * 16 / 9 - 8

            if is_training:
                agent.updateQnet(obs, action, reward, next_obs)

            obs = np.copy(next_obs)

            self.pub_cart.publish(Float64(data=float(force)))

            time2 = time.time()
            interval = time2 - time1
            if interval < time_interval:
                time.sleep(time_interval - interval)

            if is_done and is_training:
                print('Episode: {0} Finished after {1} time steps with reward {2}'.format(
                    episode, step + 1, episode_reward))
                self.plot_durations(step)
                break
            elif is_done and not is_training:
                print('Evaluation: Finished after {} time steps'.format(step + 1))
                break

    def plot_durations(self, step):
        plt.figure(2)
        plt.clf()
        self.number_of_steps.append(step)
        x = np.arange(0, len(self.number_of_steps))
        plt.title('Training')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(x, self.number_of_steps)

        plt.pause(0.001)


def main():
    rclpy.init()
    node = DQNSimulationNode()

    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    gamma = 0.7
    r = 0.99
    lr = 0.001
    num_states = 4
    num_actions = 10

    agent = Agent(num_states, num_actions, gamma, r, lr)

    num_episodes = 1000
    is_training = True
    try:
        for i_episode in range(num_episodes):
            node.simulate(i_episode, is_training, agent)

        x = np.arange(0, len(node.number_of_steps))
        plt.plot(x, node.number_of_steps)
        plt.show()
    finally:
        rclpy.shutdown()


if __name__ == '__main__':
    main()
