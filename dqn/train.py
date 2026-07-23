import numpy as np

import matplotlib
import matplotlib.pyplot as plt

import random
import torch
from torch import nn

from experience_replay import ReplayMemory
from agent import Agent, device

from datetime import datetime, timedelta
import argparse
import itertools

# For printing date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# 'Agg': used to generate plots as images and save them to a file instead of rendering to screen
matplotlib.use('Agg')

# Trains a Deep Q-Learning Agent.
# Composes an Agent for the hyperparameter config, file paths, and the shared
# env/network/action helpers, and owns all training-only machinery.
class Trainer():

    def __init__(self, hyperparameter_set):
        # Reuse Agent for config, file paths, and env/network/action helpers.
        self.agent = Agent(hyperparameter_set)

        # Neural Network
        self.loss_fn = nn.MSELoss()          # NN Loss function. MSE=Mean Squared Error can be swapped to something else.
        self.optimizer = None                # NN Optimizer. Initialize later.

    def train(self):
        agent = self.agent

        start_time = datetime.now()
        last_graph_update_time = start_time

        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(agent.LOG_FILE, 'w') as file:
            file.write(log_message + '\n')

        # Create instance of the environment.
        env = agent.make_env(render=False)

        # Number of possible actions
        num_actions = env.action_space.n

        # Get observation space size
        num_states = env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)

        # List to keep track of rewards collected per episode.
        rewards_per_episode = []

        # Create policy network.
        policy_dqn = agent.build_dqn(num_states, num_actions)

        # Initialize epsilon
        epsilon = agent.epsilon_init

        # Initialize replay memory
        memory = ReplayMemory(agent.replay_memory_size)

        # Create the target network and make it identical to the policy network
        target_dqn = agent.build_dqn(num_states, num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=agent.learning_rate_a)

        # List to keep track of epsilon decay
        epsilon_history = []

        # Track number of steps taken. Used for syncing policy => target network.
        step_count = 0

        # Track best reward
        best_reward = -9999999

        # Train INDEFINITELY, manually stop the run when you are satisfied (or unsatisfied) with the results
        for episode in itertools.count():

            state, _ = env.reset()  # Initialize environment. Reset returns (state,info).
            state = torch.tensor(state, dtype=torch.float, device=device) # Convert state to tensor directly on device

            terminated = False      # True when agent reaches goal or fails
            episode_reward = 0.0    # Used to accumulate rewards per episode

            # Perform actions until episode terminates or reaches max rewards
            # (on some envs, it is possible for the agent to train to a point where it NEVER terminates, so stop on reward is necessary)
            while(not terminated and episode_reward < agent.stop_on_reward):

                # Select action based on epsilon-greedy
                if random.random() < epsilon:
                    # select random action
                    action = env.action_space.sample()
                    action = torch.tensor(action, dtype=torch.int64, device=device)
                else:
                    # select best action
                    action = agent.select_action(state, policy_dqn)

                # Execute action. Truncated and info is not used.
                new_state, reward, terminated, truncated, info = env.step(action.item())

                # Accumulate rewards
                episode_reward += reward

                # Convert new state and reward to tensors on device
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                reward = torch.tensor(reward, dtype=torch.float, device=device)

                # Save experience into memory
                memory.append((state, action, new_state, reward, terminated))

                # Increment step counter
                step_count += 1

                # Move to the next state
                state = new_state

            # Keep track of the rewards collected per episode.
            rewards_per_episode.append(episode_reward)

            # Save model when new best reward is obtained.
            if episode_reward > best_reward:
                log_message = f"{datetime.now().strftime(DATE_FORMAT)}: New best reward {episode_reward:0.1f} ({(episode_reward-best_reward)/best_reward*100:+.1f}%) at episode {episode}, saving model..."
                print(log_message)
                with open(agent.LOG_FILE, 'a') as file:
                    file.write(log_message + '\n')

                torch.save(policy_dqn.state_dict(), agent.MODEL_FILE)
                best_reward = episode_reward

            # Update graph every x seconds
            current_time = datetime.now()
            if current_time - last_graph_update_time > timedelta(seconds=10):
                self.save_graph(rewards_per_episode, epsilon_history)
                last_graph_update_time = current_time

            # If enough experience has been collected
            if len(memory) > agent.mini_batch_size:
                mini_batch = memory.sample(agent.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn)

                # Decay epsilon
                epsilon = max(epsilon * agent.epsilon_decay, agent.epsilon_min)
                epsilon_history.append(epsilon)

                # Copy policy network to target network after a certain number of steps
                if step_count > agent.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0

    def save_graph(self, rewards_per_episode, epsilon_history):
        # Save plots
        fig = plt.figure(1)

        # Plot average rewards (Y-axis) vs episodes (X-axis)
        mean_rewards = np.zeros(len(rewards_per_episode))
        for x in range(len(mean_rewards)):
            mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
        plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
        # plt.xlabel('Episodes')
        plt.ylabel('Mean Rewards')
        plt.plot(mean_rewards)

        # Plot epsilon decay (Y-axis) vs episodes (X-axis)
        plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
        # plt.xlabel('Time Steps')
        plt.ylabel('Epsilon Decay')
        plt.plot(epsilon_history)

        plt.subplots_adjust(wspace=1.0, hspace=1.0)

        # Save plots
        fig.savefig(self.agent.GRAPH_FILE)
        plt.close(fig)

    # Optimize policy network
    def optimize(self, mini_batch, policy_dqn, target_dqn):

        # Transpose the list of experiences and separate each element
        states, actions, new_states, rewards, terminations = zip(*mini_batch)

        # Stack tensors to create batch tensors
        # tensor([[1,2,3]])
        states = torch.stack(states)

        actions = torch.stack(actions)

        new_states = torch.stack(new_states)

        rewards = torch.stack(rewards)
        terminations = torch.tensor(terminations).float().to(device)

        with torch.no_grad():
            if self.agent.enable_double_dqn:
                best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)

                target_q = rewards + (1-terminations) * self.agent.discount_factor_g * \
                                target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
            else:
                # Calculate target Q values (expected returns)
                target_q = rewards + (1-terminations) * self.agent.discount_factor_g * target_dqn(new_states).max(dim=1)[0]
                '''
                    target_dqn(new_states)  ==> tensor([[1,2,3],[4,5,6]])
                        .max(dim=1)         ==> torch.return_types.max(values=tensor([3,6]), indices=tensor([3, 0, 0, 1]))
                            [0]             ==> tensor([3,6])
                '''

        # Calcuate Q values from current policy
        current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()
        '''
            policy_dqn(states)  ==> tensor([[1,2,3],[4,5,6]])
                actions.unsqueeze(dim=1)
                .gather(1, actions.unsqueeze(dim=1))  ==>
                    .squeeze()                    ==>
        '''

        # Compute loss
        loss = self.loss_fn(current_q, target_q)

        # Optimize the model (backpropagation)
        self.optimizer.zero_grad()  # Clear gradients
        loss.backward()             # Compute gradients
        self.optimizer.step()       # Update network parameters i.e. weights and biases


if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Train a model.')
    parser.add_argument('hyperparameters', help='')
    args = parser.parse_args()

    trainer = Trainer(hyperparameter_set=args.hyperparameters)
    trainer.train()
