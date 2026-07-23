import gymnasium as gym
import torch
import yaml

from dqn import DQN

import argparse
import itertools

import flappy_bird_gymnasium
import os

# Directory for saving run info
RUNS_DIR = "runs"
os.makedirs(RUNS_DIR, exist_ok=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # force cpu, sometimes GPU not always faster than CPU due to overhead of moving data to GPU

# Deep Q-Learning Agent.
# Holds the hyperparameter config and the shared building blocks (env, network,
# greedy action selection) used by both inference (here) and training (train.py).
class Agent():

    def __init__(self, hyperparameter_set):
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
            # print(hyperparameters)

        self.hyperparameter_set = hyperparameter_set

        # Hyperparameters (adjustable)
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']        # learning rate (alpha)
        self.discount_factor_g  = hyperparameters['discount_factor_g']      # discount rate (gamma)
        self.network_sync_rate  = hyperparameters['network_sync_rate']      # number of steps the agent takes before syncing the policy and target network
        self.replay_memory_size = hyperparameters['replay_memory_size']     # size of replay memory
        self.mini_batch_size    = hyperparameters['mini_batch_size']        # size of the training data set sampled from the replay memory
        self.epsilon_init       = hyperparameters['epsilon_init']           # 1 = 100% random actions
        self.epsilon_decay      = hyperparameters['epsilon_decay']          # epsilon decay rate
        self.epsilon_min        = hyperparameters['epsilon_min']            # minimum epsilon value
        self.stop_on_reward     = hyperparameters['stop_on_reward']         # stop after reaching this number of rewards
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.env_make_params    = hyperparameters.get('env_make_params',{}) # Get optional environment-specific parameters, default to empty dict
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']      # double dqn on/off flag
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']     # dueling dqn on/off flag

        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def make_env(self, render=False):
        # Create instance of the environment.
        # Use "**self.env_make_params" to pass in environment-specific parameters from hyperparameters.yml.
        return gym.make(self.env_id, render_mode='human' if render else None, **self.env_make_params)

    def build_dqn(self, num_states, num_actions):
        # Create a DQN with the right input/output dims on the target device.
        # Number of nodes in the hidden layer can be adjusted.
        return DQN(num_states, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)

    def select_action(self, state, policy_dqn):
        # Select best action (greedy) from the policy network.
        with torch.no_grad():
            # state.unsqueeze(dim=0): Pytorch expects a batch layer, so add batch dimension i.e. tensor([1, 2, 3]) unsqueezes to tensor([[1, 2, 3]])
            # policy_dqn returns tensor([[1], [2], [3]]), so squeeze it to tensor([1, 2, 3]).
            # argmax finds the index of the largest element.
            return policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()

    def run(self, render=True):
        # Inference only: load a trained policy from MODEL_FILE and roll it out.
        env = self.make_env(render)

        # Number of possible actions
        num_actions = env.action_space.n

        # Get observation space size
        num_states = env.observation_space.shape[0] # Expecting type: Box(low, high, (shape0,), float64)

        policy_dqn = self.build_dqn(num_states, num_actions)

        # Load learned policy
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))

        # switch model to evaluation mode
        policy_dqn.eval()

        # Run episodes INDEFINITELY, manually stop the run when you are satisfied.
        for episode in itertools.count():

            state, _ = env.reset()  # Initialize environment. Reset returns (state,info).
            state = torch.tensor(state, dtype=torch.float, device=device) # Convert state to tensor directly on device

            terminated = False      # True when agent reaches goal or fails
            episode_reward = 0.0    # Used to accumulate rewards per episode

            # Perform actions until episode terminates or reaches max rewards
            while(not terminated and episode_reward < self.stop_on_reward):

                # Select best action
                action = self.select_action(state, policy_dqn)

                # Execute action. Truncated and info is not used.
                new_state, reward, terminated, truncated, info = env.step(action.item())

                # Accumulate rewards
                episode_reward += reward

                # Convert new state to tensor on device and move to the next state
                new_state = torch.tensor(new_state, dtype=torch.float, device=device)
                state = new_state


if __name__ == '__main__':
    # Parse command line inputs
    parser = argparse.ArgumentParser(description='Run a trained model (inference).')
    parser.add_argument('hyperparameters', help='')
    args = parser.parse_args()

    dql = Agent(hyperparameter_set=args.hyperparameters)
    dql.run(render=True)
