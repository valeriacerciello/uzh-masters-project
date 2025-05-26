#%%
from enum import EnumType, Enum

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from numpy.f2py.auxfuncs import throw_error
from pettingzoo.mpe import simple_speaker_listener_v4
from scipy.ndimage import label


def make_env():
    env = simple_speaker_listener_v4.parallel_env(continuous_actions=False, render_mode="rgb_array", max_cycles=25)
    env.reset()
    return env

def make_env_human():
    env = simple_speaker_listener_v4.parallel_env(continuous_actions=False, render_mode="human", max_cycles=25)
    env.reset()
    return env

class DiscType(Enum):
    DECENTRALIZED = "decentralized"
    CENTRALIZED = "centralized"
    SINGLE = "single"

#%% md
# # Test the Environment
#%%
env = make_env()
observations, infos = env.reset()  # returns a dict
observations
# env.action_space
#%% md
# **Here we can see that the global state is a concatenation of local observations**
#%%
env.state()
#%%
print("Agents:", env.agents)  # ['speaker_0', 'listener_0']
#%% md
# # Hyperparameters
#%%
device = "cuda:0" if torch.cuda.is_available() else "cpu"

USE_KFAC = False

DISC_TYPE = DiscType.CENTRALIZED
training_epochs = 2000

centralized_learning_rate = 5e-4 # Try 3e-4
centralized_weight_decay = 0.15

decentralized_learning_rate = 1e-3
decentralized_weight_decay = 1e-3

policy_learning_rate = 1e-3
policy_learning_rate_kfac = 1e-2

value_learning_rate = 1e-3

num_expert_episodes = 200
BATCH_SIZE = 64

# Two layers network, applied for discriminator, policy and value network
all_hid_dim = 256



#%% md
# # Environment Variables
#%%
# Agent dimensions
speaker_obs_dim = 3     #[goal_id]
listener_obs_dim = 11 #[self_vel, all_landmark_rel_positions, communication]

speaker_act_dim = 3
listener_act_dim = 5 # [no_action, move_left, move_right, move_down, move_up]

all_agents = env.agents

obs_dims = {
    "speaker_0": speaker_obs_dim,
    "listener_0": listener_obs_dim
}

act_dims = {
    "speaker_0": speaker_act_dim,
    "listener_0": listener_act_dim
}

hid_dims = {
    "speaker_0": all_hid_dim,
    "listener_0": all_hid_dim
}

# Evaluated from 5000 expert episodes
expert_mean_reward = -39.53
#%% md
# # Expert Demonstrations
# Load the expert policy from paper Inverse Factorized Soft Q-Learning for Cooperative Multi-agent Imitation Learning
# 
#%%
# Load the expert policy
expert_policies = torch.jit.load("Expert_data/simple_speaker_listener.pt").to(device)
expert_policies.eval()

# Try to retrieve h_dim from the policy, fallback to 128
try:
    h_dim = expert_policies.h_dim
    print(f"Retrieved h_dim: {h_dim}")
except AttributeError:
    h_dim = 128  # From error message
    print("h_dim not accessible, using 128")

def expert_policy(obs, rnn_actor, deterministic=True):
    with torch.no_grad():
        # Convert observations to tensors
        obs_speaker = torch.FloatTensor(obs["speaker_0"]).to(device)
        obs_listener = torch.FloatTensor(obs["listener_0"]).to(device)
        # Pad speaker's observation (3) to match listener’s (11)
        obs_speaker_padded = torch.nn.functional.pad(obs_speaker, (0, 11 - 3))
        obs_batch = torch.stack([obs_speaker_padded, obs_listener], dim=0)

        # Define masks and available actions
        masks = torch.ones((2, 1), dtype=torch.bool, device=device)  # For 2 agents
        avails = torch.ones((2, 5), dtype=torch.float32, device=device)  # Assuming 5 actions per agent

        # Forward pass through the policy
        actions, _, new_rnn_actor = expert_policies._forward(
            obs_batch, rnn_actor, masks, avails, deterministic
        )
        actions_dict = {
            "speaker_0": actions[0].item(),
            "listener_0": actions[1].item()
        }
        return actions_dict, new_rnn_actor

def generate_expert_data_decentralized(num_episodes=50):
    env = make_env()  # Assuming this is defined elsewhere

    # TODO should change the data structure to actually store tuple (s,a)
    expert_data = {agent: {"states": [], "actions": []} for agent in env.agents}

    for _ in range(num_episodes):
        obs, _ = env.reset()
        # Initialize rnn_actor with shape (n_agents, 1, h_dim) on device
        rnn_actor = torch.zeros((2, 1, h_dim), device=device)

        while env.agents:
            actions, rnn_actor = expert_policy(obs, rnn_actor, deterministic=True)
            for agent in env.agents:
                expert_data[agent]["states"].append(obs[agent])
                expert_data[agent]["actions"].append(actions[agent])
                joint_state = env.state()
            obs, rewards, terminations, truncations, infos = env.step(actions)

    env.close()
    return expert_data

def convert_decentralized_data_to_centralized(decentralized_data):
    """
    Converts decentralized expert data to centralized format.

    Args:
        decentralized_data (dict): Decentralized expert data with the structure:
            {
                agent1: {"states": [state1, state2, ...], "actions": [action1, action2, ...]},
                agent2: {"states": [state1, state2, ...], "actions": [action1, action2, ...]},
                ...
            }
    Returns:
        dict: Centralized expert data with the structure:
            {
                "global_states": [state1, state2, ...],
                "joint_actions": [actions1, actions2, ...],
            }
    """
    centralized_expert_data = {
        "joint_states": [],  # Global states across all agents
        "joint_actions": []   # Joint actions per global state
    }

    # Ensure we use the same length throughout agents
    num_samples = len(decentralized_data[all_agents[0]]["states"])  # Extract the length from first agent

    for i in range(num_samples):
        # Combine states of all agents into a global state
        global_state = []
        for agent in decentralized_data:
            global_state.extend(decentralized_data[agent]["states"][i])
        centralized_expert_data["joint_states"].append(global_state)

        # Collect actions from all agents into a joint action
        joint_action = [decentralized_data[agent]["actions"][i] for agent in decentralized_data]
        centralized_expert_data["joint_actions"].append(joint_action)

    return centralized_expert_data

#%%
import pickle

with open("Expert_data/expert_data_rllib_simple_listener_speaker_200.pickle", "rb") as f:
    decentralized_expert_data = pickle.load(f)
#%%
# Generate data
# decentralized_expert_data = generate_expert_data_decentralized(num_episodes=num_expert_episodes)

if DISC_TYPE == DiscType.CENTRALIZED:
    centralized_expert_data = convert_decentralized_data_to_centralized(decentralized_expert_data)

#%%
decentralized_expert_data["speaker_0"]["states"][0]
#%%
decentralized_expert_data["listener_0"]["states"][0]
#%%
centralized_expert_data = convert_decentralized_data_to_centralized(decentralized_expert_data)
centralized_expert_data["joint_states"][0]
#%% md
# Shape Checking
#%%
if DISC_TYPE == DiscType.CENTRALIZED:
    expert_states = torch.FloatTensor(centralized_expert_data["joint_states"]).to(device)
    expert_actions = torch.LongTensor(centralized_expert_data["joint_actions"]).to(device)
elif DISC_TYPE == DiscType.DECENTRALIZED:
    agent = "speaker_0"
    expert_states = torch.FloatTensor(decentralized_expert_data[agent]["states"]).to(device)
    expert_actions = torch.LongTensor(decentralized_expert_data[agent]["actions"]).to(device)

#%%
expert_states.shape
#%%
expert_actions.shape
#%%
expert_actions.cpu().numpy()
#%% md
# ### Evaluate the Expert Policy
#%%
def evaluate_policy_for_expert(policy_func, num_episodes=50, threshold=0.1, h_dim=128, device=device):
    env = make_env()
    avg_rewards = []

    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0
        rnn_actor = torch.zeros((2, 1, h_dim), device=device)

        while env.agents:
            actions, rnn_actor = policy_func(obs, rnn_actor, deterministic=True)
            obs, rewards, done, _, _ = env.step(actions)
            total_reward += sum(rewards.values())

        avg_rewards.append(total_reward)

    env.close()
    return {
        "avg_reward": np.mean(avg_rewards),
    }

#%%
# Evaluate the expert policy
# expert_eval = evaluate_policy_for_expert(expert_policy, num_episodes=20)
# print("Expert Policy Evaluation:", expert_eval)

#%% md
# # Policy Evaluation Function
#%%
def learned_policy(policies, obs, device=device):
    """
    Retrieves actions from the learned policy networks for both agents based on observations.

    Args:
        policies (dict): Dictionary mapping agent names to their Policy network instances.
        obs (dict): Dictionary of observations for each agent.
        device (str): Device to perform computations on ("cuda:0" or "cpu").

    Returns:
        dict: Dictionary mapping agent names to selected actions.
    """
    actions = {}
    for agent in obs:
        policy = policies[agent]
        # Convert observation to tensor and move to device
        obs_tensor = torch.FloatTensor(obs[agent]).to(device)
        with torch.no_grad():
            # Get action probabilities from the policy network
            action_probs = policy(obs_tensor)
            # Select the action with the highest probability
            action = torch.argmax(action_probs).item()
        actions[agent] = action
    return actions

def evaluate_policy_for_training(policies, num_episodes=50, threshold=0.1, device=device):
    """
    Evaluates the performance of learned policies over multiple episodes.

    Args:
        policies (dict): Dictionary of Policy networks for each agent.
        num_episodes (int): Number of episodes to run for evaluation (default: 50).
        threshold (float): Distance threshold to determine success (default: 0.1).
        device (str): Device to perform computations on (default: "cuda:0" or "cpu").

    Returns:
        dict: Dictionary containing evaluation metrics:
              - "success_rate": Mean success rate across episodes.
              - "avg_reward": Mean total reward per episode.
              - "avg_steps": Mean number of steps per episode.
    """
    # Set to evaluation mode to switch off dropout
    for policy in policies.values():
        policy.eval()

    # Initialize the environment
    env = make_env()
    avg_rewards = []

    # Run evaluation over specified number of episodes
    for episode in range(num_episodes):
        obs, _ = env.reset()
        total_reward = 0

        # Run the episode
        while env.agents:
            # Get actions using the learned policy
            actions = learned_policy(policies, obs, device)
            # Step the environment
            obs, rewards, done, _, _ = env.step(actions)
            total_reward += sum(rewards.values())  # Sum rewards from both agents

        avg_rewards.append(total_reward)

    # Clean up environment resources
    env.close()

    # Compute and return average metrics
    return {
        "avg_reward": np.mean(avg_rewards),
    }
#%% md
# # MAGAIL Training
#%% md
# ## Define MAGAIL Networks
#%%
import torch
import torch.nn as nn

# Policy Networks (one per agent)
class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Softmax(dim=-1)
        )
    
    def forward(self, x):
        return self.net(x)

# Discriminator Networks (one per agent)
# Output the probability of (s,a) Coming from the EXPERT
class Discriminator(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid())
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))
    
# Value Network as baseline
class ValueNet(nn.Module):
    def __init__(self, obs_dim, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def forward(self, x):
        return self.net(x)


# Initialize Networks 

if DISC_TYPE == DiscType.CENTRALIZED:
    # For Discriminator the output is a single value
    joint_state_dim = sum(obs_dims.values())  # Global state dimension
    joint_action_dim = sum(act_dims.values())
    discriminator_centralized = Discriminator(obs_dim=joint_state_dim, act_dim=joint_action_dim, hidden_dim=hid_dims[all_agents[0]]).to(device)

elif DISC_TYPE == DiscType.DECENTRALIZED:
    discriminators = {
        a: Discriminator(obs_dim=obs_dims[a], act_dim=act_dims[a], hidden_dim=hid_dims[a]).to(device)
        for a in all_agents
    }

# Initialize value networks to reduce variance (actor critic)

policies = {
    a: Policy(obs_dim=obs_dims[a], act_dim=act_dims[a], hidden_dim=hid_dims[a]).to(device)
    for a in all_agents
}

if DISC_TYPE == DiscType.CENTRALIZED:
    joint_state_dim = sum(obs_dims.values())  # Global state dimension
    value_nets = {}
    for agent in all_agents:
        value_nets[agent] = ValueNet(joint_state_dim, hid_dims[agent]).to(device)
elif DISC_TYPE == DiscType.DECENTRALIZED:
    value_nets = {
        agent: ValueNet(obs_dims[agent], hid_dims[agent]).to(device)
        for agent in all_agents
    }
#%%
sum(act_dims.values())
#%% md
# ## Training MAGAIL Loop
#%%
from torch_kfac import KFAC

all_expert_rewards = []  # Store all expert rewards
all_episode_rewards = []  # Store all episode rewards
all_generator_rewards = []
all_advantages = []
all_values_estimates = []
all_policy_losses = []

all_disc_real_prob = []
all_disc_fake_prob = []


def train_magail(expert_data, num_epochs=1000, batch_size=32):

    optimizers = init_optimizers()

    for epoch in range(num_epochs):

        decentralized_policy_data = collect_policy_trajectories(policies, batch_size=batch_size)
        centralized_policy_data = convert_decentralized_data_to_centralized(decentralized_policy_data)

        # --- Update discriminators ---
        real_prob, fake_prob = update_discriminators(centralized_policy_data, decentralized_policy_data, optimizers, batch_size)

        # --- Update policies and value (Actor Critic) ---
        generator_rewards, advantages, value_loss, policy_loss = update_policies(epoch, decentralized_policy_data, centralized_policy_data, optimizers, batch_size)

        # Diagnostic
        print_diagnostics(epoch, generator_rewards, advantages, value_loss, policy_loss, real_prob, fake_prob)

    env.close()

def init_optimizers():
    if USE_KFAC:
        optimizers = {
        agent: {
            # Replace Adam with KFAC for policy
            "policy": KFAC(
                policies[agent],
                learning_rate=policy_learning_rate_kfac,  # Typically larger than Adam
                damping=1e-2,           # Added to the curvature approximation (Fisher matrix) for numerical stability.  lower damping → lower learning_rate
                momentum=0.95,# Similar to Nesterov momentum
                momentum_type='regular',
                norm_constraint=0.002,       # For KL ≤ 0.001
                cov_ema_decay=0.95,
                adapt_damping=True,     # Let KFAC adjust damping
                update_cov_manually=True  # Required for control
            ),
            "value": torch.optim.Adam(value_nets[agent].parameters(), lr=value_learning_rate)
            } for agent in all_agents
        }

    else:
        optimizers = {
        agent: {
            # Replace Adam with KFAC for policy
            "policy": torch.optim.Adam(policies[agent].parameters(), lr=policy_learning_rate),
            "value": torch.optim.Adam(value_nets[agent].parameters(), lr=value_learning_rate)
            } for agent in all_agents
        }

    if DISC_TYPE == DiscType.CENTRALIZED:
        optimizers["disc"] = torch.optim.Adam(discriminator_centralized.parameters(), lr=centralized_learning_rate, weight_decay=centralized_weight_decay)
    elif DISC_TYPE == DiscType.DECENTRALIZED:
        for agent in all_agents:
            optimizers[agent]["disc"] = torch.optim.Adam(discriminators[agent].parameters(), lr=decentralized_learning_rate, weight_decay=decentralized_weight_decay)

    # for agent in all_agents:
    #     policies[agent].to(device)
    return optimizers

#region Data sample and processing

def collect_policy_trajectories(policies, batch_size=32):
    env = make_env()
    policy_data = {
        agent: {
            "states": [],
            "actions": [],
            "log_prob": [],
            "reward": [],
            "next_state": []
        }
        for agent in env.agents
    }
    obs, _ = env.reset()
    agent = env.agents[0]
    policy_data_length = len(policy_data[agent]["states"])
    while policy_data_length < batch_size:
        while env.agents:
            actions = {}
            for agent in env.agents:
                with torch.no_grad():
                    obs_tensor = torch.FloatTensor(obs[agent]).to(device)
                    action_probs = policies[agent](obs_tensor)

                # Add numerical stability checks
                if torch.isnan(action_probs).any() or (action_probs < 0).any():
                    print("Invalid action probabilities detected!")
                    action_probs = torch.softmax(action_probs, dim=-1)  # Force normalization
                    action_probs = torch.clamp(action_probs, min=1e-5, max=1-1e-5)

                # Simpling by probability - ensure exploration
                # Similar with Epsilon-Greedy, but used in policy-based algorithms. Epsilon-Greedy is in valued based
                action = torch.multinomial(action_probs, 1).item()
                actions[agent] = action

                # Data collection
                policy_data[agent]["states"].append(obs[agent])
                policy_data[agent]["actions"].append(action)

                log_prob = torch.log(action_probs[action].clamp(min=1e-8))
                policy_data[agent]["log_prob"].append(log_prob)

            obs, rewards, terminations, truncations, infos= env.step(actions)

            for agent in env.agents:

                with torch.no_grad():
                    # Decentralized
                    reward = torch.log(discriminators[agent](obs[agent], actions[agent]))

                policy_data[agent]["reward"].append(reward)

                if not terminations[agent]:
                    policy_data[agent]["next_state"].append(obs[agent])
                else:
                    policy_data[agent]["next_state"].append(None)

        env.reset()
        policy_data_length = len(policy_data[agent]["states"])

    return policy_data

def sample_batch_state_action_pairs(disc_type : DiscType, data, batch_size=32, agent = None):

    # Sample Expert data
    if disc_type == DiscType.CENTRALIZED:
        states = torch.FloatTensor(data["joint_states"]).to(device)
        actions = torch.LongTensor(data["joint_actions"]).to(device)
        # TODO centralized log_prob
    elif disc_type == DiscType.DECENTRALIZED:
        if not agent:
            throw_error("Please specify the agent in DECENTRALIZED mode")
        states = torch.FloatTensor(data[agent]["states"]).to(device)
        actions = torch.LongTensor(data[agent]["actions"]).to(device)
        log_probs = torch.stack(data[agent]["log_prob"]).to(device)

    # Make sure the (s,a) order does not change
    sample_indices = np.random.choice(len(states), batch_size, replace=False)
    batch_states = states[sample_indices]
    batch_actions = actions[sample_indices]
    batch_old_log_probs = log_probs[sample_indices]

    return batch_states, batch_actions, batch_old_log_probs

def encode_joint_actions(joint_actions):
    '''
    Convert joint actions to one-hot
    action [2,4] => [0,0,1,0,0,0,0,1] joint actions one hot
    action [2,3] => [0,0,1,0,0,0,1,0]
    :param joint_actions:
    :return:
    '''
    encoded = []
    for ja in joint_actions:
        speaker_act = torch.nn.functional.one_hot(torch.tensor(ja[0]), speaker_act_dim)
        listener_act = torch.nn.functional.one_hot(torch.tensor(ja[1]), listener_act_dim)
        encoded.append(torch.cat([speaker_act, listener_act]))
    return torch.stack(encoded).to(device)
#endregion

#region discriminators Networks

def update_discriminators(centralized_policy_data, decentralized_policy_data, optimizers, batch_size):
    if DISC_TYPE == DiscType.CENTRALIZED:
        real_prob, fake_prob = update_discriminators_centralized(centralized_expert_data, centralized_policy_data, optimizers, batch_size)
    elif DISC_TYPE == DiscType.DECENTRALIZED:
        for agent in all_agents:
           real_prob, fake_prob =  update_discriminators_decentralized(agent, decentralized_expert_data, decentralized_policy_data, optimizers, batch_size)

    return real_prob, fake_prob

def update_discriminators_decentralized(agent, expert_data, policy_data, optimizers, batch_size):
    # Sample batch from dataset
    expert_states, expert_actions = sample_batch_state_action_pairs(DISC_TYPE, expert_data, batch_size, agent)
    policy_states, policy_actions = sample_batch_state_action_pairs(DISC_TYPE, policy_data, batch_size, agent)

    # One-hot encode actions (different for speaker/listener)
    num_classes = act_dims[agent]
    expert_actions_onehot = torch.nn.functional.one_hot(expert_actions, num_classes=num_classes).float().to(device)
    policy_actions_onehot = torch.nn.functional.one_hot(policy_actions, num_classes=num_classes).float().to(device)

    expert_prob, policy_prob = discriminator_backward(discriminators[agent], optimizers[agent]["disc"], expert_states, expert_actions_onehot, policy_states, policy_actions_onehot)

    return expert_prob, policy_prob

def update_discriminators_centralized(centralized_expert_data, centralized_policy_data, optimizers, batch_size):
    # Centralized update
    # Shape [batch_size, 14], [batch_size, 2]
    expert_joint_states, expert_joint_actions = sample_batch_state_action_pairs(DISC_TYPE, centralized_expert_data, batch_size)
    policy_joint_states, policy_joint_actions = sample_batch_state_action_pairs(DISC_TYPE, centralized_policy_data, batch_size)

    expert_joint_actions_onehot = encode_joint_actions(expert_joint_actions)
    policy_joint_actions_onehot = encode_joint_actions(policy_joint_actions)

    expert_prob, policy_prob = discriminator_backward(discriminator_centralized, optimizers["disc"], expert_joint_states, expert_joint_actions_onehot, policy_joint_states, policy_joint_actions_onehot)

    return expert_prob, policy_prob

def discriminator_backward(discriminator, discriminator_optimizer, expert_states, expert_actions_onehot, policy_states, policy_actions_onehot):

    # Discriminator loss
    # max[log(D(expert)) + log(1 - D(policy))] => min[-log(D(expert)) - log(1 - D(policy))],

    expert_prob = discriminator(expert_states, expert_actions_onehot)
    policy_prob = discriminator(policy_states, policy_actions_onehot)

    real_loss = -torch.log(expert_prob).mean()
    fake_loss = -torch.log(1 - policy_prob).mean()
    disc_loss = real_loss + fake_loss

    discriminator_optimizer.zero_grad()
    disc_loss.backward()
    discriminator_optimizer.step()

    return expert_prob, policy_prob

#endregion

#region Policy and Value Networks
def update_policies(epoch, decentralized_policy_data, centralized_policy_data, optimizers, batch_size):
    for agent in all_agents:
        if USE_KFAC:
            optimizers[agent]["policy"].update_cov()
        if DISC_TYPE == DiscType.CENTRALIZED:
            generator_rewards, advantages, value_loss, policy_loss = update_policies_centralized(epoch, agent, decentralized_policy_data, centralized_policy_data, optimizers, batch_size)
        elif DISC_TYPE == DiscType.DECENTRALIZED:
            generator_rewards, advantages, value_loss, policy_loss = update_policies_decentralized(epoch, agent, decentralized_policy_data, optimizers, batch_size)

    return  generator_rewards, advantages, value_loss, policy_loss

def update_policies_centralized(epoch, agent, decentralized_policy_data, centralized_policy_data, optimizers, batch_size):

    # Centralized data
    policy_joint_states, policy_joint_actions = sample_batch_state_action_pairs(DISC_TYPE, centralized_policy_data, batch_size)
    policy_joint_actions_onehot = encode_joint_actions(policy_joint_actions)# actions [0,0] => [1,0,0,1,0,0,0,0]

    # decentralized data
    policy_states, policy_actions, old_log_probs = sample_batch_state_action_pairs(disc_type= DiscType.DECENTRALIZED, data=decentralized_policy_data, batch_size=batch_size, agent=agent)

    # Adversarial reward: log(D(s,a))
    # D(s,a) How much the descrimiator think it is from the expert, and we wanna maximize this reward
    with torch.no_grad():

        # MAGAIL paper section 4.1
        # Implicitly, Di - discriminators plays the role of a reward function for the generator,which in turn attempts to train the agent to maximize its reward thus fooling the discriminator
        # In centralized case, this input the joint state actions
        generator_rewards = torch.log(discriminator_centralized(policy_joint_states, policy_joint_actions_onehot)) # Shape [batch_size,1]

        # Compute value baseline
        values = value_nets[agent](policy_joint_states) # Shape [batch_size,1]


    advantages, policy_loss = update_value_and_policy_network(agent, optimizers, generator_rewards, values, policy_states, policy_actions, old_log_probs)
    return generator_rewards, advantages, values.mean(), policy_loss

def update_policies_decentralized(epoch, agent, policy_data, optimizers, batch_size):
    policy_states, policy_actions, old_log_probs = sample_batch_state_action_pairs(disc_type= DiscType.DECENTRALIZED, data=policy_data, batch_size=batch_size, agent=agent)
    # One-hot encode actions (different for speaker/listener)
    num_classes = act_dims[agent]
    policy_actions_onehot = torch.nn.functional.one_hot(policy_actions, num_classes=num_classes).float().to(device)

    # Adversarial reward: log(D(s,a))
    # D(s,a) How much the descrimiator think it is from the expert, and we wanna maximize this reward
    with torch.no_grad():

        # MAGAIL paper section 4.1
        # Implicitly, Di - discriminators plays the role of a reward function for the generator,which in turn attempts to train the agent to maximize its reward thus fooling the discriminator
        # In decentralized case, this input the local observations and agent's actions
        generator_rewards = torch.log(discriminators[agent](policy_states, policy_actions_onehot))

        # Compute value baseline
        values = value_nets[agent](policy_states)

    advantages, policy_loss = update_value_and_policy_network(agent, optimizers, generator_rewards, values, policy_states, policy_actions, old_log_probs)

    return generator_rewards, advantages, values.mean(), policy_loss

def update_value_and_policy_network(agent, optimizers, generator_rewards, values, decentralized_policy_states, decentralized_policy_actions, old_log_probs):

    # Compute advantages
    # TD error: (actual reward - estimate(baseline))
    # Then use this advantage as reward/reinforce signal
    # squeeze() function removes "trivia" dim (dimensions of size 1) form tensor's shape
    # Here values shape [batch_size, 1] => values.squeeze() => [batch_size] it become a vector
    # It can only apply on dimensions of size 1, eg: policy_actions shape [batch_size,2] policy_actions.squeeze().shape is till [batch_size,2]
    advantages = generator_rewards - values # Shape [batch_size,1]

    # Update value network (MSE Loss)
    value_loss = (advantages).pow(2).mean()
    # Should always clear the gradient before update
    value_loss.requires_grad = True
    optimizers[agent]["value"].zero_grad()
    value_loss.backward()
    optimizers[agent]["value"].step()

    if USE_KFAC:
        # --- KFAC Policy gradient ---
        policy_optimizer = optimizers[agent]["policy"]

        # Track forward pass
        with policy_optimizer.track_forward():
            action_probs = policies[agent](decentralized_policy_states)
            log_probs = torch.log(action_probs.gather(1, decentralized_policy_actions.unsqueeze(1)))
            policy_loss = -(log_probs * advantages.detach()).mean()

        # Track backward pass
        with policy_optimizer.track_backward():
            policy_loss.backward()

        # KFAC update steps
        policy_optimizer.update_cov()  # Update curvature approximation
        policy_optimizer.step(loss=policy_loss)  # Precondition gradients

        # In KFAC, the gradient can explode, causing softmax logits to underflow/overflow
        # Use clipping as a safeguard
        torch.nn.utils.clip_grad_norm_(policies[agent].parameters(), max_norm=0.5)
    else:

        PPO_EPOCHS = 5
        PPO_CLIP = 0.2
        ENTROPY_COEF = 0

        # PPO Policy update with multiple epochs
        policy_loss_total = 0.0
        for _ in range(PPO_EPOCHS):
            action_probs = policies[agent](decentralized_policy_states)  # [batch_size, act_dim]
            new_log_probs = torch.log(
                action_probs.gather(1, decentralized_policy_actions.unsqueeze(1)).clamp(min=1e-8)
            ).squeeze(1)  # [batch_size]
            ratio = torch.exp(new_log_probs - old_log_probs)  # [batch_size]

            print(f"PPO ratio: {ratio.mean()}")

            # PPO clipped surrogate objective
            surr1 = ratio * advantages.detach().squeeze()
            surr2 = torch.clamp(ratio, 1 - PPO_CLIP, 1 + PPO_CLIP) * advantages.detach().squeeze()
            policy_loss = -torch.min(surr1, surr2).mean()

            # Add entropy bonus for exploration
            # entropy = -torch.sum(action_probs * torch.log(action_probs.clamp(min=1e-8)), dim=-1).mean()
            # policy_loss = policy_loss - ENTROPY_COEF * entropy

            optimizers[agent]["policy"].zero_grad()
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policies[agent].parameters(), max_norm=0.5)  # Gradient clipping
            optimizers[agent]["policy"].step()

            policy_loss_total += policy_loss.item()

        policy_loss_avg = policy_loss_total / PPO_EPOCHS

    return advantages, policy_loss_avg

#endregion

def print_diagnostics(epoch, generator_rewards, advantages, value_loss, policy_loss, real_prob, fake_prob):
    if epoch % 50 == 0:
        print("------------------------------------------------------")
        training_eval_stats = evaluate_policy_for_training(policies, num_episodes=20)
        print(f"Epoch {epoch}: Avg Reward={training_eval_stats['avg_reward']:.2f}")
        all_episode_rewards.append(training_eval_stats["avg_reward"])

        # expert_eval = evaluate_policy_for_expert(expert_policy, num_episodes=20)
        all_expert_rewards.append(expert_mean_reward)
        print(f"Expert Policy: Avg Reward={expert_mean_reward}")

    all_generator_rewards.append(generator_rewards.detach().cpu().numpy().mean())
    all_advantages.append(advantages.detach().cpu().numpy().mean())
    all_values_estimates.append(value_loss.detach().cpu().numpy().mean())
    all_policy_losses.append(policy_loss.detach().cpu().numpy().mean())

    all_disc_real_prob.append(real_prob.detach().cpu().numpy().mean())
    all_disc_fake_prob.append(fake_prob.detach().cpu().numpy().mean())

    if epoch % 50 == 0 and epoch >= 50:

        mean_generator_rewards = np.mean(all_generator_rewards[-50:])
        mean_advantages = np.mean(all_advantages[-50:])
        mean_value_losses = np.mean(all_values_estimates[-50:])
        mean_policy_losses = np.mean(all_policy_losses[-50:])
        mean_disc_real_prob = np.mean(all_disc_real_prob[-50:])
        mean_disc_fake_prob = np.mean(all_disc_fake_prob[-50:])

        print(f"Epoch {epoch}: disc_real_prob={mean_disc_real_prob:.2f}, disc_fake_prob={mean_disc_fake_prob:.2f}")
        print("---")
        # print(f"Epoch {epoch}: generator_rewards={mean_generator_rewards:.2f}, advantages = {mean_advantages:.2f}, value_loss={mean_value_losses:.2f}, policy_loss={mean_policy_losses:.2f}")

# train_magail(decentralized_expert_data, num_epochs=2, batch_size=BATCH_SIZE)

#%% md
# ## Training
#%%
train_magail(decentralized_expert_data, num_epochs=training_epochs, batch_size=BATCH_SIZE)

#%%
