import pickle
import numpy as np
import ray
from ray import tune
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from pettingzoo.mpe import simple_tag_v3
from ray.tune.registry import register_env
from ray.rllib.algorithms.ppo import PPOConfig

# Define the environment
def env_creator(config):
    env = simple_tag_v3.env(num_good=1, num_adversaries=3, num_obstacles=2, max_cycles=25, continuous_actions=False)

    env.reset()
    return PettingZooEnv(env)  # Wrap to make it Gymnasium-compatible

# Register the environment
register_env("simple_tag_v3", env_creator)

# Create a temporary environment to retrieve spaces.
temp_env = env_creator({})

# Get observation and action spaces
temp_env = env_creator({})
adversary_obs_space = temp_env.observation_space["adversary_0"]
adversary_act_space = temp_env.action_space["adversary_0"]
agent_obs_space = temp_env.observation_space["agent_0"]
agent_act_space = temp_env.action_space["agent_0"]

all_agents = ['adversary_0', 'adversary_1', 'adversary_2', 'agent_0']

def policy_mapping_fn(agent_id, episode, **kwargs):
    if "adversary" in agent_id:
        return "adversary_policy"
    else:
        return "agent_policy"


# Configure the PPO algorithm

config = (
    PPOConfig() \
    .environment("simple_tag_v3") \
    .framework("torch") \
    .multi_agent(
        policies={
            "adversary_policy": (None, adversary_obs_space, adversary_act_space, {}),#{"lr": 0.0001} Lower learning rate for stability in compatitive setting???
            "agent_policy": (None, agent_obs_space, agent_act_space, {}),
        },
        policy_mapping_fn=policy_mapping_fn,
        policies_to_train=["adversary_policy", "agent_policy"],
    ) \
    .training(
        lr=0.0001,  # Lower learning rate for stability in competitive settings
        clip_param=0.2,  # PPO Clipping
        train_batch_size=4000,
    ) \
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False) \
    .env_runners(num_env_runners=4,  # Parallelize for faster training
        num_envs_per_env_runner=8,)
    ).resources(num_gpus=1)


# ray.init()
# analysis = tune.run(
#     "PPO",
#     config=config.to_dict(),
#     stop={"training_iteration": 50},
#     # stop={"env_runners/episode_reward_mean": -12},
#     checkpoint_freq=10,
#     checkpoint_at_end=True,
# )





# best_trial = analysis.get_best_trial(metric="env_runners/episode_reward_mean", mode="max")
# if best_trial:
#     best_checkpoint = analysis.get_best_checkpoint(trial=best_trial, metric="env_runners/episode_reward_mean", mode="max")
# else:
#     raise ValueError("No valid trial found.")

# # Restore the trainer from the best checkpoint
# trainer = config.build()
# trainer.restore(best_checkpoint)


# Define the checkpoint path (replace this with your actual path)
checkpoint_path = "C:/Users/wangy/ray_results/PPO_2025-04-07_22-16-16/PPO_simple_tag_v3_28e64_00000_0_2025-04-07_22-16-16/checkpoint_000002"  # This could be the path to a checkpoint folder or a specific checkpoint file
# Restore the trainer from the best checkpoint
trainer = config.build()
trainer.restore(checkpoint_path)

def generate_expert_data(trainer, num_episodes=50):
    # env = simple_tag_v3.parallel_env(continuous_actions=False, render_mode="rgb_array", max_cycles=50)
    env = simple_tag_v3.parallel_env(continuous_actions=False, render_mode="rgb_array", max_cycles=50)

    obs, _ = env.reset()  # Unpack properly
    
    all_agent_rewards_per_episode = {agent: [] for agent in env.agents}

    
    expert_data = {agent: {"states": [], "actions": []} for agent in env.agents}

    for _ in range(num_episodes):
        obs, _ = env.reset()  # Unpack here as well
        # print(f"Agents after reset: {env.agents}")  # Check if agents are populated after reset
        done = {agent: False for agent in env.agents}
        episode_reward = {agent: 0 for agent in env.agents}

        while not all(done.values()):
            actions = {}
            for agent in env.agents:
                policy_id = policy_mapping_fn(agent, None)  # Get correct policy
                policy = trainer.get_policy(policy_id)  # Fetch policy from trainer

                return_from_compute_single_action = policy.compute_single_action(obs[agent], explore=False)
                actions[agent] = return_from_compute_single_action[0]
            
            next_obs, rewards, done, infos, _ = env.step(actions)
            
            for agent in env.agents:
                if not done[agent]:
                    expert_data[agent]["states"].append(obs[agent])
                    expert_data[agent]["actions"].append(actions[agent])
                episode_reward[agent] += rewards[agent]
            
            obs = next_obs  # Update observations

        for agent in all_agents:
            all_agent_rewards_per_episode[agent].append(episode_reward[agent])
        

    agent_mean_rewards = {agent: np.mean(all_agent_rewards_per_episode[agent]) for agent in all_agents}
    return expert_data, agent_mean_rewards


for num_episodes in [50, 100, 150, 200]:
    # Generate expert data using the trained policy
    expert_data, agent_mean_rewards = generate_expert_data(trainer, num_episodes=num_episodes)
    print(f"num_episodes: {num_episodes}, agent_mean_rewards: {agent_mean_rewards}")
    file_name = f"expert_data_rllib_simple_tag_{num_episodes}.pickle"
    # Save expert data
    with open(file_name, "wb") as f:
        pickle.dump(expert_data, f)

# Shutdown Ray
ray.shutdown()
