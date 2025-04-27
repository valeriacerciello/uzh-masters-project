import pickle
import ray
from ray import tune
from ray.rllib.env import PettingZooEnv
from ray.rllib.algorithms.ppo import PPOConfig
from pettingzoo.mpe import simple_speaker_listener_v4

# Define the environment
def env_creator(config):
    env = simple_speaker_listener_v4.env()
    env.reset()
    return PettingZooEnv(env)  # Wrap to make it Gymnasium-compatible

# Register the environment
from ray.tune.registry import register_env
register_env("simple_speaker_listener", env_creator)


import gymnasium as gym

# Create a temporary environment to retrieve spaces.
temp_env = env_creator({})

speaker_obs_space = temp_env.observation_space["speaker_0"]
speaker_act_space = temp_env.action_space["speaker_0"]
listener_obs_space = temp_env.observation_space["listener_0"]
listener_act_space = temp_env.action_space["listener_0"]

def policy_mapping_fn(agent_id, episode, **kwargs):
    if "speaker" in agent_id:
        return "speaker_policy"
    else:
        return "listener_policy"


# Configure the PPO algorithm
from ray.rllib.algorithms.ppo import PPOConfig

config = PPOConfig() \
    .environment("simple_speaker_listener") \
    .framework("torch") \
    .multi_agent(
        policies={
            "speaker_policy": (None, speaker_obs_space, speaker_act_space, {}),
            "listener_policy": (None, listener_obs_space, listener_act_space, {}),
        },
        policy_mapping_fn=policy_mapping_fn,
    ) \
    .api_stack(enable_rl_module_and_learner=False, enable_env_runner_and_connector_v2=False) \
    .env_runners(num_env_runners=0)


ray.init()
analysis = tune.run(
    "PPO",
    config=config.to_dict(),
    stop={"env_runners/episode_reward_mean": 10},
    checkpoint_at_end=True,
    
)





best_trial = analysis.get_best_trial(metric="env_runners/episode_reward_mean", mode="max")
if best_trial:
    best_checkpoint = analysis.get_best_checkpoint(trial=best_trial, metric="env_runners/episode_reward_mean", mode="max")
else:
    raise ValueError("No valid trial found.")

# Restore the trainer from the best checkpoint
trainer = config.build()
trainer.restore(best_checkpoint)


# # Define the checkpoint path (replace this with your actual path)
# checkpoint_path = "C:/Users/wangy/ray_results/PPO_2025-02-23_14-39-33/PPO_simple_speaker_listener_9d625_00000_0_2025-02-23_14-39-33/checkpoint_000000"  # This could be the path to a checkpoint folder or a specific checkpoint file
# # Restore the trainer from the best checkpoint
# trainer = config.build()
# trainer.restore(checkpoint_path)


def generate_expert_data(trainer, num_episodes=50):
    env = simple_speaker_listener_v4.parallel_env(continuous_actions=False, render_mode="rgb_array", max_cycles=25)
    
    obs, _ = env.reset()  # Unpack properly
    
    expert_data = {agent: {"states": [], "actions": []} for agent in env.agents}

    for _ in range(num_episodes):
        obs, _ = env.reset()  # Unpack here as well
        # print(f"Agents after reset: {env.agents}")  # Check if agents are populated after reset
        done = {agent: False for agent in env.agents}
        
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
            
            obs = next_obs  # Update observations
    return expert_data

# Generate expert data using the trained policy
expert_data = generate_expert_data(trainer, num_episodes=50)

# Save expert data
with open("expert_data_rllib.pickle", "wb") as f:
    pickle.dump(expert_data, f)

# Shutdown Ray
ray.shutdown()
