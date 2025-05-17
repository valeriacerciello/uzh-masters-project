# Expert Data Info

---

In this folder we have pickle file for expert state action pairs for different episodes
Each were sampled using trained expert policy, max timestep = 50

## Simple Push
Reward obtained by experts
```
num_episodes: 50, agent_mean_rewards: {'adversary_0': np.float64(-3.8612088458516167), 'agent_0': np.float64(-9.842320439565086)}
num_episodes: 100, agent_mean_rewards: {'adversary_0': np.float64(-3.6196483258264704), 'agent_0': np.float64(-8.829925458328013)}
num_episodes: 150, agent_mean_rewards: {'adversary_0': np.float64(-3.443898794170685), 'agent_0': np.float64(-8.906258705020702)}
num_episodes: 200, agent_mean_rewards: {'adversary_0': np.float64(-3.388806385134402), 'agent_0': np.float64(-9.126263699317953)}
```

## Simple Tag (Predator Prey)
Reward obtained by experts
The adversary_0, adversary_1, adversary_2 are sharing the same reward. (Checked MPE code)
```
num_episodes: 50, agent_mean_rewards: {'adversary_0': np.float64(11.8), 'adversary_1': np.float64(11.8), 'adversary_2': np.float64(11.8), 'agent_0': np.float64(-17.815057470447442)}
num_episodes: 100, agent_mean_rewards: {'adversary_0': np.float64(12.7), 'adversary_1': np.float64(12.7), 'adversary_2': np.float64(12.7), 'agent_0': np.float64(-22.968722793442744)}
num_episodes: 150, agent_mean_rewards: {'adversary_0': np.float64(13.4), 'adversary_1': np.float64(13.4), 'adversary_2': np.float64(13.4), 'agent_0': np.float64(-23.07583968667756)}
num_episodes: 200, agent_mean_rewards: {'adversary_0': np.float64(15.25), 'adversary_1': np.float64(15.25), 'adversary_2': np.float64(15.25), 'agent_0': np.float64(-21.15318950220318)}
```


