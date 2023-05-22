import sys
import os
from pathlib import Path

root_dir = Path(os.getcwd())

if str(root_dir.parents[1]) not in sys.path:
  sys.path.append(str(root_dir.parents[1]))

#print(sys.path)

from LMORL.BAN.API.agents.DQNHybrid import DQNHybrid

import gym

# available LL versions that are in gym and are MO:
# -LunarLander-v2-mo-custom
# -LunarLander-v2-mo


env = gym.make("LunarLander-v2-mo-custom")

input_size = env.observation_space.shape[0]
num_actions = int(env.action_space.n)
action_space = list(range(env.action_space.n))
learning_rate = 0.001
epsilon_decay = 0.996
epsilon_min = 0.1
batch_size = 64
train_start = 64
hidden_size = 128
BAN_SIZE = 3
max_memory_size = 10000

agent = DQNHybrid(input_size=input_size, num_actions=num_actions,
                  action_space=action_space, learning_rate=learning_rate,
                  epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                  batch_size=batch_size, hidden_size=hidden_size,
                  ban_size=3, max_memory_size=max_memory_size, train_start=100)

episodes = 350
mname = "fooo.model"

rewards, avg_rewards, timings, infos_lists = agent.learning(env=env,episodes = episodes, mname=mname, render=False)


"""
With 350 episodes, 
learning_rate = 0.001
epsilon_decay = 0.996
epsilon_min = 0.1
batch_size = 64
train_start = 64
hidden_size = 128
BAN_SIZE = 3
max_memory_size = 10000
Results:
Episode 349     timesteps:      277     Took    15.434138 sec - reward: [20.876232145678017, -100.0, -45.15000000000006]  | 100AvgReward: [-86.61653347279771, -98.0, -42.32069999999999]
"""
