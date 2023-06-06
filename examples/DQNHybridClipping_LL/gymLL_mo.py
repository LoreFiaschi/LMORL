import sys
import os
from pathlib import Path

from matplotlib import pyplot as plt

root_dir = Path(os.getcwd())

if str(root_dir.parents[1]) not in sys.path:
  sys.path.append(str(root_dir.parents[1]))

#print(sys.path)
from LMORL.BAN.API.ban_utils import Ban
from LMORL.BAN.API.agents.DQNHybrid import DQNHybrid

import gym

MAX_TIMESTEPS = 1000

env = gym.make("LunarLander-v2-mo", render_mode="rgb_array", max_episode_steps=MAX_TIMESTEPS)


input_size = env.observation_space.shape[0]
num_actions = int(env.action_space.n)
action_space = list(range(env.action_space.n))
learning_rate = 0.001
epsilon_decay = 0.995
epsilon_min = 0.1
batch_size = 64
hidden_size = 128
BAN_SIZE = 3
max_memory_size=100000
train_start = 1000
use_clipping = True
clipping_tol = 1.0

agent = DQNHybrid(input_size=input_size, num_actions=num_actions,
                  action_space=action_space, learning_rate=learning_rate,
                  epsilon_decay=epsilon_decay, epsilon_min=epsilon_min,
                  batch_size=batch_size, hidden_size=hidden_size,
                  ban_size=3, max_memory_size=max_memory_size, train_start=train_start, use_clipping=use_clipping, clipping_tol=clipping_tol)


total_reward, num_timestep, elapsed_episode, animated_gif_file = agent.run_episode(env, title="First run", render=False, verbose=False)

#img = IpyImg(data=animated_gif_file.getbuffer(), format='png')
with open("gymLL_mo_results/before_training.gif", "wb+") as f:
    f.write(animated_gif_file.getbuffer())

EPISODES = 600
REPLAY_FREQUENCY=8
mname = "fooo.model"

total_rewards = []
total_avg_rewards   = []
total_timings = []

def early_stopping(reward:list)-> bool:
    if sum(reward) >= 200:
        return True
    return False

THRESHOLD_EXCEEDED_CONSECUTIVELY = 2

rewards, avg_rewards, timings, infos_lists = agent.learning(env=env,episodes=EPISODES, replay_frequency=REPLAY_FREQUENCY, mname=mname, verbose=True, early_stopping=early_stopping, THRESHOLD_EXCEEDED_CONSECUTIVELY=THRESHOLD_EXCEEDED_CONSECUTIVELY)

total_reward, num_timestep, elapsed_episode, animated_gif_file = agent.run_episode(env, title="After training", render=False, verbose=False)
#IpyImg(data=animated_gif_file.getbuffer(), format='png')

with open("gymLL_mo_results/after_training.gif", "wb+") as f:
    f.write(animated_gif_file.getbuffer())

r = Ban.display_plot(rewards, len(rewards), "Total rewards", call_plot=False, use_BanPlots=False)
r.savefig(f"gymLL_mo_results/total_rewards_plot_max_timesteps{MAX_TIMESTEPS}_episodes{len(rewards)}.png")

r = Ban.display_plot(rewards, len(rewards), "Total rewards", call_plot=True, use_BanPlots=True)
#r.savefig("gymLL_mo_results/total_rewards_plot__BANPlots.png")

r = Ban.display_plot(avg_rewards, len(avg_rewards), "Total AVG rewards", call_plot=False, use_BanPlots=False)
r.savefig(f"gymLL_mo_results/avg_rewards_plot_max_timesteps{MAX_TIMESTEPS}_episodes{len(rewards)}.png")

r = Ban.display_plot(avg_rewards, len(avg_rewards), "Total AVG rewards", call_plot=True, use_BanPlots=True)
#r.savefig("gymLL_mo_results/avg_rewards_plot__BANPlots.png")