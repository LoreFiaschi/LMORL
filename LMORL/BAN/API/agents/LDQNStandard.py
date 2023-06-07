import gymnasium as gym
import random, math
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm 
from collections import namedtuple, deque
import math
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from LMORL.BAN.API.agents.agent import Agent

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state'))

# DQN model
class DQN(nn.Module):

	def __init__(self, n_observations, out_size, hidden):
		super(DQN, self).__init__()
		self.out_size = out_size
		self.layer1 = nn.Linear(n_observations, hidden)
		self.layer2 = nn.Linear(hidden, hidden)
		self.layer3 = nn.Linear(hidden, np.prod(out_size))

	# Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
	def forward(self, x):
		x = F.relu(self.layer1(x))
		x = F.relu(self.layer2(x))
		x = self.layer3(x)
		return x.view((x.size(0),) + self.out_size)

class LDQNStandard(Agent):

    PRINT_AS_BANs = False

    def __init__(self, input_size : int, num_actions : int, action_space, 
	      learning_rate : float, epsilon_decay : float, 
	      epsilon_min : float, batch_size : int, 
	      hidden_size : int,
	      reward_size : int, 
	      device, 
	      max_memory_size : int = 10000, train_start : int = 100, 
          discount_factor : float = 0.99, 
	      slack : float = 0.05):
		# reward_size is assumed == ban_size
        super().__init__(input_size, num_actions, action_space, reward_size, max_memory_size, train_start)


        self.input_size = input_size
        self.action_space = action_space
        self.num_actions = num_actions
        self.reward_size = reward_size
        self.out_size = (reward_size, num_actions)
        self.slack = slack
        self.device = device
        self.criterion = nn.SmoothL1Loss()
        self.permissible_actions = range(self.num_actions)

        # hyper parameters for DQN
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate
        self.epsilon = 1.0
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.train_start = train_start

        # create replay memory using deque
        self.memory = deque(maxlen=max_memory_size)

        # create main model and target model
        self.model = DQN(input_size, self.out_size, hidden_size).to(self.device)
        self.target_model = DQN(input_size, self.out_size, hidden_size).to(self.device)
        # initialize target model
        self.target_model.load_state_dict(self.model.state_dict())

        self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, amsgrad=True)

    def _episode_end(self):
        self._update_target_model()
        self._update_epsilon()

    def act(self, state):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        return self.__select_action(state).item()

	# Epsilon-greedy selection
    def __select_action(self, state):
        if random.random() <= self.epsilon:
            action = random.choice(self.action_space)

        else:
            with torch.no_grad():
                q_value = self.model(state).squeeze()
                action = self.__arglexmax(q_value)

        return torch.tensor([action])


    def __arglexmax(self, Q):
        permissible_actions = self.permissible_actions

        for i in range(self.reward_size):
            #if len(permissible_actions) == 1:
            #	break
            
            lower_bound = Q[i, permissible_actions].max(0)[0]
            lower_bound -= self.slack * abs(lower_bound)
            permissible_actions = [a for a in permissible_actions if Q[i, a] >= lower_bound]

        return random.choice(permissible_actions)

    def _add_experience(self, state, action, reward, next_state, done: bool):
        state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        next_state = None if done else torch.tensor(next_state, dtype=torch.float32, device=self.device).unsqueeze(0)
        reward = torch.tensor(reward[:self.reward_size], device=self.device)
        # NOTE: in order to obtain the right shape for the reward tensor, 
        # we have to keep only the first reward_size components of the reward vector
    
        self.__add_experience(state, action, reward, next_state)

	# save sample <s, a, r, s'>. into replay memory
    def __add_experience(self, state, action, reward, next_state):
        self.memory.append((state,action,reward,next_state))


    # Compute Q-value of the arrival state
    def __get_q_value(self, non_final_next_states):
        # choose a' in s' according with most updated netweork
        q_values = self.model(non_final_next_states)
        actions = torch.empty(len(non_final_next_states), device=self.device, dtype=torch.int64)

        for i in range(len(non_final_next_states)):
            actions[i] = self.__arglexmax(q_values[i,:])
        actions = torch.vstack([actions] * self.reward_size).T.unsqueeze(-1)
        # NOTE: in order to obtain the right shape for the actions tensor, 
        # we have vstack self.reward_size times the original actions

        # evaluate a' according wih target network

        return self.target_model(non_final_next_states).gather(2, actions).squeeze()

    def _update_target_model(self):
         self.__update_target_model()
    # change the weights of the target network
    def __update_target_model(self, tau = 0.01):
        weights = self.model.state_dict()
        target_weights = self.target_model.state_dict()
        for i in target_weights:
            target_weights[i] = weights[i] * tau + target_weights[i] * (1-tau)
        self.target_model.load_state_dict(target_weights)

    def _experience_replay(self):
        self.__experience_replay()

    # learn from ERB
    def __experience_replay(self):
        # if there are not enough samples: do nothing
        if len(self.memory) < self.train_start:
            return

        batch = random.sample(self.memory, self.batch_size)
        batch = Transition(*zip(*batch))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])

        state_batch = torch.cat(batch.state)
        action_batch = torch.tensor([batch.action] * self.reward_size).T.unsqueeze(-1)
        # NOTE: in order to obtain the right shape for the actions tensor, 
        # we have vstack self.reward_size times the original actions

        reward_batch = torch.stack(batch.reward)

        state_action_values = self.model(state_batch).gather(2, action_batch).squeeze()
        next_state_values = torch.zeros((self.batch_size, self.reward_size), device=self.device)
        
        with torch.no_grad():
            next_state_values[non_final_mask] = self.__get_q_value(non_final_next_states)

        expected_state_action_values = (self.discount_factor * next_state_values) + reward_batch

        loss = self.criterion(state_action_values, expected_state_action_values)
            
        # train the Q network
        self.optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()

    def _update_epsilon(self):
         return self.__update_epsilon()

    # decrease exploration, increase exploitation
    def __update_epsilon(self):
        # if there are not enough samples: continue exploring
        if len(self.memory) < self.train_start:
            return

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def get_epsilon(self):
        return self.epsilon

    def save_model(self, filename):
        dir_path = os.path.dirname(os.path.realpath(__file__))
        torch.save(self.model.state_dict(), '{}-model.pt'.format(os.path.join(dir_path, '/'+filename)))


#if __name__ == '__main__':
#
#    device = torch.device("cpu")
#
#    env = gym.make('LunarLander-v2')
#    env.reset(seed = 0)
#    input_size = env.observation_space.shape[0]
#    num_actions = env.action_space.n
#    reward_size = 2
#    landings = []
#    landings_in_pad = []
#    score = []
#    vector_score = []
#    epsilon_record = []
#    episodes = 700
#    replay_frequency = 1 # every 2
#
#    # create a DQN model
#    discount_factor = 0.99
#    learning_rate = 0.0001
#    epsilon_decay = 0.992
#    epsilon_min = 0.05
#    hidden = 128
#    batch_size = 128
#    train_start = 128
#    slack = 0.05
#    agent = LDQNStandard(env.action_space, input_size, reward_size, slack, epsilon_decay, epsilon_min, discount_factor, learning_rate, hidden, batch_size, train_start, device)
#
#
#    for e in tqdm(range(episodes), desc="Learning..."):
#        done = False
#        iteration = 1
#
#        state, _ = env.reset()
#        episode_score = 0
#        #episode_vector_score = torch.zeros(reward_size, device=device)
#
#        while not done:
#            # take an action
#            action = agent.act(state)
#            next_state, reward_sc, terminated, truncated, info = env.step(action)
#
#            # instructions for consistency
#            done = terminated or truncated
#            
#            episode_score += reward_sc
#            
#            #reward = torch.tensor([reward_sc + info["fuel_consumption"], -info["fuel_consumption"]], device=device)
#            #episode_vector_score += reward # where reward is a tensor
#
#            # add <s,a,r,s'> to ERB
#            reward = [reward_sc + info["fuel_consumption"], -info["fuel_consumption"]]
#            agent.__add_experience(state, action, reward, next_state)
#
#            state = next_state
#
#            iteration += 1
#
#            if iteration & replay_frequency == 0:
#                # train the agent
#                agent.__experience_replay()
#                # update target model
#                agent.__update_target_model(0.01)
#
#            if done:				
#                agent.__update_epsilon()
#                landings.append(int(info["landed"]))
#                landings_in_pad.append(int(info["landed"] and info["in_pad"]))
#                score.append(episode_score)
#                #vector_score.append(episode_vector_score)
#                epsilon_record.append(agent.get_epsilon())
#
#    env.close()
#
#    #agent.save_model("models/LL")
#
#    plot_score(score, 15, "Original Reward")
#
#    plt.figure()
#    plt.plot(epsilon_record);
#    plt.title("Epsilon decay");
#
#    vs = VectorScore(*zip(*vector_score))
#    plot_score(vs.flight, 15, 'Flight')
#    #plot_score(vs.landing, 15, 'Landing')
#    plot_score(vs.fuel, 15, 'Fuel')
#
#    plot_score(landings, 15, 'N. Landings')	
#    plot_score(landings_in_pad, 15, 'N. Landings in Pad')
#
#    plt.show()