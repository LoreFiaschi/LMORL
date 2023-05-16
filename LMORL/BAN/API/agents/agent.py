import mo_gymnasium as mo_gym
from LMORL.Environment import Environment

from timestep import Timestep

from agent import Agent
from julia.api import Julia

from julia import Main

from datetime import datetime
from abc import ABC, abstractmethod

class Agent(ABC):
    def __init__(self, input_size : int, num_actions : int, action_space, ban_size : int, max_memory_size : int = 100, train_start : int = 100) -> None:
        
        self.input_size = input_size
        self.num_actions = num_actions
        self.action_space = action_space
        
        self.max_memory_size = max_memory_size
        self.train_start = train_start
        self.memory = []
        self.ban_size = ban_size

        self._jl = Julia(compiled_modules=False)
        self._main = Main
        
        # TODO: decide if the right version of BAN library has to be included here

        # self._jl.eval('include("julia-test-1.jl")')
        pass

    def agent_learning(self, env : Environment, episodes : int, mname : str, replay_frequency : int = 1, dump_period : int = 50, reward_threshold : float = None):
        # we need to know the size of the reward, 
        # that same size will be used for BANs dimension
        reward_dim = env.get_reward_dim()
        
        rewards = []
        avg_rewards = []

        timings = [0]   #TODO: should this list be empty?

        solved = False

        i = 1

        while i < episodes and not solved:
            state = env.reset()
            done = False
            t = 0
            totrew = 0
            while not done:
                start_time = datetime.now()
                action_index = self._act(state)
                action = self.action_space[action]
                next_state,reward,done=self._ban_step(env, action)
                # reward is MO, then it is a list
                # totrew+=reward
                totrew = [sum(foo) for foo in zip(totrew, reward)]

                self._add_experience(state,action_index,reward,next_state,done)
                state=next_state
                t+=1
                if t % replay_frequency==0 :
                    self._experience_replay()
                tmng = ( datetime.now() - start_time ).total_seconds()
                timings.append(tmng)
            
            self._episode_end()

            rewards.append(totrew)

            if i % dump_period == 0:
                self._dump_model_to_file(mname)

            if i >= 100:
                # https://www.geeksforgeeks.org/python-ways-to-sum-list-of-lists-and-return-sum-list/
                avg_reward = [sum(comp[i - 100 : i - 1])/100 for comp in zip(*rewards)]
                #( sum(rewards[i - 100: i - 1]) ) / 100  --> not sium because reward is a NA vector

                # TODO: consider if adding an exit condition that use as threshold the
                # input parameter reward_threshold

            else:
                avg_reward = [sum(comp)/i for comp in zip(*rewards)]

            avg_rewards.append(avg_reward)
            print(f"Episode {i} - reward: {totrew} | 100AvgReward: {avg_reward}")

        return rewards, avg_rewards, timings

    @abstractmethod
    def _act(self, state):
        """
        returns the index of the action to perform
        - agent dependent
        """
        # TODO

        pass

    def _ban_step(self, env : Environment, action):
        """
        returns next_state,reward,done
        - the reward is returned as a list
        """
        state, reward, done, information = env.step(action)

        # (in the case the reward has to be returned as a BAN):
        # TODO: consider if it is needed to check if 
        # the reward's first component is <> 0 or not (ref 4571)

        #TODO: at the moment the reward is returned as an array, not a BAN
        return state, reward, done, information

    def _add_experience(self, state,action,reward,next_state,done : bool):
        """
        stores the information about the last episode in agent memory
        """
        episode = Timestep(state, action, reward, next_state, done)
        if len(self.memory) >= self.max_memory_size:
            self.memory.pop(0)
        self.memory.append(episode)

    @abstractmethod
    def _experience_replay(self):
        """
        it depends on the agent type, must be implemented in the derived class
        """
        if len(self.memory) < self.train_start:
            return
        
        pass

    #@abstractmethod
    def _episode_end():
        """
        this method is called at each episode end,
        extend this method to add custom methods call and updates for agent-specific needs
        """
        pass

    def _dump_model_to_file(self, mname):
        pass