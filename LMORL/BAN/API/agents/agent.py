import mo_gymnasium as mo_gym
from LMORL.Environment import Environment

from LMORL.BAN.API.agents.timestep import Timestep

from julia.api import Julia

from julia import Main

from datetime import datetime
from abc import ABC, abstractmethod
import pathlib
import numpy as np

DEBUG = True
ELAPSED_THRESHOLD = 1

class Agent(ABC):
    def __init__(self, input_size : int, num_actions : int, action_space, ban_size : int, max_memory_size : int = 100, train_start : int = 100) -> None:
        
        self.input_size = input_size
        self.num_actions = num_actions
        self.action_space = action_space
        
        self.max_memory_size = max_memory_size
        self.train_start = train_start
        self.memory = []
        self.ban_size = ban_size

        self._jl = Julia(compiled_modules=True)
        self._main = Main

        path = pathlib.Path(__file__).parent.resolve()
        path = str(path).replace("\\", "\\\\")
        self._julia_eval(f"cd(\"{path}\")")

        # TODO: decide if the right version of BAN library has to be included here
        self._main.BAN_SIZE = self.ban_size

        # self._julia_eval('include("julia-test-1.jl")')
        pass
    
    def _julia_eval(self, cmd_string : str):
        return self._jl.eval(cmd_string)

    def _get_reward_dim(self):
        """
        consider if allow to use a lower ban_size than reward_dim
        """
        return self.ban_size

    def agent_learning(self, env : Environment, episodes : int, mname : str, replay_frequency : int = 1, dump_period : int = 50, reward_threshold : float = None):
        """
        returns rewards, avg_rewards, timings
        """
        # we need to know the size of the reward, 
        # that same size will be used for BANs dimension
        #reward_dim = env.get_reward_dim()
        
        rewards = []
        avg_rewards = []

        timings = [0]   #TODO: should this list be empty?

        solved = False

        i = 1

        while i < episodes and not solved:
            begin_episode_time = datetime.now()
            state, infos = env.reset()
            done = False
            t = 0
            totrew = [0] * self._get_reward_dim()
            while not done:
                start_time = datetime.now()
                action_index = self._act(state)
                action = self.action_space[action_index]
                next_state,reward,terminated, truncated, infos=self._ban_step(env, action)
                done = bool( terminated or truncated )
                # reward is MO, then it is a list
                # totrew+=reward
                before = datetime.now()
                totrew = [sum(foo) for foo in zip(totrew, reward)]
                if DEBUG:
                    elapsed = (datetime.now() - before).total_seconds()
                    if elapsed > ELAPSED_THRESHOLD:
                        print(f"sum(foo) for foo in zip(totrew, reward) took {elapsed} seconds")

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
            now = datetime.now()
            elapsed_episode = (now - begin_episode_time).total_seconds()
            print_time = now.strftime("%H:%M:%S")
            print(f"{print_time}\tEpisode\t{i}\ttimesteps:\t{t}\tTook\t{elapsed_episode} sec - reward:\t{totrew}\t| 100AvgReward: {avg_reward}")
            i+=1
    
        return rewards, avg_rewards, timings

    @abstractmethod
    def _act(self, state):
        """
        returns the index of the action to perform
        - agent dependent
        """
        print("Agent._act() method without implementation was called! You have to implement custom agent's act method!")

        pass

    def _ban_step(self, env : Environment, action):
        """
        returns next_state,reward,done
        - the reward is returned as a list
        """
        state, reward, terminated, truncated, information = env.step(action)

        # (in the case the reward has to be returned as a BAN):
        # consider if it is needed to check if
        # the reward's first component is <> 0 or not (ref 4571)
        # answer: this is done by the method add_experience() which has to call parse_ban_from_array(reward_list, ban_size) which
        # is in charge of normalizing the BAN representation

        #assert type(reward) in [list, np.ndarray], f"[!] reward type is {type(reward)} instead of list or ndarray, content: {reward}"
        #currently the reward is returned as an array, not a BAN
        return state, reward, terminated, truncated, information

    def _add_experience(self, state,action,reward,next_state,done : bool):
        """
        stores the information about the last episode in agent memory
        """
        print("+++++++++++ Agent._add_experience() was called ++++++++++++++++++")
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