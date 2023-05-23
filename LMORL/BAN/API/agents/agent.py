import mo_gymnasium as mo_gym
from LMORL.Environment import Environment

from LMORL.BAN.API.agents.timestep import Timestep

from julia.api import Julia

from julia import Main

from datetime import datetime
from abc import ABC, abstractmethod
import pathlib
import numpy as np

from PIL import Image
import PIL.ImageDraw as ImageDraw
import matplotlib.pyplot as plt

from io import BytesIO

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

        # DONE: decide if the right version of BAN library has to be included here
        # the include is performed inside jl file of the agent
        self._main.BAN_SIZE = self.ban_size
    
    def _julia_eval(self, cmd_string : str):
        return self._jl.eval(cmd_string)

    def _get_reward_dim(self):
        """
        TODO: consider if allow to use a lower ban_size than reward_dim
        """
        return self.ban_size

    def _render(self, env, title, timestep, live_rendering : bool=True):

        frame = env.render()
        img_frame = Image.fromarray(frame)

        drawer = ImageDraw.Draw(img_frame)
        if np.mean(img_frame) < 128:
            text_color = (255,255,255)
        else:
            text_color = (0,0,0)
        drawer.text((img_frame.size[0]/20,img_frame.size[1]/18), f'{title}', fill=text_color)


        if live_rendering:
            plt.figure(1); plt.clf()
            plt.title(f'{title}')
            plt.imshow(img_frame)
            plt.pause(0.01)
        return img_frame

    def learning(self, env : Environment, episodes : int, mname : str, replay_frequency = None, dump_period = None, reward_threshold : float = None, render : bool = False, verbose:bool=True):
        """
        - if replay_frequency is None or is <= 0, then the learning is done at episode end
        - if replay_frequency is > 0, the learning is performed every replay_frequency timesteps
        - if dump_period is None no dump is performed, else it is performed every dump_period episodes
        returns rewards, avg_rewards, timings, infos
        - full_infos_dict is a dict of lists of lists: each element of the dictionary is a key of the env infos, and each of its lists is the list of infos of each timestep of an episode
        """
        # we need to know the size of the reward, 
        # that same size will be used for BANs dimension
        #reward_dim = env.get_reward_dim()
        
        rewards = []
        avg_rewards = []
        full_infos_dict = {}

        timings = []

        solved = False
        threshold_exceeded = 0
        THRESHOLD_EXCEEDED_CONSECUTIVELY = 3

        episode_number = 1

        learn_at_episode_end = (replay_frequency is None or replay_frequency <= 0)

        while episode_number <= episodes and not solved:
            begin_episode_time = datetime.now()
            state, infos = env.reset()
            done = False
            t = 0
            totrew = [0] * self._get_reward_dim()
            infos_dict = {}
            while not done:
                start_time = datetime.now()
                action_index = self.act(state)
                action = self.action_space[action_index]
                next_state,reward,terminated, truncated, infos=self._ban_step(env, action)
                # render environment animation
                if render:
                    self._render(env, f"Episode {episode_number}", t, live_rendering=True)
                done = bool( terminated or truncated )
                # reward is MO, then it is a list
                # totrew+=reward
                totrew = [sum(foo) for foo in zip(totrew, reward)]
 
                self._add_experience(state,action_index,reward,next_state,done)
                state=next_state
                t+=1
                if not learn_at_episode_end and t % replay_frequency==0 :
                    self._experience_replay()
                tmng = ( datetime.now() - start_time ).total_seconds()
                timings.append(tmng)
                
                if type(infos) == dict:
                    for key in infos.keys():
                        if key not in infos_dict.keys():
                            infos_dict[key] = []
                        infos_dict[key].append(infos[key])

            if learn_at_episode_end:
                self._experience_replay()
            self._episode_end()

            rewards.append(totrew)

            if dump_period is not None and episode_number % dump_period == 0:
                self.dump_model_to_file(mname)

            if episode_number >= 100:
                # https://www.geeksforgeeks.org/python-ways-to-sum-list-of-lists-and-return-sum-list/
                avg_reward = [sum(comp[episode_number - 100 : episode_number ])/100 for comp in zip(*rewards)]
                #( sum(rewards[i - 100: i - 1]) ) / 100  --> not sium because reward is a NA vector

            else:
                avg_reward = [sum(comp)/episode_number for comp in zip(*rewards)]

            avg_rewards.append(avg_reward)
            
            for key in infos_dict.keys():
                if key not in full_infos_dict.keys():
                    full_infos_dict[key] = []
                full_infos_dict[key].append(infos_dict[key])

            now = datetime.now()
            elapsed_episode = (now - begin_episode_time).total_seconds()
            print_time = now.strftime("%H:%M:%S")
            if verbose: print(f"{print_time}\tEpisode\t{episode_number}\ttimesteps:\t{t}\tTook\t{elapsed_episode} sec - reward:\t{totrew}\t| 100AvgReward: {avg_reward}")
            
            if reward_threshold is not None and reward[0] >= reward_threshold:
                # use as reward threshold the first component of the reward vector
                threshold_exceeded += 1
                if threshold_exceeded >= THRESHOLD_EXCEEDED_CONSECUTIVELY:
                    if verbose: print(f"Terminating at episode {episode_number} since the reward's first component exceeded for {threshold_exceeded} consecutive times the reward threshold {reward_threshold}")
                    solved = True
            else:
                threshold_exceeded = 0
            
            episode_number+=1
    
        return rewards, avg_rewards, timings, full_infos_dict

    @abstractmethod
    def act(self, state):
        """
        returns the index of the action to perform
        - agent dependent
        """
        print("Agent._act() method without implementation was called! You have to implement custom agent's act method!")

        pass

    def run_episode(self, env : Environment, render : bool = False, title : str = "", verbose:bool=True): #-> tuple[list, int, float, BytesIO]:
        """
        - return total_rewards : list, num_timesteps : int, elapsed_episode : float (in seconds), animated_gif_file : BytesIO
        """

        begin_episode_time = datetime.now()
        state, infos = env.reset()
        done = False
        timestep = 0
        totrew = [0] * self._get_reward_dim()
        img_frames = []
        while not done:
            #start_time = datetime.now()
            action_index = self.act(state)
            action = self.action_space[action_index]
            next_state,reward,terminated, truncated, infos=self._ban_step(env, action)
            # render environment animation

            img_frame = self._render(env, title, timestep, live_rendering=render)
            img_frames.append(img_frame)

            done = bool( terminated or truncated )

            totrew = [sum(foo) for foo in zip(totrew, reward)]

            #self._add_experience(state,action_index,reward,next_state,done)
            state=next_state
            timestep+=1

            #tmng = ( datetime.now() - start_time ).total_seconds()
        end_episode_time = datetime.now()
        elapsed_episode = (end_episode_time - begin_episode_time).total_seconds()

        print_time = end_episode_time.strftime("%H:%M:%S")
        if verbose: print(f"{print_time}\tEpisode\t\ttimesteps:\t{timestep}\tTook\t{elapsed_episode} sec - reward:\t{totrew}\t")

        animated_gif_file = BytesIO()

        img_frames[0].save(animated_gif_file, format="GIF",
               save_all=True, append_images=img_frames[1:], delay=0.1,loop=0)# optimize=False,
        
        return totrew, timestep, elapsed_episode, animated_gif_file



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

    def dump_model_to_file(self, model_filename : str):
        # TODO: implement this method (understand if it is agent-dependent)
        pass

    def load_model_from_file(self, model_filepath : str):
        # TODO: implement this method (understand if it is agent-dependent)
        # refer to: "https://github.com/LoreFiaschi/DeepLearning/test/DQN_LL_BAN - Test.ipynb"
        pass