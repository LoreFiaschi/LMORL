from LMORL.BAN.API.agents.agent import Agent
from julia.api import Julia

from julia import Main

from datetime import datetime
import os

DEBUG = False
ELAPSED_THRESHOLD = 1

class DQNFullNA(Agent):
    
    def __init__(self, input_size : int,
                 num_actions : int, action_space, 
                 learning_rate : float, epsilon_decay : float,
                 epsilon_min : float,
                 batch_size : int, 
                 hidden_size : int,
                 ban_size : int,
                  max_memory_size: int = 100, train_start: int = 100, 
                  use_clipping : bool = False, clipping_tol : float = 1.0) -> None:
        super().__init__(input_size, num_actions, action_space, ban_size, max_memory_size, train_start)

        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        print("[!] NOT IMPLEMENTED [!]")

    def _episode_end(self):
        print("[!] NOT IMPLEMENTED [!]")
        pass

    def _update_target_model(self):
        print("[!] NOT IMPLEMENTED [!]")
        pass

    def _update_epsilon(self):
        print("[!] NOT IMPLEMENTED [!]")
        pass

    def act(self, state):
        print("[!] NOT IMPLEMENTED [!]")
        pass
        
    def _add_experience(self, state, action_index, reward, next_state, done: bool):
        print("[!] NOT IMPLEMENTED [!]")
        pass

    def _experience_replay(self):
        print("[!] NOT IMPLEMENTED [!]")
        pass

    def dump_model_to_file(self, model_filename : str, models_path : str = None):
        print("[!] NOT IMPLEMENTED [!]")
        pass

    def load_model_from_file(self, model_filepath : str):
        print("[!] NOT IMPLEMENTED [!]")
        pass
        

        



        
        