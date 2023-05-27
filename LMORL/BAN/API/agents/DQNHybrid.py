from LMORL.BAN.API.agents.agent import Agent
from julia.api import Julia

from julia import Main

from datetime import datetime
import os

DEBUG = False
ELAPSED_THRESHOLD = 1

class DQNHybrid(Agent):
    
    def __init__(self, input_size : int,
                 num_actions : int, action_space, 
                 learning_rate : float, epsilon_decay : float,
                 epsilon_min : float,
                 batch_size : int, 
                 hidden_size : int,
                 ban_size : int,
                 discount_factor : float = 0.99,
                  max_memory_size: int = 100, train_start: int = 100, 
                  use_clipping : bool = False, clipping_tol : float = 1.0) -> None:
        super().__init__(input_size, num_actions, action_space, ban_size, max_memory_size, train_start)

        self.learning_rate = learning_rate
        self.epsilon = 1
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.gamma = discount_factor

        self._julia_eval("""
        (@isdefined DQNAgent) ? nothing : include(\"../../agents/DQN_Gym_BAN_Hybrid.jl\")
        (@isdefined parse_ban_from_array) ? nothing : include(\"../../custom_BAN_utils.jl\")
        """
        )
        # passing parameters to julia env
        self._main.inputsize = self.input_size
        self._main.numactions = self.num_actions
        self._main.actions = self.action_space
        self._main.max_memory = self.max_memory_size
        self._main.learning_rate = self.learning_rate
        self._main.epsilon_decay = self.epsilon_decay
        self._main.epsilon_min = self.epsilon_min
        self._main.batch_size = self.batch_size
        self._main.train_start = self.train_start
        self._main.hidden_size = self.hidden_size
        self._main.gamma = self.gamma
        self._main.use_clipping = use_clipping
        self._main.clipping_tol = abs( clipping_tol )
        #
        self._julia_eval("""agent=DQNAgent(input_size=inputsize,
                    numactions=numactions,actionspace=actions,max_memory=max_memory,learning_rate= learning_rate,epsilon_decay=epsilon_decay,
                    epsilon_min=epsilon_min, batch_size=batch_size,train_start=train_start,hidden_size=hidden_size, use_clipping=use_clipping, clipping_tol=clipping_tol, gamma=gamma)""")
        

    def _episode_end(self):
        self._update_target_model()
        self._update_epsilon()

    def _update_target_model(self):
        #TODO: check if it is specific to DQN or if it is global
        before = datetime.now()
        self._julia_eval("update_target_model!(agent)")
        if DEBUG: 
            elapsed = (datetime.now() - before).total_seconds()
            if elapsed > ELAPSED_THRESHOLD:
                print(f"update target model took {elapsed} seconds")

    def _update_epsilon(self):
        # TODO: check if this is agent dependent or not, 
        # and consider if implementing it directly in python
        before = datetime.now()
        self._julia_eval("update_epsilon!(agent)")
        if DEBUG: 
            elapsed = (datetime.now() - before).total_seconds()
            if elapsed > ELAPSED_THRESHOLD:
                print(f"update_epsilon! took {elapsed} seconds")

    def act(self, state):
        """
        returns the index of the action to perform
        """
        before = datetime.now()
        #self._main.observation = state
        #action_index = self._julia_eval("act!(agent, observation)")
        
        action_index = self._main.act_b(self._main.agent, state)
        
        action_index -= 1  # -1 because julia is 1-based, while python is 0-based
        #if DEBUG: assert action_index in range(len(self.action_space)), f"action index '{action_index}' not in allowed action_space indexing (action space: {self.action_space})"
        if DEBUG: 
            elapsed = (datetime.now() - before).total_seconds()
            if elapsed > ELAPSED_THRESHOLD:
                print(f"act! took {elapsed} seconds | action_index: {action_index}")
        return action_index
        
    def _add_experience(self, state, action_index, reward, next_state, done: bool):
        """
        this agent implementation needs the previous episodes to be accessed by 
        julia methods too, so the most effective way is to store the timesteps each time
        that they happen
        """
        before = datetime.now()
        self._main.add_experience_custom_types_b(self._main.agent, state,action_index+1, reward, next_state, done)

        if DEBUG: 
            elapsed = (datetime.now() - before).total_seconds()
            if elapsed > ELAPSED_THRESHOLD:
                print(f"add_experience! took {elapsed} seconds")

        return
        #before = datetime.now()
        #self._main.state = state
        #self._main.action_index = action_index + 1 #because julia is 1-based, while python is 0-based
        #self._main.reward_list = reward
        #self._main.next_state = next_state
        #self._main.done = done
        #self._main.ban_size = self._main.BAN_SIZE
        #after_passing = datetime.now()
        #cmd_string = """
        #reward_ban = parse_ban_from_array(reward_list, ban_size)
        #action_index=convert(Int32, action_index)
        #add_experience!(agent,state,action_index, reward_ban, next_state, done)
        #"""
        #self._julia_eval(cmd_string)
        ##self._julia_eval("reward_ban = parse_ban_from_array(reward_list, ban_size)")
        ##self._julia_eval("action_index=convert(Int32, action_index)")
        ##self._julia_eval("add_experience!(agent,state,action_index, reward_ban, next_state, done)")
        #if DEBUG: 
        #    elapsed = (datetime.now() - before).total_seconds()
        #    elapsed_passing = (after_passing - before).total_seconds()
        #    if elapsed > ELAPSED_THRESHOLD:
        #        print(f"add_experience! took {elapsed} seconds\telapsed passing params: {elapsed_passing}")
        
        # this agent does not store the timesteps in Python,
        #  since those would not be accessed by this implementation
        #return super()._add_experience(state, action_index, reward, next_state, done)

    def _experience_replay(self):
        # the following check is performed by julia method
        # 'experience_replay!()'
        #if len(self.memory) < self.train_start:
        #    return
        before = datetime.now()
        self._julia_eval("experience_replay!(agent)")
        if DEBUG: 
            elapsed = (datetime.now() - before).total_seconds()
            if elapsed > ELAPSED_THRESHOLD:
                print(f"experience_replay! took {elapsed} seconds")

    def dump_model_to_file(self, model_filename : str, models_path : str = None):
        '''
        Dump model to BSON file
        - model_filename can include .bson file extension 
        - models_path is the absolute path where model is going to be saved
        '''

        complete_saving_path = os.path.join(models_path, model_filename)

        save_cmd = f"""
            weights=Flux.params(agent.model)
            BSON.@save "{complete_saving_path}" weights
        """

        self._julia_eval(save_cmd)

    def load_model_from_file(self, model_filepath : str):
        '''
        Load model from BSON file
        - model_filepath is the absolute path from which model is going to be loaded
        '''

        load_cmd = f"""
            @load "{model_filepath}" weights
            Flux.loadparams!(agent.model,weights)
        """

        self._julia_eval(load_cmd)
        
    def get_epsilon_values_list(self, num_episodes : int):
        """
        returns the list of values assumed by epsilon for the current agent for the specified number of episodes
        - assuming to start from the first episode
        """
        epsilon = self.epsilon
        epsilon_decay = self.epsilon_decay
        epsilon_min = self.epsilon_min
        epsilon_values = []
        for i in range(num_episodes):
            epsilon_values.append(epsilon)
            if epsilon > epsilon_min:
                epsilon = epsilon * epsilon_decay
        return epsilon_values

    def plot_epsilon_values(self, num_episodes : int):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots()

        fig.suptitle(f"Epsilon values for {num_episodes} episodes")

        ax.set(ylabel="epsilon")
        ax.plot(range(num_episodes), self.get_epsilon_values_list(num_episodes))
        

        plt.xlabel("Episodes")
        plt.show()
        return fig



        
        