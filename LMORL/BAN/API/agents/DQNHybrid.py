from LMORL.BAN.API.agents.agent import Agent
from julia.api import Julia

from julia import Main

from datetime import datetime

DEBUG = True
ELAPSED_THRESHOLD = 1

class DQNHybrid(Agent):
    
    def __init__(self, input_size : int,
                 num_actions : int, action_space, 
                 learning_rate : float, epsilon_decay : float,
                 epsilon_min : float,
                 batch_size : int, 
                 hidden_size : int,
                 ban_size : int,
                  max_memory_size: int = 100, train_start: int = 100) -> None:
        super().__init__(input_size, num_actions, action_space, ban_size, max_memory_size, train_start)

        self.learning_rate = learning_rate
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        self.batch_size = batch_size
        self.hidden_size = hidden_size

        self._julia_eval("include(\"../../agents/DQN_Gym_BAN_Hybrid.jl\")")
        self._julia_eval("include(\"../../custom_BAN_utils.jl\")")
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
        #
        self._julia_eval("""agent=DQNAgent(input_size=inputsize,
                    numactions=numactions,actionspace=actions,max_memory=max_memory,learning_rate= learning_rate,epsilon_decay=epsilon_decay,
                    epsilon_min=epsilon_min, batch_size=batch_size,train_start=train_start,hidden_size=hidden_size)""")
        

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

    def _act(self, state):
        """
        returns the index of the action to perform
        """
        before = datetime.now()
        self._main.observation = state
        action_index = self._julia_eval("act!(agent, observation)")
        # TODO: check if action is correctly returned,
        # and check if action should be in the range (1, num actions) 
        # or (0, num_actions)
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
        self._main.state = state
        self._main.action_index = action_index + 1 #because julia is 1-based, while python is 0-based
        self._main.reward_list = reward
        self._main.next_state = next_state
        self._main.done = done
        self._main.ban_size = self._main.BAN_SIZE
        self._julia_eval("reward_ban = parse_ban_from_array(reward_list, ban_size)")
        self._julia_eval("action_index=convert(Int32, action_index)")
        self._julia_eval("add_experience!(agent,state,action_index, reward_ban, next_state, done)")
        if DEBUG: 
            elapsed = (datetime.now() - before).total_seconds()
            if elapsed > ELAPSED_THRESHOLD:
                print(f"add_experience! took {elapsed} seconds")
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
        
        