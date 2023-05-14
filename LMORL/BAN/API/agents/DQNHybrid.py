from agent import Agent

class DQNHybrid(Agent):
    

    def _episode_end(self):
        self._update_target_model()
        self._update_epsilon()

    def _update_target_model(self):
        #TODO: check if it is specific to DQN or if it is global
        pass

    def _update_epsilon(self):
        pass
