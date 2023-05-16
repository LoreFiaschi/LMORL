class Timestep:
    """
    this class is julia lib Replay class
    """
    def __init__(self, state, action, reward, next_state, done) -> None:
        self.state = state
        self.action = action
        self.reward = reward
        self.next_state = next_state
        self.done = done
    
    #def __init__(self, **kwargs):
    #    self.__dict__.update(kwargs)