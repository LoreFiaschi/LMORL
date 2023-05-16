from gymnasium import Env
#from mo_gymnasium.utils

class Environment(Env):
    """
    this class extends MO Gymnasium Env class, it exposes the same methods to define a custom environment
    - In addition it let specify NA weights for the multi objective reward vector
    
    """
    def __init__(self) -> None:
        pass

    def get_reward_dim(self) -> int:
        pass

    def get_scalar_reward(self):
        """
        The idea is that you can always get the scalarized reward.
        1. In case of single objective reward, it is simply returned.
        2. In case of MultiObjective reward array: it is obtained by multiplying the MO reward array by its weights (that can be Non Archimedean)).
        """
        pass
