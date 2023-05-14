from LMORL.Environment import Environment

import gymnasium as gym
from gymnasium.utils import EzPickle

import numpy as np

from typing import Iterator, Tuple, TypeVar

ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

# https://github.com/Farama-Foundation/MO-Gymnasium/blob/main/mo_gymnasium/utils.py class LinearReward
class NAScalarizationLinearReward(gym.Wrapper, EzPickle):
    """Makes the env return a scalar reward, which is the dot-product between the reward vector and the weight vector."""

    def __init__(self, env: Environment, weight: np.ndarray = None):
        """Makes the env return a scalar reward, which is the dot-product between the reward vector and the weight vector.

        Args:
            env: env to wrap
            weight: weight vector to use in the dot product
        """
        super().__init__(env)
        EzPickle.__init__(self, env, weight)
        if weight is None:
            weight = np.ones(shape=env.reward_space.shape)
        self.set_weight(weight)

    def set_weight(self, weight: np.ndarray):
        """Changes weights for the scalarization.

        Args:
            weight: new weights to set
        Returns: nothing
        """
        #TODO: redefine the set_weight method so that can store NA numbers exploiting Julia BAN library
        assert weight.shape == self.env.reward_space.shape, "Reward weight has different shape than reward vector."
        self.w = weight

    def step(self, action: ActType) -> Tuple[ObsType, float, bool, bool, dict]:
        """Steps in the environment.

        Args:
            action: action to perform
        Returns: obs, scalarized_reward, terminated, truncated, info
        """
        #TODO: redefine the dot product so that can be applied between NA numbers exploiting Julia BAN library
        observation, reward, terminated, truncated, info = self.env.step(action)
        scalar_reward = np.dot(reward, self.w)
        info["vector_reward"] = reward

        return observation, scalar_reward, terminated, truncated, info