import numpy as np

from typing import Dict


class RewardData:
    name: str
    min_value: float
    max_value: float
    value_type: str
    is_normalize: bool

    def __init__(self, reward: Dict) -> None:
        self.name = reward['name']
        self.min_value = reward['min_value']
        self.max_value = reward['max_value']
        self.value_type = reward['value_type']
        if 'normalize' not in reward:
            self.is_normalize = False
        else:
            self.is_normalize = reward['normalize']

    def get_raw_reward(self, reward: float) -> float:
        """ Return the scaled-up reward.
        
        params:
            - reward: a normalized float reward in [0, 1].

        returns:
            - raw_reward: a scaled-up float reward in [min_value, max_value] with given value_type.
        """
        raw_reward = reward * (self.max_value - self.min_value) + self.min_value

        if self.value_type != 'CONT':
            return np.floor(raw_reward)
        return raw_reward
        
    def get_scale_reward(self, reward: float) -> float:
        """ Return the normalized reward.

        params:
            - reward: a scaled-up float reward in [min_value, max_value] with given value_type.
        
        returns:
            - normalized_reward: a normalized float reward in [0, 1].
        """
        normalized_reward = (reward - self.min_value) / (self.max_value - self.min_value)

        return normalized_reward
    
    def get_reward(self, raw_reward: float) -> float:
        """ Return the reward value given the normalization option in configs file. """
        if self.value_type != "CONT":
            raw_reward = np.around(raw_reward)
        raw_reward = np.clip(raw_reward, self.min_value, self.max_value)

        if self.is_normalize and (self.min_value != 0.0 or self.max_value != 1.0):
            return self.get_scale_reward(raw_reward)
        return raw_reward
