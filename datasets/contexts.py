from typing import List, Dict
from scipy.stats import randint, bernoulli, uniform, norm

import numpy as np


class ContextAllocateData:
    min_val: float
    max_val: float
    type: str
    normalize: bool
    distribution: Dict

    def __init__(self, min_val: float, max_val: float, type: str, normalize: bool, distribution: Dict) -> None:
        self.min_val = min_val
        self.max_val = max_val
        self.type = type
        self.normalize = normalize
        self.distribution = distribution

    def get_rvs(self) -> float:
        distribution_copy = self.distribution.copy()
        dis_type = distribution_copy.pop('type', None)
        
        if dis_type is not None:
            random_val = eval(dis_type).rvs(**distribution_copy)
            if self.type != "CONT":
                random_val = np.floor(random_val)
            random_val = np.clip(random_val, self.min_val, self.max_val)
            
            if self.normalize:
                random_val = (random_val - self.min_val) / (self.max_val - self.min_val)
            
            return round(random_val, 2)
        return None
