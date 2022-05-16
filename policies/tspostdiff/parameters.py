import pandas as pd
from typing import Dict

from policies.types import PolicyParameter


class TSPostDiffParameter(PolicyParameter):
    parameters: Dict

    def __init__(self, initParams: Dict) -> None:
        super().__init__(initParams)
    
    def update_params(self, arm_name: str, **kwargs) -> None:
        for key, value in kwargs.items(): 
            self.parameters["priors"][arm_name][key] = value
