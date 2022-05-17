from enum import Enum

from typing import Dict


class PolicyType(Enum):
    TSCONTEXTUAL = "TSCONTEXTUAL"
    TSPOSTDIFF = "TSPOSTDIFF"
    TOPTWOTS = "TOPTWOTS"


class PolicyParameter:
    parameters: Dict

    def __init__(self, params: Dict) -> None:
        self.parameters = params

        if "uniform_threshold" not in self.parameters:
            self.parameters["uniform_threshold"] = 0
        
        if "batch_size" not in self.parameters:
            self.parameters["batch_size"] = 1
    
    def is_burn_in(self) -> bool:
        return self.parameters["uniform_threshold"] != 0
