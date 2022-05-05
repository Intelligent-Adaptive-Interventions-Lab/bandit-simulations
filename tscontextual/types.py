from statistics import variance
from typing import Dict, List

class TSContextualParams:
    parameters: Dict
    
    def __init__(self, initParams: Dict) -> None:
        self.parameters = initParams
