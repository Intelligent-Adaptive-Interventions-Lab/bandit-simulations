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
