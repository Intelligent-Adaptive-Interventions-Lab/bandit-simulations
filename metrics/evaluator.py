import pandas as pd

from typing import Dict, Union

from policies.types import PolicyType
from metrics.confidence_interval import estimate_confidence_interval


class Evaluator:
    simulation_df: pd.DataFrame
    metrics: Dict[str, pd.DataFrame]

    def __init__(self, simulation_df: pd.DataFrame) -> None:
        self.simulation_df = simulation_df
        self.metrics = {}


class TSPostDiffEvaluator(Evaluator):

    def __init__(self, simulation_df: pd.DataFrame) -> None:
        super().__init__(simulation_df)


class TSContextualEvaluator(Evaluator):

    def __init__(self, simulation_df: pd.DataFrame) -> None:
        super().__init__(simulation_df)
        self.metrics = {
            "confidence_interval": self._evaluate_confidence_interval()
        }
    
    def _evaluate_confidence_interval(self) -> pd.DataFrame:
        return estimate_confidence_interval(self.simulation_df)


class EvaluatorFactory:
    evaluator: Union[TSPostDiffEvaluator, TSContextualEvaluator]

    def __init__(self, policy_configs: Dict, simulation_df: pd.DataFrame) -> None:
        self.evaluator = None
        if policy_configs["type"] == PolicyType.TSCONTEXTUAL.name:
            self.evaluator = TSContextualEvaluator(simulation_df)
        elif policy_configs["type"] == PolicyType.TSPOSTDIFF.name:
            self.evaluator = TSPostDiffEvaluator(simulation_df)
    
    def get_evaluator(self) -> Union[TSPostDiffEvaluator, TSContextualEvaluator]:
        return self.evaluator
