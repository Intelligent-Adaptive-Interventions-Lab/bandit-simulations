import pandas as pd

from typing import Dict, Union

from datasets.policies import Policy
from policies.types import PolicyType
from metrics.confidence_interval import estimate_confidence_interval
from metrics.wald_test import perfrom_wald_test


class Evaluator:
    policy: Policy
    simulation_df: pd.DataFrame
    metrics: Dict[str, pd.DataFrame]

    def __init__(self, simulation_df: pd.DataFrame, policy: Policy) -> None:
        self.simulation_df = simulation_df
        self.policy = policy
        self.metrics = {}


class TopTwoTSEvaluator(Evaluator):

    def __init__(self, simulation_df: pd.DataFrame, policy: Policy) -> None:
        super().__init__(simulation_df, policy)
        self.metrics = {
            "wald_test": self._test_wald()
        }
    
    def _test_wald(self) -> pd.DataFrame:
        return perfrom_wald_test(self.simulation_df, self.policy)


class TSPostDiffEvaluator(Evaluator):

    def __init__(self, simulation_df: pd.DataFrame, policy: Policy) -> None:
        super().__init__(simulation_df, policy)
        self.metrics = {
            "wald_test": self._test_wald()
        }
    
    def _test_wald(self) -> pd.DataFrame:
        return perfrom_wald_test(self.simulation_df, self.policy)


class TSContextualEvaluator(Evaluator):

    def __init__(self, simulation_df: pd.DataFrame, policy: Policy) -> None:
        super().__init__(simulation_df, policy)
        regression_formula = self.policy.configs["regression_formula"]
        self.metrics = {
            "confidence_interval": self._evaluate_confidence_interval(regression_formula)
        }
    
    def _evaluate_confidence_interval(self, regression_formula: str) -> pd.DataFrame:
        return estimate_confidence_interval(self.simulation_df, regression_formula)


class EvaluatorFactory:
    evaluator: Union[TSPostDiffEvaluator, TSContextualEvaluator]

    def __init__(self, simulation_df: pd.DataFrame, policy: Policy) -> None:
        self.evaluator = None
        if policy.get_type() == PolicyType.TSCONTEXTUAL.name:
            self.evaluator = TSContextualEvaluator(simulation_df, policy)
        elif policy.get_type() == PolicyType.TSPOSTDIFF.name:
            self.evaluator = TSPostDiffEvaluator(simulation_df, policy)
        elif policy.get_type() == PolicyType.TOPTWOTS.name:
            self.evaluator = TopTwoTSEvaluator(simulation_df, policy)
    
    def get_evaluator(self) -> Union[TSPostDiffEvaluator, TSContextualEvaluator]:
        return self.evaluator
