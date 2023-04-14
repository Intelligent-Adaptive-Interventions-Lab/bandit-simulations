import pandas as pd

from typing import List, Dict, Union

from datasets.policies.policy import Policy
from policies.types import PolicyType
from metrics.confidence_interval import estimate_confidence_interval
from metrics.wald_test import perfrom_wald_test
from metrics.arm_summary import arm_summary
from metrics.context_summary import context_summary
from metrics.power_fpr import power_fpr


class Evaluator:
    policy: Policy
    simulation_dfs: List[pd.DataFrame]
    metrics: Dict[str, pd.DataFrame]

    def __init__(self, simulation_dfs: List[pd.DataFrame], policy: Policy) -> None:
        self.simulation_dfs = simulation_dfs
        self.policy = policy
        self.metrics = {}


class TopTwoTSEvaluator(Evaluator):

    def __init__(self, simulation_dfs: List[pd.DataFrame], policy: Policy) -> None:
        super().__init__(simulation_dfs, policy)
        self.metrics = {
            "wald_test": self._test_wald()
        }
    
    def _test_wald(self) -> pd.DataFrame:
        return perfrom_wald_test(self.simulation_dfs, self.policy)
    
    def _arm_summary(self, reward: str) -> pd.DataFrame:
        return arm_summary(self.simulation_dfs, reward)


class TSPostDiffEvaluator(Evaluator):

    def __init__(self, simulation_dfs: List[pd.DataFrame], policy: Policy) -> None:
        super().__init__(simulation_dfs, policy)
        self.metrics = {
            "wald_test": self._test_wald()
        }
    
    def _test_wald(self) -> pd.DataFrame:
        return perfrom_wald_test(self.simulation_df, self.policy)
    
    def _arm_summary(self, reward: str) -> pd.DataFrame:
        return arm_summary(self.simulation_df, reward)


class TSContextualEvaluator(Evaluator):

    def __init__(self, simulation_dfs: List[pd.DataFrame], policy: Policy) -> None:
        super().__init__(simulation_dfs, policy)
        regression_formula = self.policy.configs["regression_formula"]
        reward = self.policy.bandit.reward.name
        noncont_contexts = self.policy.bandit.get_noncont_contextual_variables()
        self.metrics = {
            "confidence_interval": self._evaluate_confidence_interval(regression_formula),
            "arm_summary": self._arm_summary(reward)
        }
        for context in noncont_contexts:
            self.metrics["{}_summary".format(context)] = self._context_summary(reward, context)
        
        self.metrics["wald_test"] = self._test_wald()
        self.metrics["power_fpr"] = self._compute_power_fpr(reward)

    def _evaluate_confidence_interval(self, regression_formula: str) -> pd.DataFrame:
        return estimate_confidence_interval(self.simulation_dfs, regression_formula)

    def _arm_summary(self, reward: str) -> pd.DataFrame:
        return arm_summary(self.simulation_dfs, reward)
    
    def _context_summary(self, reward: str, context: str) -> pd.DataFrame:
        return context_summary(self.simulation_dfs, reward, context)

    def _test_wald(self) -> pd.DataFrame:
        return perfrom_wald_test(self.simulation_dfs, self.policy)
    
    def _compute_power_fpr(self, reward: str) -> pd.DataFrame:
        return power_fpr(self.simulation_dfs, self.metrics["wald_test"], reward)


class EvaluatorFactory:
    evaluator: Union[TopTwoTSEvaluator, TSPostDiffEvaluator, TSContextualEvaluator]

    def __init__(self, simulation_dfs: List[pd.DataFrame], policy: Policy) -> None:
        self.evaluator = None
        if policy.get_type() == PolicyType.TSCONTEXTUAL.name:
            self.evaluator = TSContextualEvaluator(simulation_dfs, policy)
        elif policy.get_type() == PolicyType.TSPOSTDIFF.name:
            self.evaluator = TSPostDiffEvaluator(simulation_dfs, policy)
        elif policy.get_type() == PolicyType.TOPTWOTS.name:
            self.evaluator = TopTwoTSEvaluator(simulation_dfs, policy)
    
    def get_evaluator(self) -> Union[TSPostDiffEvaluator, TSContextualEvaluator]:
        return self.evaluator
