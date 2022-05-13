import pandas as pd
import numpy as np

from typing import List, Dict, Union

from datasets.bandits import Bandit
from policies.types import PolicyType
from policies.tscontextual.parameters import TSContextualParams
from policies.tscontextual.ts_contextual import thompson_sampling_contextual
from policies.tscontextual.ts_contextual import calculate_outcome


class Policy:
    
    def __init__(self, policy_configs: Dict, bandit: Bandit) -> None:
        self.configs = policy_configs
        self.bandit = bandit
    
    def get_type(self) -> str:
        return self.configs["type"]


class TSPostDiffPolicy(Policy):
    
    def __init__(self, policy_configs: Dict, bandit: Bandit) -> None:
        super().__init__(policy_configs, bandit)


class TSContextualPolicy(Policy):
    
    def __init__(self, policy_configs: Dict, bandit: Bandit) -> None:
        super().__init__(policy_configs, bandit)

        # Get regression equation terms.
        terms = self.bandit.terms

        # Get the length of regression equation terms.
        length = len(terms) + 1 if self.configs["include_intercept"] == 1 else len(terms)

        # Initialize some parameters in ts contextual policy.
        self.configs["coef_cov"] = np.eye(length, dtype=float).tolist()
        self.configs["coef_mean"] = np.zeros(length, dtype=float).tolist()
        self.configs["action_space"] = self.bandit.action_space
        self.configs["contextual_variables"] = self.bandit.get_contextual_variables()
        self.configs["outcome_variable"] = self.bandit.reward.name

        # Initialize regression equation if this is not provided.
        if "regression_formula" not in self.configs or self.configs["regression_formula"] is None:
            self.configs["regression_formula"] = "{} ~ {}".format(self.bandit.reward.name, ' + '.join(terms))
        
        print("regression_formula: {}".format(self.configs["regression_formula"]))

        # Initialize parameters.
        self.params = TSContextualParams(self.configs)

        # Initialize columns of simulation dataframe.
        self.columns = ["learner", "arm", self.bandit.reward.name] + \
            self.bandit.get_actions() + self.bandit.get_contextual_variables() + \
            ["coef_cov", "coef_mean", "variance_a", "variance_b", "precesion_draw", "coef_draw", "update_batch"]
        
        # Initialize the indicator of update batch.
        self.update_count = 0
    
    def run(self, new_learner: str) -> pd.DataFrame:
        new_learner_df = {}
        new_learner_df["learner"] = new_learner

        # Get best action and datapoints (e.g. assigned arm, generated contexts) for the new learner.
        best_action, assignment_data = thompson_sampling_contextual(self.params, self.bandit.contexts_dict)

        # Record the arm name from the best action and action dataframe.
        arm_df = self.bandit.get_arm_df()
        best_arm_row = arm_df.loc[(arm_df[list(best_action)] == pd.Series(best_action)).all(axis=1)]
        new_learner_df["arm"] = str(best_arm_row["name"][0])

        # Merge to a complete datapoints collection.
        new_learner_df = new_learner_df | assignment_data | best_action

        return new_learner_df
    
    def get_reward(self, new_learner_df: pd.DataFrame) -> pd.DataFrame:
        true_estimate = self.params.parameters["true_estimate"]
        true_coef_mean = self.params.parameters["true_coef_mean"]
        include_intercept = self.params.parameters["include_intercept"]
        reward = self.bandit.reward

        # Get variable names from true_estimate.
        formula = true_estimate.strip()
        vars_list = list(map(str.strip, formula.split('~')[1].strip().split('+')))
        reward_name = formula.split('~')[0].strip()

        assert reward_name == reward.name

        # Get regression equation terms from true_estimate.
        terms = []
        for var in vars_list:
            if "*" in var:
                interacting_vars = list(map(str.strip, var.split('*')))
                terms += interacting_vars
            else:
                terms.append(var)
        terms = list(set(terms))

        # Get scale of error.
        err_scale = (reward.max_value - reward.min_value) / 3

        # Update reward for the new learner dataframe.
        for index, row in new_learner_df.iterrows():
            row_terms = row[terms]
            error = np.random.normal(0, err_scale, 1)[0]
            true_reward = calculate_outcome(row_terms.to_dict(), np.array(true_coef_mean), include_intercept, true_estimate) + error
            row[reward_name] = reward.get_reward(true_reward)

        return new_learner_df
    
    def update_params(self, assignment_df: pd.DataFrame) -> pd.DataFrame:
        # Record update batch indicator.
        assignment_df["update_batch"] = self.update_count

        # Update parameters.
        value_col = [self.bandit.reward.name] + self.bandit.get_actions() + self.bandit.get_contextual_variables()
        self.params.update_params(assignment_df[value_col], self.bandit.reward.name)

        # Update the indicator.
        self.update_count += 1

        return assignment_df


class PolicyFactory:
    policy: Union[TSPostDiffPolicy, TSContextualPolicy]

    def __init__(self, policy_configs: Dict, bandit: Bandit) -> None:
        self.policy = None
        if policy_configs["type"] == PolicyType.TSCONTEXTUAL.name:
            self.policy = TSContextualPolicy(policy_configs, bandit)
        elif policy_configs["type"] == PolicyType.TSPOSTDIFF.name:
            self.policy = TSPostDiffPolicy(policy_configs, bandit)
    
    def get_policy(self) -> Union[TSPostDiffPolicy, TSContextualPolicy]:
        return self.policy
