import pandas as pd
import numpy as np

from typing import List, Dict, Union

from datasets.bandits import Bandit
from policies.types import PolicyType
from policies.toptwots.parameters import TopTwoTSParameter
from policies.toptwots.toptwo_ts import top_two_thompson_sampling
from policies.tspostdiff.parameters import TSPostDiffParameter
from policies.tspostdiff.ts_postdiff import thompson_sampling_postdiff
from policies.tscontextual.parameters import TSContextualParameter
from policies.tscontextual.ts_contextual import thompson_sampling_contextual
from policies.tscontextual.ts_contextual import calculate_outcome


class Policy:
    configs: Dict
    bandit: Bandit
    reward_generate_plan: List
    
    def __init__(self, policy_configs: Dict, bandit: Bandit) -> None:
        self.configs = policy_configs
        self.bandit = bandit
        self.reward_generate_plan = []
    
    def get_type(self) -> str:
        return self.configs["type"]
    
    def get_name(self) -> str:
        return self.configs["name"]
    
    def is_unique_reward(self) -> bool:
        return self.configs["unique_reward"]
    
    def is_unique_contexts(self) -> bool:
        if self.get_type() != PolicyType.TSCONTEXTUAL.name:
            return False
        return self.configs["unique_contexts"]
    
    def get_learner_column_name(self) -> str:
        return "learner"
    
    def get_burn_in_column_name(self) -> str:
        return "uniform_threshold"
    
    def get_arm_column_name(self) -> str:
        return "arm"
    
    def get_reward_column_name(self) -> str:
        return self.bandit.reward.name
    
    def get_udpate_batch_column_name(self) -> str:
        return "update_batch"
    
    def get_reward_generate_plan(self) -> str:
        parameter_subset = {key: self.configs[key] for key in self.reward_generate_plan}
        return str(parameter_subset)


class TopTwoTSPolicy(Policy):

    def __init__(self, policy_configs: Dict, bandit: Bandit) -> None:
        super().__init__(policy_configs, bandit)

        # Initialize a dict of versions to successes and failures e.g.:
	    # {arm1: {success: 1, failure: 1}, arm2: {success: 1, failure: 1}, ...}
        version_dict = {}
        columns = []
        for index, row in self.bandit.arm_data.arms.iterrows():
            version_dict[row["name"]] = {
                "success": row["success"],
                "failure": row["failure"]
            }
            columns.append("{} Success".format(row["name"]).replace(" ", "_").lower())
            columns.append("{} Failure".format(row["name"]).replace(" ", "_").lower())
            columns.append("{} Count".format(row["name"]).replace(" ", "_").lower())
        
        # Initialize some parameters in top-two ts policy.
        self.configs["priors"] = version_dict

         # Initialize parameters.
        self.params = TopTwoTSParameter(self.configs)

        # Initialize columns of simulation dataframe.
        self.columns = [
            self.get_learner_column_name(), 
            self.get_burn_in_column_name(), 
            self.get_arm_column_name(), 
            self.get_reward_column_name()
        ] + self.bandit.get_actions() + columns + [self.get_udpate_batch_column_name()]
        
        # Initialize the indicator of update batch.
        self.update_count = 0

        # Initialize the parameter names which are used for generating rewards.
        self.reward_generate_plan = [self.get_burn_in_column_name(), "true_arm_probs"]
    
    def run(self, new_learner: str, new_learner_df: Dict={}) -> pd.DataFrame:
        new_learner_df[self.get_learner_column_name()] = new_learner
        new_learner_df[self.get_burn_in_column_name()] = int(self.params.is_burn_in())

        # Get best action and datapoints (e.g. assigned arm, generated contexts) for the new learner.
        best_action_name, assignment_data = top_two_thompson_sampling(self.params)

        # Record the arm name from the best action and action dataframe.
        new_learner_df[self.get_arm_column_name()] = best_action_name

        # Update arm count to arm dataframe.
        arm_count = self.bandit.arm_data.get_from_arm_name(best_action_name, "count")
        self.bandit.arm_data.update_from_arm_name(best_action_name, "count", arm_count + 1)

        # Add arm count to each arm column.
        for index, row in self.bandit.arm_data.arms.iterrows():
            action_name = row["name"]
            assignment_data[f"{action_name} Count".replace(" ", "_").lower()] = row["count"]

        # Get the action space for the best arm.
        best_action = self.bandit.arm_data.get_action_space_from_name(new_learner_df[self.get_arm_column_name()])

        # Merge to a complete datapoints collection.
        new_learner_df = new_learner_df | assignment_data | best_action

        return new_learner_df

    def get_reward(self, new_learner_df: pd.DataFrame) -> pd.DataFrame:
        true_arm_probs = dict(self.params.parameters["true_arm_probs"])
        reward = self.bandit.reward

        # Update reward for the new learner dataframe.
        for index, row in new_learner_df.iterrows():
            arm_name = row[self.get_arm_column_name()]
            true_reward = np.random.binomial(reward.max_value - reward.min_value, true_arm_probs[arm_name]) + reward.min_value
            row[reward.name] = reward.get_reward(true_reward)
        
        return new_learner_df
    
    def update_params(self, assignment_df: pd.DataFrame) -> pd.DataFrame:
        # Record update batch indicator.
        assignment_df[self.get_udpate_batch_column_name()] = self.update_count

        reward = self.bandit.reward
        arm_names = self.bandit.arm_data.arms["name"].tolist()

        for arm_name in arm_names:
            # Get reward sum and reward count for each arm.
            sum_rewards = float(sum(assignment_df[assignment_df[self.get_arm_column_name()] == arm_name][reward.name]))
            count_rewards = float(len(assignment_df[assignment_df[self.get_arm_column_name()] == arm_name].index))

            # Scale-up reward sum if reward is normalized.
            if reward.is_normalize:
                sum_rewards = sum_rewards * (reward.max_value - reward.min_value) + count_rewards * reward.min_value
            
            # Update success (e.g. alpha) for each arm.
            success_update = (sum_rewards - count_rewards * reward.min_value) / (reward.max_value - reward.min_value)
            success_update = self.params.parameters["priors"][arm_name]["success"] + success_update

            # Update failure (e.g. beta) for each arm.
            failure_update = (count_rewards * reward.max_value - sum_rewards) / (reward.max_value - reward.min_value)
            failure_update = self.params.parameters["priors"][arm_name]["failure"] + failure_update

            # Update success and failure to arm dataframe and parameter's priors. 
            self.bandit.arm_data.update_from_arm_name(arm_name, "success", success_update)
            self.bandit.arm_data.update_from_arm_name(arm_name, "failure", failure_update)
            self.params.update_params(arm_name, success=success_update, failure=failure_update)
        
        # Update the indicator.
        self.update_count += 1

        return assignment_df


class TSPostDiffPolicy(Policy):
    
    def __init__(self, policy_configs: Dict, bandit: Bandit) -> None:
        super().__init__(policy_configs, bandit)

        # Initialize a dict of versions to successes and failures e.g.:
	    # {arm1: {success: 1, failure: 1}, arm2: {success: 1, failure: 1}, ...}
        version_dict = {}
        columns = []
        for index, row in self.bandit.arm_data.arms.iterrows():
            version_dict[row["name"]] = {
                "success": row["success"],
                "failure": row["failure"]
            }
            columns.append("{} Success".format(row["name"]).replace(" ", "_").lower())
            columns.append("{} Failure".format(row["name"]).replace(" ", "_").lower())
            columns.append("{} Count".format(row["name"]).replace(" ", "_").lower())
        
        # Initialize some parameters in ts postdiff policy.
        self.configs["priors"] = version_dict

        # Initialize parameters.
        self.params = TSPostDiffParameter(self.configs)

        # Initialize columns of simulation dataframe.
        self.columns = [
            self.get_learner_column_name(), 
            self.get_burn_in_column_name(), 
            self.get_arm_column_name(), 
            self.get_reward_column_name()
        ] + self.bandit.get_actions() + columns + [self.get_udpate_batch_column_name()]
        
        # Initialize the indicator of update batch.
        self.update_count = 0

        # Initialize the parameter names which are used for generating rewards.
        self.reward_generate_plan = [self.get_burn_in_column_name(), "true_arm_probs"]
    
    def run(self, new_learner: str, new_learner_df: Dict={}) -> pd.DataFrame:
        new_learner_df[self.get_learner_column_name()] = new_learner
        new_learner_df[self.get_burn_in_column_name()] = int(self.params.is_burn_in())

        # Get best action and datapoints (e.g. assigned arm, generated contexts) for the new learner.
        best_action_name, assignment_data = thompson_sampling_postdiff(self.params)

        # Record the arm name from the best action and action dataframe.
        new_learner_df[self.get_arm_column_name()] = best_action_name

        # Update arm count to arm dataframe.
        arm_count = self.bandit.arm_data.get_from_arm_name(best_action_name, "count")
        self.bandit.arm_data.update_from_arm_name(best_action_name, "count", arm_count + 1)

        # Add arm count to each arm column.
        for index, row in self.bandit.arm_data.arms.iterrows():
            action_name = row["name"]
            assignment_data[f"{action_name} Count".replace(" ", "_").lower()] = row["count"]

        # Get the action space for the best arm.
        best_action = self.bandit.arm_data.get_action_space_from_name(new_learner_df[self.get_arm_column_name()])

        # Merge to a complete datapoints collection.
        new_learner_df = new_learner_df | assignment_data | best_action

        return new_learner_df
    
    def get_reward(self, new_learner_df: pd.DataFrame) -> pd.DataFrame:
        true_arm_probs = dict(self.params.parameters["true_arm_probs"])
        reward = self.bandit.reward

        # Update reward for the new learner dataframe.
        for index, row in new_learner_df.iterrows():
            arm_name = row[self.get_arm_column_name()]
            true_reward = np.random.binomial(reward.max_value - reward.min_value, true_arm_probs[arm_name]) + reward.min_value
            row[reward.name] = reward.get_reward(true_reward)
        
        return new_learner_df
    
    def update_params(self, assignment_df: pd.DataFrame) -> pd.DataFrame:
        # Record update batch indicator.
        assignment_df[self.get_udpate_batch_column_name()] = self.update_count

        reward = self.bandit.reward
        arm_names = self.bandit.arm_data.arms["name"].tolist()

        for arm_name in arm_names:
            # Get reward sum and reward count for each arm.
            sum_rewards = float(sum(assignment_df[assignment_df[self.get_arm_column_name()] == arm_name][reward.name]))
            count_rewards = float(len(assignment_df[assignment_df[self.get_arm_column_name()] == arm_name].index))

            # Scale-up reward sum if reward is normalized.
            if reward.is_normalize:
                sum_rewards = sum_rewards * (reward.max_value - reward.min_value) + count_rewards * reward.min_value
            
            # Update success (e.g. alpha) for each arm.
            success_update = (sum_rewards - count_rewards * reward.min_value) / (reward.max_value - reward.min_value)
            success_update = self.params.parameters["priors"][arm_name]["success"] + success_update

            # Update failure (e.g. beta) for each arm.
            failure_update = (count_rewards * reward.max_value - sum_rewards) / (reward.max_value - reward.min_value)
            failure_update = self.params.parameters["priors"][arm_name]["failure"] + failure_update

            # Update success and failure to arm dataframe and parameter's priors. 
            self.bandit.arm_data.update_from_arm_name(arm_name, "success", success_update)
            self.bandit.arm_data.update_from_arm_name(arm_name, "failure", failure_update)
            self.params.update_params(arm_name, success=success_update, failure=failure_update)
        
        # Update the indicator.
        self.update_count += 1

        return assignment_df


class TSContextualPolicy(Policy):
    
    def __init__(self, policy_configs: Dict, bandit: Bandit) -> None:
        super().__init__(policy_configs, bandit)

        columns = []
        for index, row in self.bandit.arm_data.arms.iterrows():
            columns.append("{} Count".format(row["name"]).replace(" ", "_").lower())

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
        
        print("Policy {} regression_formula: {}".format(self.get_name(), self.configs["regression_formula"]))

        # Initialize parameters.
        self.params = TSContextualParameter(self.configs)

        # Initialize columns of simulation dataframe.
        self.columns = [
            self.get_learner_column_name(), 
            self.get_burn_in_column_name(), 
            self.get_arm_column_name(), 
            self.get_reward_column_name()
        ] + self.bandit.get_actions() + self.bandit.get_contextual_variables() + columns + \
        self.get_parameter_column_names() + [self.get_udpate_batch_column_name()]
        
        # Initialize the indicator of update batch.
        self.update_count = 0

        # Initialize the parameter names which are used for generating rewards.
        self.reward_generate_plan = [self.get_burn_in_column_name(), "true_estimate", "true_coef_mean", "include_intercept"]
    
    def get_contexts(self, new_learner_dict: Dict={}) -> Dict:
        for var in self.bandit.contexts_dict:
            new_learner_dict[var] = self.bandit.contexts_dict[var].get_rvs()
        
        return new_learner_dict

    def get_parameter_column_names(self) -> List[str]:
        return ["coef_cov", "coef_mean", "variance_a", "variance_b", "precesion_draw", "coef_draw"]
    
    def run(self, new_learner: str, new_learner_df: Dict={}) -> pd.DataFrame:
        # Generate contexts
        if not new_learner_df:
            new_learner_df = self.get_contexts(new_learner_df)
        context_dict = {key: new_learner_df[key] for key in self.bandit.get_contextual_variables()}

        new_learner_df[self.get_learner_column_name()] = new_learner
        new_learner_df[self.get_burn_in_column_name()] = int(self.params.is_burn_in())

        # Get best action and datapoints (e.g. assigned arm, generated contexts) for the new learner.
        best_action, assignment_data = thompson_sampling_contextual(self.params, context_dict)

        # Record the arm name from the best action and action dataframe.
        new_learner_df[self.get_arm_column_name()] = self.bandit.arm_data.get_from_action_space(best_action, "name")
        arm_count = self.bandit.arm_data.get_from_action_space(best_action, "count")
        self.bandit.arm_data.update_from_action_space(best_action, "count", arm_count + 1)
        
        for index, row in self.bandit.arm_data.arms.iterrows():
            action_name = row["name"]
            assignment_data[f"{action_name} Count".replace(" ", "_").lower()] = row["count"]

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

        # Update reward for the new learner dataframe.
        for index, row in new_learner_df.iterrows():
            row_terms = row[terms]
            raw_reward = calculate_outcome(row_terms.to_dict(), np.array(true_coef_mean), include_intercept, true_estimate)
            if reward.value_type == "BIN":
                true_reward = np.random.binomial(1, raw_reward, 1)[0]
            else:
                # Get scale of error.
                err_scale = (reward.max_value - reward.min_value) / 6
                error = np.random.normal(0, err_scale, 1)[0]
                true_reward = raw_reward + error
            row[reward_name] = reward.get_reward(true_reward)

        return new_learner_df
    
    def update_params(self, assignment_df: pd.DataFrame) -> pd.DataFrame:
        # Record update batch indicator.
        assignment_df[self.get_udpate_batch_column_name()] = self.update_count

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
        elif policy_configs["type"] == PolicyType.TOPTWOTS.name:
            self.policy = TopTwoTSPolicy(policy_configs, bandit)
    
    def get_policy(self) -> Union[TSPostDiffPolicy, TSContextualPolicy]:
        return self.policy
