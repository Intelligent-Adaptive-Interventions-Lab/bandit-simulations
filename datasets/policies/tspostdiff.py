import pandas as pd
import numpy as np

from typing import Dict

from datasets.bandits import Bandit
from datasets.policies.policy import Policy
from policies.tspostdiff.parameters import TSPostDiffParameter
from policies.tspostdiff.ts_postdiff import thompson_sampling_postdiff


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
        new_learner_df.update(assignment_data)
        new_learner_df.update(best_action)

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
