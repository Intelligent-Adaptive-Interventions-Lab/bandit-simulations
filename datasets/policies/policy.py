from typing import List, Dict

from datasets.bandits import Bandit
from policies.types import PolicyType


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
