from typing import Dict, Union

from datasets.bandits import Bandit
from policies.types import PolicyType
from datasets.policies.tscontextual import TSContextualPolicy, TSContextualPolicySimple
from datasets.policies.tspostdiff import TSPostDiffPolicy, TSPostDiffPolicySimple
from datasets.policies.toptwots import TopTwoTSPolicy


class PolicyFactory:
    policy: Union[TopTwoTSPolicy, TSPostDiffPolicy, TSContextualPolicy]

    def __init__(self, policy_configs: Dict, bandit: Bandit) -> None:
        self.policy = None
        if policy_configs["type"] == PolicyType.TSCONTEXTUAL.name:
            self.policy = TSContextualPolicy(policy_configs, bandit)
        elif policy_configs["type"] == PolicyType.TSPOSTDIFF.name:
            self.policy = TSPostDiffPolicy(policy_configs, bandit)
        elif policy_configs["type"] == PolicyType.TOPTWOTS.name:
            self.policy = TopTwoTSPolicy(policy_configs, bandit)
    
    def get_policy(self) -> Union[TopTwoTSPolicy, TSPostDiffPolicy, TSContextualPolicy]:
        return self.policy
    
class PolicyFactorySimple:
    policy: Union[TopTwoTSPolicy, TSPostDiffPolicySimple, TSContextualPolicySimple]

    def __init__(self, policy_configs: Dict, bandit: Bandit) -> None:
        self.policy = None
        if policy_configs["type"] == PolicyType.TSCONTEXTUAL.name:
            self.policy = TSContextualPolicySimple(policy_configs, bandit)
        elif policy_configs["type"] == PolicyType.TSPOSTDIFF.name:
            self.policy = TSPostDiffPolicySimple(policy_configs, bandit)
        elif policy_configs["type"] == PolicyType.TOPTWOTS.name:
            self.policy = TopTwoTSPolicy(policy_configs, bandit)
    
    def get_policy(self) -> Union[TopTwoTSPolicy, TSPostDiffPolicySimple, TSContextualPolicySimple]:
        return self.policy
