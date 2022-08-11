import pandas as pd
import numpy as np

from typing import List, Dict

from datasets.arms import ArmData
from datasets.contexts import ContextAllocateData
from datasets.rewards import RewardData


class Bandit:

    def __init__(
        self, 
        reward: Dict, 
        arms: Dict, 
        contexts: Dict = None
    ) -> None:
        # Initialize reward dictionary to a RewardData object.
        self.reward = RewardData(reward)

        # Initialize default regression equation terms.
        self.terms = []

        # Initialize arms in two types: 
        #   (1) action_space: a dict with keys of action variables and values of [0, 1].
        #   (2) arm_data: a ArmData object with a dataframe called "arms" with columns of action variables and row of arm names.
        self.action_space = {}
        self.arm_data = self._init_arms(arms) # This method will update "terms".

        # Initialize contextual variables:
        #   contexts_dict: a dict with keys of contextual variables and values of ContextAllocateData.
        if contexts:
            self.contexts_dict = self._init_contexts(contexts) # This method will update "terms".
    
    def _init_arms(self, arms: Dict) -> ArmData:
        for arm in arms:
            arm = dict(arm)
            if arm["action_variable"] is not None and arm["action_variable"] not in self.action_space:
                self.action_space[arm["action_variable"]] = [0, 1]
                self.terms.append(arm["action_variable"])
        
        return ArmData(self.get_actions(), arms)
    
    def _init_contexts(self, contexts: Dict) -> Dict:
        contexts_dict = {}
        for context in contexts:
            context = dict(context)
            context_name = context["name"]
            contexts_dict[context_name] = ContextAllocateData(
                context["min_value"], 
                context["max_value"], 
                context["value_type"], 
                context["normalize"], 
                context["distribution"]
            )
            if context['extra'] is True:
                self.terms.append(context_name)
            if context['interaction'] is True:
                for name in self.get_actions():
                    self.terms.append(f"{name} * {context_name}")
        
        return contexts_dict
    
    def get_actions(self) -> List:
        return list(self.action_space.keys())
    
    def get_contextual_variables(self) -> List:
        return list(self.contexts_dict.keys())
    
    def get_noncont_contextual_variables(self) -> List:
        lst = []
        for context in list(self.contexts_dict.keys()):
            if self.contexts_dict[context].type != "CONT":
                lst.append(context)
        return lst
