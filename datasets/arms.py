import pandas as pd

from typing import List, Dict, Union


class ArmData:
    arms: pd.DataFrame
    action_variables: List[str]

    def __init__(self, actions: List[str], arms: List[Dict]) -> None:
        columns = ["name", "count"] + actions + ["success", "failure"]
        self.action_variables = actions
        self.arms = pd.DataFrame(columns=columns)

        init_arms = dict.fromkeys(actions, 0)

        for arm in arms:
            arm_row = init_arms.copy()
            arm_row[arm["action_variable"]] = arm["value"]
            arm_row.update({
                "name": arm["name"],
                "count": 0 if "count" not in arm else arm["count"],
                "success": 1 if "success" not in arm else arm["success"],
                "failure": 1 if "failure" not in arm else arm["failure"]
            })
            self.arms = pd.concat([self.arms, pd.DataFrame.from_records([arm_row])])
        
        self.arms = self.arms.dropna(axis=1)
    
    def get_action_space_from_name(self, arm_name: str) -> Dict:
        arm_row = self.arms.loc[self.arms["name"] == arm_name, self.action_variables]
        return dict(zip(arm_row.columns, list(arm_row.values.flatten())))
    
    def get_from_action_space(self, action_space: Dict, column_name: str) -> Union[str, float]:
        arm_row = self.arms.loc[(self.arms[list(action_space)] == pd.Series(action_space)).all(axis=1)]
        return arm_row[column_name].item()
    
    def get_from_arm_name(self, arm_name: str, column_name: str) -> Union[str, float]:
        arm_row = self.arms.loc[self.arms["name"] == arm_name]
        return arm_row[column_name].item()
    
    def update_from_action_space(self, action_space: Dict, column_name: str, value: Union[str, float]) -> None:
        self.arms.loc[(self.arms[list(action_space)] == pd.Series(action_space)).all(axis=1), column_name] = value

    def update_from_arm_name(self, arm_name: str, column_name: str, value: Union[str, float]) -> None:
        self.arms.loc[self.arms["name"] == arm_name, column_name] = value
