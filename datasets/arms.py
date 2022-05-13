import pandas as pd

from typing import List, Dict


class ArmData:
    arms: pd.DataFrame

    def __init__(self, actions: List[str], arms: List[Dict]) -> None:
        columns = ["name"] + actions
        self.arms = pd.DataFrame(columns=columns)

        init_arms = dict.fromkeys(actions, 0)

        for arm in arms:
            arm_row = init_arms.copy()
            arm_row[arm["action_variable"]] = arm["value"]
            arm_row["name"] = arm["name"]
            self.arms = pd.concat([self.arms, pd.DataFrame.from_records([arm_row])])
    