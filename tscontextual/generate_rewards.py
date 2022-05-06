import pandas as pd
import numpy as np

from typing import List, Dict
from tscontextual.ts_contextual import calculate_outcome


def get_reward(assignment_df: pd.DataFrame, true_estimate: str, true_coef_mean: List[float], include_intercept: int, reward: Dict) -> pd.DataFrame:
    formula = true_estimate.strip()
    vars_list = list(map(str.strip, formula.split('~')[1].strip().split('+')))
    reward_name = formula.split('~')[0].strip()

    assert reward_name == reward["name"]

    terms = []
    for var in vars_list:
        if "*" in var:
            interacting_vars = list(map(str.strip, var.split('*')))
            terms += interacting_vars
        else:
            terms.append(var)
    terms = list(set(terms))

    error_var = (reward["max_value"] - reward["min_value"]) / 3

    for index, row in assignment_df.iterrows():
        row_terms = row[terms]
        error = np.random.normal(0, error_var, 1)[0]
        true_reward = calculate_outcome(row_terms.to_dict(), np.array(true_coef_mean), include_intercept, true_estimate) + error
        if reward["value_type"] != "CONT":
            true_reward = np.floor(true_reward)
        row[reward_name] = np.clip(true_reward, reward["min_value"], reward["max_value"])

    return assignment_df
