import pandas as pd
import numpy as np

from typing import List
from tscontextual.ts_contextual import calculate_outcome


def get_reward(assignment_df: pd.DataFrame, true_estimate: str, true_coef_mean: List[float], include_intercept: int) -> pd.DataFrame:
    formula = true_estimate.strip()
    vars_list = list(map(str.strip, formula.split('~')[1].strip().split('+')))
    reward_name = formula.split('~')[0].strip()

    terms = []
    for var in vars_list:
        if "*" in var:
            interacting_vars = list(map(str.strip, var.split('*')))
            terms += interacting_vars
        else:
            terms.append(var)
    terms = list(set(terms))

    for index, row in assignment_df.iterrows():
        row_terms = row[terms]
        true_reward = calculate_outcome(row_terms.to_dict(), np.array(true_coef_mean), include_intercept, true_estimate)
        row[reward_name] = true_reward

    return assignment_df
