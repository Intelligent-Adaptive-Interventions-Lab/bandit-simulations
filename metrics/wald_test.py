import pandas as pd
import numpy as np
from scipy.stats import norm

from typing import List, Tuple, Union

from datasets.policies import TopTwoTSPolicy, TSPostDiffPolicy

np.seterr(divide='ignore',invalid='ignore')


def wald_test_statistics(
    success_a: Union[np.ndarray, float], 
    success_b: Union[np.ndarray, float], 
    num_a: Union[np.ndarray, float], 
    num_b: Union[np.ndarray, float]
) -> Tuple[Union[np.ndarray, float], Union[np.ndarray, float]]:
    num_a = np.asarray(list(map(lambda x: 1.0 if x == 0 else x, num_a)))
    num_b = np.asarray(list(map(lambda x: 1.0 if x == 0 else x, num_b)))

    est_a = success_a / num_a
    est_b = success_b / num_b
    stats = (est_a - est_b)/np.sqrt(est_a * (1.0 - est_a) / num_a + est_b * (1.0 - est_b) / num_b)
    pval = (1.0 - norm.cdf(np.abs(stats))) * 2 # Two sided, symetric, so compare to 0.05

    return stats, pval


def perfrom_wald_test(
    simulation_df: pd.DataFrame, 
    policy: Union[TopTwoTSPolicy, TSPostDiffPolicy]
) -> pd.DataFrame:
    arm_dict = {}
    count = 0
    for index, row in policy.bandit.arm_data.arms.iterrows():
        arm_df = {}

        arm_name = row["name"]
        arm_success = "{} Success".format(arm_name).replace(" ", "_").lower()
        arm_count = "{} Count".format(arm_name).replace(" ", "_").lower()

        arm_dict[count] = {
            "success": simulation_df[arm_success].tolist(),
            "count": simulation_df[arm_count].tolist()
        }

        count += 1

    wald_statistics, pval = wald_test_statistics(
        arm_dict[0]["success"], arm_dict[1]["success"], 
        arm_dict[0]["count"], arm_dict[1]["count"]
    )

    wald_df = {
        "wald_statistics": wald_statistics,
        "wald_p_value": pval,
        "wald_significant": (pval < 0.05).astype(int)
    }

    evaluation_df = pd.concat([simulation_df, pd.DataFrame(wald_df).reset_index(drop=True)], axis=1)

    return evaluation_df
