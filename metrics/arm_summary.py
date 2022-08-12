import pandas as pd
import numpy as np

from typing import List


def arm_summary(simulation_dfs: List[pd.DataFrame], reward_name: str) -> pd.DataFrame:
    all_simulation_df = []

    for i in range(len(simulation_dfs)):
        simulation_dfs[i]["trail_number"] = i
        all_simulation_df.append(simulation_dfs[i])

    arm_group = pd.concat(all_simulation_df).groupby(by=["trail_number", "arm"]).agg({reward_name: ['min', 'max', 'mean', 'std', 'sem', 'count'], 'arm' : ['count']})
    
    return arm_group
