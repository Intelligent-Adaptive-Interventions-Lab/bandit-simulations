import pandas as pd
import numpy as np

from typing import List


def arm_summary(simulation_dfs: List[pd.DataFrame], reward_name: str) -> pd.DataFrame:
    trail_column = ["trail_number"] + simulation_dfs[0].columns
    
    all_simulation_df = pd.DataFrame(columns=trail_column)
    
    for i in range(len(simulation_dfs)):
        simulation_dfs[0]["trail_number"] = i
        all_simulation_df = pd.concat([all_simulation_df, simulation_dfs[i]])
    
    arm_group = all_simulation_df.groupby(by=["trail_number", "arm"]).agg({reward_name: ['min', 'max', 'mean', 'std', 'sem', 'count'], 'arm' : ['count']})
    
    return arm_group
