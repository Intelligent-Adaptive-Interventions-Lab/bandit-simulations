import pandas as pd
import numpy as np

from typing import List


def context_summary(simulation_dfs: List[pd.DataFrame], reward_name: str, context: str) -> pd.DataFrame:
    all_simulation_df = []

    for i in range(len(simulation_dfs)):
        simulation_dfs[i]["trail_number"] = i
        all_simulation_df.append(simulation_dfs[i])
    
    context_group = pd.concat(all_simulation_df).groupby(by=["trail_number", context, "arm"]).agg({reward_name: ['min', 'max', 'mean', 'std', 'sem', 'count'], 'arm' : ['count']})

    return context_group
