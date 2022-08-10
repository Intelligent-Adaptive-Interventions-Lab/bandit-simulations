import pandas as pd
import numpy as np


def arm_summary(simulation_df: pd.DataFrame, reward_name: str) -> pd.DataFrame:
    arm_group = simulation_df.groupby(by=["arm"]).agg({reward_name: ['min', 'max', 'mean', 'std', 'sem', 'count'], 'arm' : ['count']})
    
    return arm_group.unstack(level=0).unstack()
