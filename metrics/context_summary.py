import pandas as pd
import numpy as np


def context_summary(simulation_df: pd.DataFrame, reward_name: str, context: str) -> pd.DataFrame:
    context_group = simulation_df.groupby(by=[context, "arm"]).agg({reward_name: ['min', 'max', 'mean', 'std', 'sem', 'count'], 'arm' : ['count']})

    return context_group.stack(level=[0, 1]).unstack(level=[1, 0])
