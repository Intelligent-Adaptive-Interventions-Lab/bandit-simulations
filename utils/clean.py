import pandas as pd
import numpy as np

pd.options.mode.chained_assignment = None


def clean_df_from_csv(simulation_df: pd.DataFrame) -> pd.DataFrame:
    simulation_df = simulation_df.dropna()

    simulation_df.loc[:,"coef_mean"] = simulation_df["coef_mean"].apply(
        lambda x: np.fromstring(
            x.replace('\n','').replace('[','').replace(']','').strip(), 
            sep=','
        )
    )

    length = len(simulation_df["coef_mean"][0])

    simulation_df.loc[:,"coef_cov"] = simulation_df["coef_cov"].apply(
        lambda x: np.fromstring(
            x.replace('\n','').replace('[','').replace(']','').strip(), 
            sep=','
        )
    ).apply(
        lambda x: x.reshape(length, length)
    )

    return simulation_df
