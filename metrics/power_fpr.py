import pandas as pd
import numpy as np

from typing import List


def compute_power(
    simulation_dfs: List[pd.DataFrame], 
    wald_dfs: pd.DataFrame, 
    reward_name: str
) -> pd.DataFrame:
    mean_rewards = []
    for simulation_df in simulation_dfs:
        mean_reward = np.mean(simulation_df[reward_name])
        mean_rewards.append(mean_reward)
    
    avg_mean_reward = np.mean(mean_rewards)
    power = np.mean(wald_dfs["wald_statistics"] > 1.96)
    
    return avg_mean_reward, power


def compute_fpr(wald_dfs: pd.DataFrame) -> pd.DataFrame:
    fpr = np.mean(wald_dfs["wald_significant"])
    
    return fpr


def power_fpr(    
    simulation_dfs: List[pd.DataFrame], 
    wald_dfs: pd.DataFrame, 
    reward_name: str
) -> pd.DataFrame:
    avg_mean_reward, power = compute_power(simulation_dfs, wald_dfs, reward_name)
    fpr = compute_fpr(wald_dfs)
    
    power_fpr_df = {
        "avg_mean_reward": avg_mean_reward,
        "power": power,
        "fpr": fpr
    }
    
    return pd.DataFrame([power_fpr_df]).reset_index(drop=True)
