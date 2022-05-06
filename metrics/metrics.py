import pandas as pd
import numpy as np
from scipy.stats import invgamma, t, sem

from typing import List

pd.options.mode.chained_assignment = None


def mean_confidence_interval(samples, confidence=0.95):
    n = len(samples)
    m, se = np.mean(samples), sem(samples)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m - h, m + h


def draw_samples(
    variance_a: float, 
    variance_b: float, 
    coef_mean: List[float], 
    coef_cov: List[float],
    size: int = 10000
) -> np.ndarray:
    precesion_draw = invgamma.rvs(variance_a, 0, variance_b, size=1)
    coef_draws = np.random.multivariate_normal(coef_mean, precesion_draw * coef_cov, size=size)

    return coef_draws


def evaluate(simulation_df: pd.DataFrame) -> pd.DataFrame:
    columns = ["coef_mean", "lower_ci", "upper_ci"]
    evaluation_df = pd.DataFrame(columns=columns)

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

    latest_params = simulation_df.iloc[-1]

    mean = np.asarray(latest_params['coef_mean'])
    cov = np.asarray(latest_params['coef_cov'])
    variance_a = float(latest_params['variance_a'])
    variance_b = float(latest_params['variance_b'])

    coef_draws = draw_samples(variance_a, variance_b, mean, cov).T

    for sample_ind in range(coef_draws.shape[0]):
        samples = coef_draws[sample_ind]
        sample_coef_mean, sample_lower_ci, sample_upper_ci = mean_confidence_interval(samples)
        sample_evaluation = {
            "coef_mean": float(sample_coef_mean), 
            "lower_ci": float(sample_lower_ci), 
            "upper_ci": float(sample_upper_ci)
        }
        evaluation_df = pd.concat([evaluation_df, pd.DataFrame.from_records([sample_evaluation])])
    
    evaluation_df = evaluation_df.assign(TermIndex=range(len(evaluation_df))).set_index('TermIndex')

    return evaluation_df
