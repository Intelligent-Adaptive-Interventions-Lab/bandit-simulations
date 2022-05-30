import pandas as pd
import numpy as np
from scipy.stats import invgamma, t, sem


from typing import List, Tuple


def mean_confidence_interval_sem(
    samples: List[float], 
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    n = len(samples)
    # we use SEM instead of STD
    m, se = np.mean(samples), sem(samples)
    h = se * t.ppf((1 + confidence) / 2., n-1)
    return m, m - h, m + h

def mean_confidence_interval_std(
    samples: List[float], 
    confidence: float = 0.95
) -> Tuple[float, float, float]:
    n = len(samples)
    m = samples.mean()
    s = samples.std()
    dof = n - 1

    t_crit = np.abs(t.ppf((1-confidence)/2,dof))
    left = m-s*t_crit/np.sqrt(n)
    right = m+s*t_crit/np.sqrt(n)

    return m, left, right

def draw_samples(
    variance_a: float, 
    variance_b: float, 
    coef_mean: List[float], 
    coef_cov: List[float],
    size: int = 10000
) -> np.ndarray:
    # could be where credible interval is too narrow
    precesion_draw = invgamma.rvs(variance_a, 0, variance_b, size=size)

    coef_draws = []
    for draw in precesion_draw:
        coef_draws.append(np.random.multivariate_normal(coef_mean, draw * coef_cov))

    coef_draws = np.asarray(coef_draws)
    return coef_draws


def estimate_confidence_interval(simulation_df: pd.DataFrame, formula: str) -> pd.DataFrame:
    columns = ["term", "coef_mean", "lower_bound", "upper_bound"]
    evaluation_df = pd.DataFrame(columns=columns)

    latest_params = simulation_df.iloc[-1]

    formula = formula.strip()
    all_vars_str = formula.split('~')[1].strip()
    dependent_var = formula.split('~')[0].strip()
    vars_list = all_vars_str.split('+')
    vars_list = ["INTERCEPT"] + list(map(str.strip, vars_list))

    mean = np.asarray(latest_params['coef_mean'])
    cov = np.asarray(latest_params['coef_cov'])
    variance_a = float(latest_params['variance_a'])
    variance_b = float(latest_params['variance_b'])

    coef_draws = draw_samples(variance_a, variance_b, mean, cov).T

    for sample_ind in range(coef_draws.shape[0]):
        samples = coef_draws[sample_ind]
        sample_coef_mean, sample_lower_ci, sample_upper_ci = mean_confidence_interval_sem(samples)
        sample_evaluation = {
            "term": vars_list[sample_ind],
            "coef_mean": float(sample_coef_mean), 
            "lower_bound": float(sample_lower_ci), 
            "upper_bound": float(sample_upper_ci)
        }
        evaluation_df = pd.concat([evaluation_df, pd.DataFrame.from_records([sample_evaluation])])

    evaluation_df = evaluation_df.assign(TermIndex=range(len(evaluation_df))).set_index('TermIndex')

    return evaluation_df
