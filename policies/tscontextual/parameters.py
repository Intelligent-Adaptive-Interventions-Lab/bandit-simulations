import pandas as pd
from typing import Dict, List

from policies.tscontextual.utils import create_design_matrix, posteriors

class TSContextualParams:
    parameters: Dict
    
    def __init__(self, initParams: Dict) -> None:
        self.parameters = initParams
    
    def update_params(self, values: pd.DataFrame, reward_name: str) -> None:
        design_matrix = create_design_matrix(
            values, 
            self.parameters["regression_formula"], 
            bool(self.parameters["include_intercept"])
        )

        numpy_design_matrix = design_matrix.values
        numpy_rewards = values[self.parameters['outcome_variable']].values

        posterior_vals = posteriors(
            numpy_rewards, 
            numpy_design_matrix, 
            self.parameters['coef_mean'], 
            self.parameters['coef_cov'], 
            self.parameters["variance_a"], 
            self.parameters["variance_b"]
        )

        self.parameters['coef_mean'] = posterior_vals["coef_mean"].tolist()
        self.parameters['coef_cov'] = posterior_vals["coef_cov"].tolist()
        self.parameters['variance_a'] = posterior_vals["variance_a"]
        self.parameters['variance_b'] = posterior_vals["variance_b"]
