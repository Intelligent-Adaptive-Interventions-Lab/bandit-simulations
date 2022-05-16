import pandas as pd

from typing import Tuple, Dict

from numpy.random import choice, beta, uniform
from policies.toptwots.parameters import TopTwoTSParameter


def top_two_thompson_sampling(params: TopTwoTSParameter) -> Tuple[Dict, Dict]:
    # A dict of versions to successes and failures e.g.:
	# {arm1: {success: 1, failure: 1}, arm2: {success: 1, failure: 1}, ...}
    priors = params.parameters["priors"]
    
    # Threshold for TopTwo TS.
    epsilon_thresh = params.parameters["epsilon_thresh"]

    assignment_data = {}
    arm_names = list(priors.keys())
    if uniform(0, 1) < epsilon_thresh:
        # Uniform randomly picking an arm.
        best_action_name = choice(arm_names)
    
        for arm in arm_names:
            assignment_data[f"{arm} Success".replace(" ", "_").lower()] = priors[arm]["success"]
            assignment_data[f"{arm} Failure".replace(" ", "_").lower()] = priors[arm]["failure"]
        
        return best_action_name, assignment_data
    
    max_beta = 0
    for arm in arm_names:
        assignment_data[f"{arm} Success".replace(" ", "_").lower()] = priors[arm]["success"]
        assignment_data[f"{arm} Failure".replace(" ", "_").lower()] = priors[arm]["failure"]
        
        success = priors[arm]["success"]
        failure = priors[arm]["failure"]
        arm_beta = beta(success, failure)
        
        # Find higher beta sample for the optimal arm.
        if arm_beta > max_beta:
            max_beta = arm_beta
            best_action_name = arm
            
    return best_action_name, assignment_data
