import pandas as pd

from typing import Tuple, Dict

from numpy.random import choice, beta
from policies.tspostdiff.parameters import TSPostDiffParameter


def thompson_sampling_postdiff(params: TSPostDiffParameter) -> Tuple[Dict, Dict]:
    # A dict of versions to successes and failures e.g.:
	# {arm1: {success: 1, failure: 1}, arm2: {success: 1, failure: 1}, ...}
    priors = params.parameters["priors"]
    
    # Threshold for TS PostDiff.
    tspostdiff_thresh = params.parameters["tspostdiff_thresh"]
    
    # Drawing samples from beta discributions for two arms.
    arm_values = list(priors.values())
    arm_beta_1 = beta(arm_values[0]["success"], arm_values[0]["failure"])
    arm_beta_2 = beta(arm_values[1]["success"], arm_values[1]["failure"])
    
    # Absolute difference between two samples
    diff = abs(arm_beta_1 - arm_beta_2)

    assignment_data = {}
    arm_names = list(priors.keys())
    if diff < tspostdiff_thresh:
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
