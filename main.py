import argparse
import os
import sys
import pandas as pd
import numpy as np
import json

from datasets.arms import ArmData
from datasets.contexts import ContextAllocateData
from tscontextual.ts_contextual import thompson_sampling_contextual
from tscontextual.parameters import TSContextualParams
from tscontextual.generate_rewards import get_reward
from metrics.metrics import evaluate


def simulate(
    config_path: str, 
    output_path: str, 
    checkpoint_path: str = None
):
    configs_file = open(config_path)
    configs = json.load(configs_file)

    numTrails = configs["numTrails"]
    horizon = configs["numLearners"]

    arms = list(configs["arms"])
    contexts = list(configs["contexts"])
    reward = dict(configs["reward"])

    terms = []
    action_space = {}
    actions = []
    for arm in arms:
        arm = dict(arm)
        if arm["action_variable"] is not None and arm["action_variable"] not in action_space:
            action_space[arm["action_variable"]] = [0, 1]
            terms.append(arm["action_variable"])
            actions.append(arm["action_variable"])
    
    arm_df = ArmData(actions, arms).arms
    
    contextual_variables = []
    contexts_lst = {}
    for context in contexts:
        context = dict(context)
        context_name = context["name"]
        contextual_variables.append(context_name)
        contexts_lst[context_name] = ContextAllocateData(context["values"], context["allocations"])
        if context['extra'] is True:
            terms.append(context_name)
        if context['interaction'] is True:
            for name in actions:
                terms.append(f"{name} * {context_name}")

    init_params = dict(configs["parameters"])
    length = len(terms) + 1 if init_params["include_intercept"] == 1 else len(terms)
    init_params["coef_cov"] = np.eye(length, dtype=float).tolist()
    init_params["coef_mean"] = np.zeros(length, dtype=float).tolist()
    init_params["action_space"] = action_space
    init_params["contextual_variables"] = contextual_variables

    reward_name = reward["name"]
    init_params["outcome_variable"] = reward_name
    if "regression_formula" in init_params and init_params["regression_formula"] is None:
        init_params["regression_formula"] = f"{reward_name} ~ {' + '.join(terms)}"
    
    print("regression_formula: {}".format(init_params["regression_formula"]))

    if checkpoint_path is None:
        params = TSContextualParams(init_params)
        
        columns = ["learner", "arm", reward_name] + actions + contextual_variables + ["coef_cov", "coef_mean", "variance_a", "variance_b", "precesion_draw", "coef_draw", "update_batch"]
        simulation_df = pd.DataFrame(columns=columns)
        update_count = 0
        for trail in range(numTrails):
            assignment_df = pd.DataFrame(columns=columns)
            for learner in range(horizon):
                new_learner_df = {}

                new_learner = f"learner_{learner:03d}_{trail:03d}"
                new_learner_df["learner"] = new_learner

                best_action, assignment_data = thompson_sampling_contextual(params, contexts_lst)
                best_arm_row = arm_df.loc[(arm_df[list(best_action)] == pd.Series(best_action)).all(axis=1)]
                new_learner_df["arm"] = str(best_arm_row["name"][0])
                new_learner_df = new_learner_df | assignment_data | best_action

                assignment_df = pd.concat([assignment_df, pd.DataFrame.from_records([new_learner_df])])
                assignment_df = get_reward(
                    assignment_df, 
                    params.parameters["true_estimate"], 
                    params.parameters["true_coef_mean"], 
                    params.parameters["include_intercept"],
                    reward
                )

                if len(assignment_df.index) >= params.parameters["batch_size"]:
                    assignment_df["update_batch"] = update_count
                    simulation_df = pd.concat([simulation_df, assignment_df])
                    value_col = [reward_name] + actions + contextual_variables
                    params.update_params(assignment_df[value_col], reward_name)
                    assignment_df = pd.DataFrame(columns=columns)
                    update_count += 1
                
                # print(f"simulation_df: {simulation_df}")

        simulation_df = simulation_df.assign(Index=range(len(simulation_df))).set_index('Index')
        simulation_df.to_csv(f"{output_path}/link_outputs.csv")

        evaluation_df = evaluate(simulation_df)
        evaluation_df.to_csv(f"{output_path}/link_evaluation.csv")
    else:
        simulation_df = pd.read_csv(checkpoint_path)
        evaluation_df = evaluate(simulation_df)
        evaluation_df.to_csv(f"{output_path}/link_evaluation.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="the input path of simulation configs")
    parser.add_argument("--output", required=True, help="the output path of simulation results")
    parser.add_argument("--checkpoint", required=False, help="the checkpoint path of simulation results")
    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    if args.checkpoint:
        simulate(args.config, args.output, args.checkpoint)
    else:
        simulate(args.config, args.output)
