import argparse
import os
import sys
import pandas as pd
import numpy as np
import json

from datasets.arms import ArmData
from datasets.contexts import ContextAllocateData
from datasets.bandits import Bandit
from datasets.policies import PolicyFactory
from metrics.metrics import evaluate
from utils.clean import clean_df_from_csv


def simulate(
    config_path: str, 
    output_path: str, 
    checkpoint_path: str = None
) -> None:
    configs_file = open(config_path)
    configs = json.load(configs_file)

    numTrails = configs["numTrails"]
    horizon = configs["numLearners"]

    arms = list(configs["arms"])
    contexts = list(configs["contexts"])
    reward = dict(configs["reward"])

    # Initialize bandit settings.
    bandit = Bandit(reward, arms, contexts)

    # Initialize policy parameters.
    init_params = dict(configs["parameters"])

    # Choose corresponding policy from policy factory.
    tscontextual_policy = PolicyFactory(init_params, bandit).get_policy()

    if checkpoint_path is None:
        # Get columns of simulation dataframe
        columns = tscontextual_policy.columns

        # Initialize simulation dataframe
        simulation_df = pd.DataFrame(columns=columns)
        for trail in range(numTrails):
            # Initialize one update batch of datapoints
            assignment_df = pd.DataFrame(columns=columns)
            for learner in range(horizon):
                # Register a new learner.
                new_learner = f"learner_{learner:03d}_{trail:03d}"

                # Initialize one dataframe for the new learner.
                new_learner_df = pd.DataFrame(columns=columns)

                # Get datapoints (e.g. assigned arm, generated contexts) for the new learner.
                new_learner_data = tscontextual_policy.run(new_learner)

                # Merge to new learner dataframe.
                new_learner_df = pd.concat([new_learner_df, pd.DataFrame.from_records([new_learner_data])])

                # Update rewards for the new learner.
                new_learner_df = tscontextual_policy.get_reward(new_learner_df)

                # Merge to the update batch.
                assignment_df = pd.concat([assignment_df, new_learner_df])

                # Check if parameters should update.
                if len(assignment_df.index) >= tscontextual_policy.params.parameters["batch_size"]:
                    # Update parameters.
                    assignment_df = tscontextual_policy.update_params(assignment_df)

                    # Merge to simulation dataframe.
                    simulation_df = pd.concat([simulation_df, assignment_df])

                    # Re-initialize the update batch of datapoints.
                    assignment_df = pd.DataFrame(columns=columns)

        simulation_df = simulation_df.assign(Index=range(len(simulation_df))).set_index('Index')

        simulation_output_path = configs["simulation"]
        simulation_df.to_csv(f"{output_path}/{simulation_output_path}.csv")

        evaluation_df = evaluate(simulation_df)
        evaluation_output_path = configs["evaluation"]
        evaluation_df.to_csv(f"{output_path}/{evaluation_output_path}.csv")
    else:
        simulation_df = pd.read_csv(checkpoint_path)
        simulation_df = clean_df_from_csv(simulation_df)
        evaluation_df = evaluate(simulation_df)
        evaluation_output_path = configs["evaluation"]
        evaluation_df.to_csv(f"{output_path}/{evaluation_output_path}.csv")


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
