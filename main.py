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
from metrics.evaluator import EvaluatorFactory
from utils.clean import clean_df_from_csv


def simulate(
    config_path: str, 
    output_path: str, 
    checkpoint_path: str = None,
    notebook_mode: bool = False,
    set_random: bool = True,
    random_seed: int = 42
) -> None:
    from tqdm import tqdm
   
    if set_random:
        np.random.seed(seed=random_seed) 
  
    if notebook_mode:
        from functools import partial
        tqdm = partial(tqdm, position=0, leave=True)
        pd.options.display.max_columns = None   

    os.makedirs(output_path, exist_ok=True)

    configs_file = open(config_path)
    configs = json.load(configs_file)

    numTrails = configs["numTrails"]
    horizon = configs["numLearners"]

    arms = list(configs["arms"])
    contexts = list(configs["contexts"]) if "contexts" in configs else None
    reward = dict(configs["reward"])

    # Initialize bandit settings.
    bandit = Bandit(reward, arms, contexts)

    # Initialize policy parameters.
    init_params = dict(configs["parameters"])

    # Choose corresponding policy from policy factory.
    policy = PolicyFactory(init_params, bandit).get_policy()

    if checkpoint_path is None:
        # Get columns of simulation dataframe
        columns = policy.columns

        # Initialize simulation dataframe
        simulation_df = pd.DataFrame(columns=columns)
        for trail in tqdm(range(numTrails), desc='Trails'):
            # Initialize one update batch of datapoints
            assignment_df = pd.DataFrame(columns=columns)
            for learner in tqdm(range(horizon), desc='Horizons'):
                # Register a new learner.
                new_learner = f"learner_{learner:03d}_{trail:03d}"

                # Initialize one dataframe for the new learner.
                new_learner_df = pd.DataFrame(columns=columns)

                # Get datapoints (e.g. assigned arm, generated contexts) for the new learner.
                new_learner_data = policy.run(new_learner)

                # Merge to new learner dataframe.
                new_learner_df = pd.concat([new_learner_df, pd.DataFrame.from_records([new_learner_data])])

                # Update rewards for the new learner.
                new_learner_df = policy.get_reward(new_learner_df)

                # Merge to the update batch.
                assignment_df = pd.concat([assignment_df, new_learner_df])

                # Check if parameters should update.
                if len(assignment_df.index) >= policy.params.parameters["batch_size"]:
                    # Update parameters.
                    assignment_df = policy.update_params(assignment_df)

                    # Merge to simulation dataframe.
                    simulation_df = pd.concat([simulation_df, assignment_df])

                    # Re-initialize the update batch of datapoints.
                    assignment_df = pd.DataFrame(columns=columns)

        print("arm data:")
        print(bandit.arm_data.arms)
        simulation_df = simulation_df.assign(Index=range(len(simulation_df))).set_index('Index')

        simulation_output_path = configs["simulation"]
        os.makedirs(f"{output_path}/{simulation_output_path}", exist_ok=True)
        simulation_df.to_csv(f"{output_path}/{simulation_output_path}/results.csv")
    else:
        simulation_df = pd.read_csv(checkpoint_path)
        simulation_df = clean_df_from_csv(simulation_df)

    # Evaluate
    evaluator = EvaluatorFactory(simulation_df, policy).get_evaluator()
    evaluation_output_path = configs["evaluation"]
    os.makedirs(f"{output_path}/{evaluation_output_path}", exist_ok=True)
    os.makedirs(f"{output_path}/{evaluation_output_path}/metrics", exist_ok=True)
    for metric in list(evaluator.metrics.keys()):
        evaluator.metrics[metric].to_csv(f"{output_path}/{evaluation_output_path}/metrics/{metric}.csv")


if __name__ == "__main__":
    import fire

    fire.Fire()

    # parser = argparse.ArgumentParser()
    # parser.add_argument("--config", required=True, help="the input path of simulation configs")
    # parser.add_argument("--output", required=True, help="the output path of simulation results")
    # parser.add_argument("--checkpoint", required=False, help="the checkpoint path of simulation results")
    # args = parser.parse_args()
    # os.makedirs(args.output, exist_ok=True)

    # if args.checkpoint:
    #     simulate(args.config, args.output, args.checkpoint)
    # else:
    #     simulate(args.config, args.output)
