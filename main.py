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
    set_random: bool = False,
    random_seed: int = 42
) -> None:
    from tqdm import tqdm
   
    if not set_random:
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

    all_params = list(configs["parameters"])
    all_policies = []
    for init_params in all_params:
        # Initialize bandit settings.
        bandit = Bandit(reward, arms, contexts)

        # Choose corresponding policy from policy factory.
        policy = PolicyFactory(init_params, bandit).get_policy()

        all_policies.append(policy)
    
    # Sort the policy list by the unique reward indicator and unique contexts indicator.
    # E.g. policies which is both unique in rewards and contexts run the simulation first;
    # The priority of unique in rewards is higher than the priority of unique in contexts, in terms of sorting.
    all_policies.sort(key=lambda x: 10 * int(x.is_unique_reward()) + int(x.is_unique_contexts()), reverse=True)

    # Print the reward generating plan for all policies.
    first_policy = all_policies[0]
    print(
        "Policy {} is used for generating rewards: {}".format(
            first_policy.get_name(), first_policy.get_reward_generate_plan()
        )
    )

    reward_pool = None
    contexts_pool = None
    for i in range(len(all_policies)):
        policy = all_policies[i]

        # Get columns of simulation dataframe
        columns = policy.columns

        # Initialize simulation dataframe
        simulation_df = pd.DataFrame(columns=columns)

        # Check whether the reward and contexts should be merged before running simulation.
        is_unique_reward = policy.is_unique_reward() and reward_pool is None
        is_unique_contexts = policy.is_unique_contexts() and contexts_pool is None
        if not is_unique_reward:
            if not is_unique_contexts:
                merged_pool = pd.concat([reward_pool, contexts_pool], axis=1)
                simulation_df = pd.concat([simulation_df, merged_pool])
            else:
                simulation_df = pd.concat([simulation_df, reward_pool])

        for trail in tqdm(range(numTrails), desc='Trails'):
            # Initialize one update batch of datapoints
            assignment_df = pd.DataFrame(columns=columns)
            for learner in tqdm(range(horizon), desc='Horizons'):
                # Register a new learner.
                new_learner = f"learner_{learner:03d}_{trail:03d}"

                # Initialize one dataframe for the new learner.
                new_learner_df = pd.DataFrame(columns=columns)

                # Copy ontext values if policy is not unique in contexts.
                new_learner_dict = {}
                if not is_unique_contexts:
                    new_learner_dict = policy.get_contexts(new_learner_dict)

                # Get datapoints (e.g. assigned arm, generated contexts) for the new learner.
                new_learner_data = policy.run(new_learner, new_learner_dict)

                # Merge to new learner dataframe.
                new_learner_df = pd.concat([new_learner_df, pd.DataFrame.from_records([new_learner_data])])

                # Update rewards for the new learner.
                # Copy reward values if policy is not unique in rewards.
                if not is_unique_reward:
                    new_learner_df[policy.get_reward_column_name()] = simulation_df.loc[trail * horizon + learner, policy.get_reward_column_name()]
                    new_learner_df = new_learner_df.rename(index={0: trail * horizon + learner})
                else:
                    new_learner_df = policy.get_reward(new_learner_df)

                # Merge to the update batch.
                assignment_df = pd.concat([assignment_df, new_learner_df])

                # Check if parameters should update.
                if len(assignment_df.index) >= policy.params.parameters["batch_size"]:
                    # Update parameters.
                    assignment_df = policy.update_params(assignment_df)

                    # Merge to simulation dataframe.
                    if not is_unique_reward:
                        simulation_df = simulation_df.fillna(assignment_df)
                    else:
                        simulation_df = pd.concat([simulation_df, assignment_df])

                    # Re-initialize the update batch of datapoints.
                    assignment_df = pd.DataFrame(columns=columns)

        print("{} arm data:".format(policy.get_name()))
        print(policy.bandit.arm_data.arms)
        simulation_df = simulation_df.assign(Index=range(len(simulation_df))).set_index('Index')

        # Create the reward pool if policy is unique in rewards.
        if is_unique_reward:
            reward_pool = simulation_df[[policy.get_reward_column_name()]]
        
        # Create the contexts pool if policy is unique in contexts.
        if is_unique_contexts:
            contexts_pool = simulation_df[policy.bandit.get_contextual_variables()]

        simulation_output_path = configs["simulation"]
        os.makedirs(f"{output_path}/{simulation_output_path}", exist_ok=True)

        simulation_result_name = "{}_results".format(policy.get_name())
        simulation_df.to_csv(f"{output_path}/{simulation_output_path}/{simulation_result_name}.csv")
    
        # Evaluate
        evaluator = EvaluatorFactory(simulation_df, policy).get_evaluator()
        evaluation_output_path = configs["evaluation"]
        os.makedirs(f"{output_path}/{evaluation_output_path}", exist_ok=True)
        os.makedirs(f"{output_path}/{evaluation_output_path}/metrics", exist_ok=True)
        for metric in list(evaluator.metrics.keys()):
            metric_name = "{}_{}".format(policy.get_name(), metric)
            evaluator.metrics[metric].to_csv(f"{output_path}/{evaluation_output_path}/metrics/{metric_name}.csv")

    # if checkpoint_path is None:
    #     # Get columns of simulation dataframe
    #     columns = policy.columns

    #     # Initialize simulation dataframe
    #     simulation_df = pd.DataFrame(columns=columns)
    #     for trail in tqdm(range(numTrails), desc='Trails'):
    #         # Initialize one update batch of datapoints
    #         assignment_df = pd.DataFrame(columns=columns)
    #         for learner in tqdm(range(horizon), desc='Horizons'):
    #             # Register a new learner.
    #             new_learner = f"learner_{learner:03d}_{trail:03d}"

    #             # Initialize one dataframe for the new learner.
    #             new_learner_df = pd.DataFrame(columns=columns)

    #             # Get datapoints (e.g. assigned arm, generated contexts) for the new learner.
    #             new_learner_data = policy.run(new_learner)

    #             # Merge to new learner dataframe.
    #             new_learner_df = pd.concat([new_learner_df, pd.DataFrame.from_records([new_learner_data])])

    #             # Update rewards for the new learner.
    #             new_learner_df = policy.get_reward(new_learner_df)

    #             # Merge to the update batch.
    #             assignment_df = pd.concat([assignment_df, new_learner_df])

    #             # Check if parameters should update.
    #             if len(assignment_df.index) >= policy.params.parameters["batch_size"]:
    #                 # Update parameters.
    #                 assignment_df = policy.update_params(assignment_df)

    #                 # Merge to simulation dataframe.
    #                 simulation_df = pd.concat([simulation_df, assignment_df])

    #                 # Re-initialize the update batch of datapoints.
    #                 assignment_df = pd.DataFrame(columns=columns)

    #     print("arm data:")
    #     print(bandit.arm_data.arms)
    #     simulation_df = simulation_df.assign(Index=range(len(simulation_df))).set_index('Index')

    #     simulation_output_path = configs["simulation"]
    #     os.makedirs(f"{output_path}/{simulation_output_path}", exist_ok=True)
    #     simulation_df.to_csv(f"{output_path}/{simulation_output_path}/results.csv")
    # else:
    #     simulation_df = pd.read_csv(checkpoint_path)
    #     simulation_df = clean_df_from_csv(simulation_df)


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
