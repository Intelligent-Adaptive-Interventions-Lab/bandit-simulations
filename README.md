# bandit-simulations
Simulation Code for Bandit Algorithms

## MHA Project
### TS-Contextual Bandit
TS-Contextual Bandit Algorithm in used for many [bandit designs](https://docs.google.com/spreadsheets/d/1KFP_gWkw4MinPCkz_qc4MFIB49jslPntWxhce8qRI8Y/edit#gid=0) in Mental Health America (MHA) project.

Detailed code of TS-Contextual Bandit can be accessed through the following link [here](https://github.com/pretendWhale/mooclet-engine/blob/a1310356785befd8928ba35b25a1d93b1f440d24/mooclet_engine/engine/policies.py#L466).

### TS-Traditional Bandit
TS-Traditional Bandit Algorithm follows Beta-Bernoulli with Thompson Sampling method.

Detailed code of Traditional Bandit can be accessed through the following link [here](https://github.com/pretendWhale/mooclet-engine/blob/a1310356785befd8928ba35b25a1d93b1f440d24/mooclet_engine/engine/policies.py#L1183).

### TS-PostDiff Bandit
TS-PostDiff Bandit Algorithm is similar to TS-Traditional but involves a threshold _c_ to adjust the policy of the bandit algorithm. Similar to epsilon-greedy, it mixes Uniform Random and TS-Traditional policy in the algorithm.

Detailed code of Traditional Bandit can be accessed through the following link [here](https://github.com/pretendWhale/mooclet-engine/blob/a1310356785befd8928ba35b25a1d93b1f440d24/mooclet_engine/engine/policies.py#L1183).

## Setup
### Install packages
* This code supports Python 3.9+
* `pip install -r requirements.txt`

## How To Run?
__Note:__ If you are running this code under [Jupyter Notebook](https://jupyter.org/)/[Google Colab](https://colab.research.google.com/) environments, you should include `--notebook_mode=True` to all following commands.

### Running Simulations
To run simulations for different policy settings, run the following command from the root directory to this repository:

```bash
python main.py simulate --config_path=<path_to_your_configs_file> --output_path=<path_to_your_outputs> --checkpoint_path=<path_to_your_checkpoints>
```
This command will write simulation results and evaluation results under two directory to `<path_to_your_outputs>`.
