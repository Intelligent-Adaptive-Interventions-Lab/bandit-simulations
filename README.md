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
