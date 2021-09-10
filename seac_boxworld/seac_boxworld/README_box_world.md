# Multi-agent Box-World with Shared Experience Actor Critic 

## Training
- To run the training execute `run_train.py`

- Within run_train you can set
`env_name = "gym_boxworld:BoxWorldMA-v0"`
or
`env_name = "gym_boxworld:BoxWorldRandMA-v0"` corresponding to Box-World setups with a fixed and a randomly placed box configuration respectively

- You can change the default hyperparameters in `run_train.py` by updating `run_config`, `algo_config` and `net_config`

## Evaluation
- Copy a saved model from `results/trained_models` to `pretrained/boxworld-ma`, make sure the model is decompressed, should be one directory for each agent
- Execute `evaluate.py`