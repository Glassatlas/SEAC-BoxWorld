# SEAC-BoxWorld
Shared-Experience Actor Critic Box World

This is a multi-agent extension of DeepMind's Relational Deep Reinforcement Learning

The project is based on the DRRL implementation: https://github.com/mavischer/DRRL
and the official SEAC implementation: https://github.com/uoe-agents/seac

 ## Installing the Environment
 This implementation makes use of gym registration.
 Install the environment by running inside the gym-box-world directory:
 
```bash
pip install -e .
```

## Training
- To run the training execute `run_train.py` within the seac directory.

- Within run_train you can set
`env_name = "gym_boxworld:BoxWorldMA-v0"`
or
`env_name = "gym_boxworld:BoxWorldRandMA-v0"` corresponding to Box-World setups with a fixed and a randomly placed box configuration respectively

- You can change the default hyperparameters in `run_train.py` by updating `run_config`, `algo_config` and `net_config`

## Evaluation
- Copy a saved model from `results/trained_models` to `pretrained/boxworld-ma`, make sure the model is decompressed, should be one directory for each agent
- Execute `evaluate.py`
