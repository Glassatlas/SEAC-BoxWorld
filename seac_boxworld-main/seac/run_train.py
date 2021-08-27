import logging
import gym_boxworld # noqa
import a2c
import train
from model import DRRLConfig

logging.basicConfig(
    level=logging.INFO,
    format="(%(process)d) [%(levelname).1s] - (%(asctime)s) - %(name)s >> %(message)s",
    datefmt="%m/%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("debug.log")
    ]
)

env_name = "gym_boxworld:BoxWorldMA-v0"
# env_name = "gym_boxworld:BoxWorldRandMA-v0"

# create train config
run_config = train.RunConfig(
    env_name=env_name,
    time_limit=3000,
    log_interval=1000,
    eval_interval=1000,
    save_interval=2000
)

algo_config = a2c.AlgoConfig()
net_config = DRRLConfig()


def run():
    train.run(
        run_id=1,
        _log=logging,
        seed=0,
        run_config=run_config,
        algo_config=algo_config,
        net_config=net_config
    )


if __name__ == '__main__':
    run()
