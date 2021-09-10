import time

import gym_boxworld  # noqa

from seac.envs import make_env
from seac.wrappers import SquashDones

if __name__ == '__main__':
    env_fn = make_env(
        env_id="gym_boxworld:BoxWorldMA-v0",
        # env_id='CartPole-v0',
        # env_id="lbforaging:Foraging-10x10-3p-3f-v1",
        seed=0,
        rank=0,
        time_limit=None,
        wrappers=[SquashDones],
        monitor_dir="recording",
        use_frame_stack=False
    )
    env = env_fn()
    # print(env.reset())
    # print("all good")

    env.reset()
    env.render()
    i = 0
    while True:
        i += 1
        obs, rew, done, info = env.step(env.action_space.sample())
        env.render()
        time.sleep(0.5)
        if done:
            break
