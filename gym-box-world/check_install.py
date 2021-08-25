import gym

import gym_boxworld  # noqa

env = gym.make('BoxWorldMA-v0')
print(env.reset())
print(env.observation_space)
