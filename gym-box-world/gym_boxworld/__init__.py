from gym.envs.registration import register

register(
    id='BoxWorld-v4',
    entry_point='gym_boxworld.envs:BoxWorldEnv'
)

register(
    id='BoxWorldRand-v4',
    entry_point='gym_boxworld.envs:BoxWorldRandEnv'
)

register(
    id='BoxWorldMA-v0',
    entry_point='gym_boxworld.envs:BoxWorldMAEnv'
)

register(
    id='BoxWorldRandMA-v0',
    entry_point='gym_boxworld.envs:BoxWorldRandMAEnv'
)
