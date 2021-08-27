import gym

env = gym.make('CartPole-v0')
env = gym.wrappers.Monitor(env, "recording", force=True)
env.reset()
for i in range(1000):
    obs, rew, done, info = env.step(env.action_space.sample())
    if done:
        env.reset()
