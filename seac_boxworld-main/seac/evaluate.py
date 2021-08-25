import gym
import gym_boxworld  # noqa
import torch
from matplotlib import pyplot as plt, animation

from a2c import A2C, AlgoConfig
from model import DRRLConfig
from wrappers import RecordEpisodeStatistics, TimeLimit


def make_animation(obs_to_plot):
    fig = plt.figure()
    ims = []
    for obs in obs_to_plot:
        im = plt.imshow(obs / 255, animated=True)
        ims.append([im])

    ani = animation.ArtistAnimation(
        fig,
        ims,
        interval=50,
        blit=True,
        repeat_delay=1000
    )

    ani.save('results/video/record.mp4')


path = "pretrained/boxworld_ma"
env_name = "gym_boxworld:BoxWorldMA-v0"
time_limit = 500  # 25 for LBF

RUN_STEPS = 300

env = gym.make(env_name)
env = TimeLimit(env, time_limit)
env = RecordEpisodeStatistics(env)
# env = TransposeImage(env, op=[2, 0, 1])
# env = MADummyVecEnv([lambda: env])
# env = VecPyTorch(env, "cpu")

algo_config = AlgoConfig(lr=0.1, adam_eps=0.1, use_recurrent_policy=False, num_steps=1, num_processes=1, device="cpu")
net_config = DRRLConfig()

agents = [
    A2C(i, osp, asp, algo_config=algo_config, net_config=net_config)
    for i, (osp, asp) in enumerate(zip(env.observation_space, env.action_space))
]

for agent in agents:
    agent.restore(path + f"/agent{agent.agent_id}")

obs = env.reset()
obs_to_plot = [o for o in obs]

for i in range(RUN_STEPS):
    obs = [torch.from_numpy(o).moveaxis(-1, -3) for o in obs]
    _, actions, _, _ = zip(*[agent.model.act(obs[agent.agent_id], None, None) for agent in agents])
    actions = [a.item() for a in actions]
    # env.render()
    obs, _, done, info = env.step(actions)
    obs_to_plot.extend([o for o in obs])
    if all(done):
        obs = env.reset()
        obs_to_plot.extend([o for o in obs])
        print("--- Episode Finished ---")
        print(f"Episode rewards: {sum(info['episode_reward'])}")
        print(info)
        print(" --- ")

make_animation(obs_to_plot)
