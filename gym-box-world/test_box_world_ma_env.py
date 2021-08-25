import matplotlib.pyplot as plt

from gym_boxworld.envs.box_world_ma_env import BoxWorldMAEnv


def test_create():
    env = BoxWorldMAEnv()
    assert len(env.agent_current_state) == 2
    assert (env.agent_current_state == [[5, 3], [12, 12]]).all()


def test_reset():
    env = BoxWorldMAEnv()
    obs = env.reset()
    plt.imshow(obs / 255)
    plt.show()


def test_step():
    env = BoxWorldMAEnv()
    res = env.step([0, 0])
    print(res[0].shape)
    print(res[1:])
