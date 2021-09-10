from typing import List

import gym
import numpy as np
import torch
from gym.spaces import Box
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnvWrapper, VecNormalize

from seac.wrappers import TimeLimit, Monitor, ClearInfo


class MADummyVecEnv(DummyVecEnv):
    def __init__(self, env_fns):
        super().__init__(env_fns)
        agents = len(self.observation_space)
        # change this because we want >1 reward
        self.buf_rews = np.zeros((self.num_envs, agents), dtype=np.float32)


def make_env(
        env_id: str,
        seed: int,
        rank,
        time_limit,
        wrappers: List,
        monitor_dir: str,
        should_clear_info: bool = True,
        use_frame_stack: bool = False
):
    def _thunk():

        if type(env_id) == tuple:
            env = gym.make(env_id[0], **env_id[1])
            env_name_id = env_id[0]
        else:
            env = gym.make(env_id)
            env_name_id = env_id

        env.seed(seed + rank)

        # algorithm implementation relies on info being empty by default
        if should_clear_info:
            env = ClearInfo(env)

        if time_limit:
            # limit number of steps
            env = TimeLimit(env, time_limit)

        # other wrappers
        for wrapper in wrappers:
            env = wrapper(env)

        if monitor_dir:
            # doesn't really work with box world
            env = Monitor(env, monitor_dir, lambda ep: int(ep == 0), force=True, uid=str(rank))

        # If the input has shape (W,H,3), wrap for PyTorch convolutions
        obs_shape = env.observation_space[0].shape
        if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env

    return _thunk


def make_vec_envs(
        env_name, seed, dummy_vecenv, parallel, time_limit, wrappers, device, monitor_dir=None
):
    envs = [
        make_env(env_name, seed, i, time_limit, wrappers, monitor_dir) for i in range(parallel)
    ]

    if dummy_vecenv or len(envs) == 1 or monitor_dir:
        envs = MADummyVecEnv(envs)
    else:
        envs = SubprocVecEnv(envs)

    envs = VecPyTorch(envs, device)
    return envs


class VecPyTorch(VecEnvWrapper):
    def __init__(self, venv, device):
        """Return only every `skip`-th frame"""
        super(VecPyTorch, self).__init__(venv)
        self.device = device
        # TODO: Fix data types

    def reset(self):
        obs = self.venv.reset()
        return [torch.from_numpy(o).to(self.device) for o in obs]
        # return obs

    def step_async(self, actions):
        actions = [a.squeeze().cpu().numpy() for a in actions]
        actions = list(zip(*actions))
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, rew, done, info = self.venv.step_wait()
        return (
            [torch.from_numpy(o).float().to(self.device) for o in obs],
            torch.from_numpy(rew).float().to(self.device),
            torch.from_numpy(done).float().to(self.device),
            info,
        )


class TransposeObs(gym.ObservationWrapper):
    def __init__(self, env=None):
        """
        Transpose observation space (base class)
        """
        super(TransposeObs, self).__init__(env)


class TransposeImage(TransposeObs):

    def __init__(self, env=None, op=(2, 0, 1)):
        """
        Transpose observation space for images
        """
        super(TransposeImage, self).__init__(env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        if isinstance(self.observation_space, gym.spaces.Tuple):
            boxes = [self.copy_box(box) for box in self.observation_space]
            self.observation_space = gym.spaces.Tuple(tuple(boxes))
        else:
            self.observation_space = self.copy_box(self.observation_space)

    def copy_box(self, box: gym.spaces.Box) -> gym.spaces.Box:
        new_box = Box(
            box.low[0, 0, 0],
            box.high[0, 0, 0],
            [
                box.shape[self.op[0]],
                box.shape[self.op[1]],
                box.shape[self.op[2]]
            ],
            dtype=box.dtype
        )

        return new_box

    def observation(self, ob):
        if isinstance(ob, tuple):
            return tuple([o.transpose(self.op[0], self.op[1], self.op[2]) for o in ob])
        else:
            return ob.transpose(self.op[0], self.op[1], self.op[2])


class MyVecNormalize(VecNormalize):
    def __init__(self, *args, **kwargs):
        super(MyVecNormalize, self).__init__(*args, **kwargs)
        self.training = True

    def _obfilt(self, obs, update=True):
        if self.ob_rms:
            if self.training and update:
                self.ob_rms.update(obs)
            obs = np.clip(
                (obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob
            )
            return obs
        else:
            return obs

    def train(self):
        self.training = True

    def eval(self):
        self.training = False


# Derived from
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/vec_frame_stack.py
class VecPyTorchFrameStack(VecEnvWrapper):
    def __init__(self, venv, nstack, device=None):
        self.venv = venv
        self.nstack = nstack

        wos = venv.observation_space  # wrapped ob space
        print(wos)
        print(wos.shape)
        self.shape_dim0 = wos.shape[0]

        low = np.repeat(wos.low, self.nstack, axis=0)
        high = np.repeat(wos.high, self.nstack, axis=0)

        if device is None:
            device = torch.device('cpu')

        self.stacked_obs = torch.zeros((venv.num_envs,) + low.shape).to(device)

        observation_space = gym.spaces.Box(low=low, high=high, dtype=venv.observation_space.dtype)
        VecEnvWrapper.__init__(self, venv, observation_space=observation_space)

    def step_wait(self):
        obs, rewards, news, infos = self.venv.step_wait()
        self.stacked_obs[:, :-self.shape_dim0] = self.stacked_obs[:, self.shape_dim0:].clone()
        for (i, new) in enumerate(news):
            if new:
                self.stacked_obs[i] = 0
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs, rewards, news, infos

    def reset(self):
        obs = self.venv.reset()
        if torch.backends.cudnn.deterministic:
            self.stacked_obs = torch.zeros(self.stacked_obs.shape)
        else:
            self.stacked_obs.zero_()
        self.stacked_obs[:, -self.shape_dim0:] = obs
        return self.stacked_obs

    def close(self):
        self.venv.close()
