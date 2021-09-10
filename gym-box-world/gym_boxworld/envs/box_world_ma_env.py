import copy
import os
from typing import List

import gym
import matplotlib.pyplot as plt
import numpy as np
from gym import spaces
from gym.utils import seeding

ACTION_MEANING = {
    0: "NOOP",
    1: "UP",
    2: "DOWN",
    3: "LEFT",
    4: "RIGHT",
}

# background: 0, 1 and agents: 16, 17 (changed from 2)
BGAndAG_COLORS = {
    0: [0., 0., 0.],
    1: [169., 169., 169.],
    16: [85., 85., 85.],
    17: [105., 105., 105.]
}

# To keep track of the agent colors
AGENT_COLOR_IDS = [16, 17]

# gem: 3
CorrectBox_COLORS = {
    3: [255., 255., 255.],
    4: [0., 255., 0.],
    5: [255.0, 0., 0.],
    6: [255., 0., 255.],
    7: [255., 255., 0.]
}

DistractorBox_COLORS = {
    8: [0., 255., 255.],
    9: [255.0, 127.5, 127.5],
    10: [127.5, 0., 255.],
    11: [255., 127.5, 0.]
}

COLORS = dict(
    list(BGAndAG_COLORS.items()) +
    list(CorrectBox_COLORS.items()) +
    list(DistractorBox_COLORS.items())
)


class BoxWorldMAEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        self.seed()
        self.use_noop_actions: bool = True
        self.n_agents = 2

        if self.use_noop_actions:
            self._action_set = [0, 1, 2, 3, 4]
        else:  # ignore noop action
            self._action_set = [1, 2, 3, 4]

        # actions space for multiple agents
        self.action_space = gym.spaces.Tuple(tuple([gym.spaces.Discrete(len(self._action_set))] * self.n_agents))
        self.action_pos_dict = {0: [0, 0], 1: [-1, 0], 2: [1, 0], 3: [0, -1], 4: [0, 1]}

        # set observation space
        self.box_size = 14
        # scale
        scale = 1
        self.obs_shape = [self.box_size * scale, self.box_size * scale, 3]
        obs_space = spaces.Box(low=0, high=255, shape=self.obs_shape, dtype=np.uint8)
        self.observation_space = gym.spaces.Tuple(tuple([obs_space] * self.n_agents))

        # initialize system
        self.CorrectBox_lists = list(range(3, 3 + len(CorrectBox_COLORS)))
        # DistractorBox_lists = range(3 + len(CorrectBox_COLORS), 3 + len(CorrectBox_COLORS) + len(DistractorBox_COLORS))

        this_file_path = os.path.dirname(os.path.realpath(__file__))
        self.world_map_path = os.path.join(this_file_path, 'plan_ma.txt')
        self.init_world_map = self._read_world_map(self.world_map_path)
        self.current_world_map = copy.deepcopy(self.init_world_map)
        self.observation = self._worldmap_to_obervation(self.init_world_map)
        self.world_map_shape = self.init_world_map.shape

        # agent states
        self.agent_init_state = np.array([self._get_agent(i, self.init_world_map) for i in range(self.n_agents)])
        self.agent_current_state = copy.deepcopy(self.agent_init_state)
        self.key = 0  # init no key (dark)

    def reset(self):
        self.current_world_map = copy.deepcopy(self.init_world_map)
        self.agent_current_state = copy.deepcopy(self.agent_init_state)
        self.observation = self._worldmap_to_obervation(self.init_world_map)
        return tuple([self.observation] * self.n_agents)

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed)
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(seed1 + 1) % 2 ** 31
        # Empirically, we need to seed before loading the ROM.
        # self.ale.setInt(b'random_seed', seed2)
        # self.ale.loadROM(self.game_path)
        return [seed1, seed2]

    def step(self, actions: List[int]):
        """
        takes a list of actions (one for each agent) and executes them one by one
        returns obs, rewards, dones, info
        """
        if len(actions) != 2:
            raise ValueError("Only works with two agents, please input two actions")

        obs = []
        rewards = []
        dones = []
        infos = []

        done = False
        for agent_id, action in enumerate(actions):
            if done:
                # if already done, don't step the rest of the agents
                obs.append(obs[-1])  # use same obs
                rewards.append(0)
                dones.append(True)  # one is done -> both are done
                infos.append(infos[-1])  # use same info
            else:
                ob, reward, done, info = self.step_agent(agent_id, action)
                obs.append(ob)
                rewards.append(reward)
                dones.append(done)
                infos.append(info)

        rewards = np.array(rewards, dtype=float)
        rewards[:] = rewards.mean()  # each agent gets the same reward
        dones = [any(dones)] * self.n_agents  # share done

        return tuple(obs), tuple(rewards), tuple(dones), infos[-1]

    def step_agent(self, agent_id: int, action: int):
        """
        steps a single agent
        """
        info = {'success': False}
        action = self._action_set[action]

        next_agent_state = self.agent_current_state[agent_id] + self.action_pos_dict[action]

        if action == 0:  # noop
            return (self.observation, 0, False, info)

        # invalid action cross border
        if next_agent_state[0] < 1 or next_agent_state[0] > (self.world_map_shape[0] - 2):
            return (self.observation, 0, False, info)
        if next_agent_state[1] < 1 or next_agent_state[1] > (self.world_map_shape[0] - 2):
            return (self.observation, 0, False, info)

        nxt_color = self.current_world_map[next_agent_state[0], next_agent_state[1]]
        nxt_left_color = self.current_world_map[next_agent_state[0], next_agent_state[1] - 1]
        nxt_right_color = self.current_world_map[next_agent_state[0], next_agent_state[1] + 1]

        # cannot step onto the other agent
        if nxt_color == AGENT_COLOR_IDS[1 - agent_id]:
            return (self.observation, 0, False, info)

        # the content of the box is inaccessible while the box is locked
        if 2 < nxt_color < AGENT_COLOR_IDS[0] and 2 < nxt_right_color < AGENT_COLOR_IDS[0]:
            return (self.observation, 0, False, info)

        if 2 < nxt_color < AGENT_COLOR_IDS[0]:
            # pick up the content of the box(single pixel)
            if nxt_left_color < 3:
                if nxt_color == 3:
                    reward = 10
                    done = True
                    info['success'] = True
                    self._update_key(nxt_color)
                    self._agent_move(agent_id, next_agent_state)
                    # self.observation = self.reset()
                    return (self.observation, reward, done, info)
                else:
                    reward = 1
                    done = False
                    self._update_key(nxt_color)
                    self._agent_move(agent_id, next_agent_state)
                    return (self.observation, reward, done, info)

            # unlock the box
            elif self.key == nxt_color:
                if nxt_left_color in self.CorrectBox_lists:
                    reward = 1
                    done = False
                    self._update_key(0)
                    self._agent_move(agent_id, next_agent_state)
                    return (self.observation, reward, done, info)
                else:
                    reward = -1
                    done = True
                    self._update_key(0)
                    self._agent_move(agent_id, next_agent_state)
                    # self.observation = self.reset()
                    return (self.observation, reward, done, info)

            return (self.observation, 0, False, info)

        self._agent_move(agent_id, next_agent_state)
        return (self.observation, 0, False, info)

    def _update_key(self, nxt_color):
        self.key = nxt_color
        self.current_world_map[0, 0] = self.key

    def _agent_move(self, agent_id: int, next_agent_state):
        """
        moves specified agent
        """
        self.current_world_map[next_agent_state[0], next_agent_state[1]] = AGENT_COLOR_IDS[agent_id]
        self.current_world_map[self.agent_current_state[agent_id, 0], self.agent_current_state[agent_id, 1]] = 1

        self.agent_current_state[agent_id] = copy.deepcopy(next_agent_state)
        self.observation = self._worldmap_to_obervation(self.current_world_map)

    def _get_agent(self, agent_id, world_map):
        """
        returns location of specified agent
        """
        location = np.where(world_map == AGENT_COLOR_IDS[agent_id])
        agent = list(map(lambda x: x[0], location))
        return agent

    def get_current_agent_position(self):
        # from 2d(x,y) to 1d
        # returns the 1d positions of both agents
        position = self.agent_current_state[:, 0] * self.box_size + self.agent_current_state[:, 1]
        return position

    def _read_world_map(self, path):
        with open(path, 'r') as f:
            world_map = f.readlines()
            # world_map_splited = list(map(lambda x: x.split(' '), world_map))
            # world_map_array = [list(map(lambda x: int(x), row)) for row in world_map_splited]
            # world_map_array = np.array(world_map_array)
            # equal to the next one line
            world_map_array = np.array(list(map(lambda x: list(map(lambda y: int(y), x.split(' '))), world_map)))

            return world_map_array

    def _worldmap_to_obervation(self, world_map):
        obs_shape = self.obs_shape
        observation = np.zeros(obs_shape, dtype=np.float32)
        gs0 = int(observation.shape[0] / world_map.shape[0])
        gs1 = int(observation.shape[1] / world_map.shape[1])
        for i in range(world_map.shape[0]):
            for j in range(world_map.shape[1]):
                observation[i * gs0:(i + 1) * gs0, j * gs1:(j + 1) * gs1] = np.array(COLORS[world_map[i, j]])
        return observation

    def render(self, model='human'):
        img = self.observation
        fig = plt.figure(0)
        plt.clf()
        plt.imshow(img / 255)
        fig.canvas.draw()
        plt.pause(0.0001)

    def get_action_meanings(self):
        return [ACTION_MEANING[i] for i in self._action_set]


if __name__ == '__main__':
    env = BoxWorldMAEnv()
    ob = env.observation
    print(ob.shape)
    # ob2 = np.mean(ob, axis=2)
    env.render()

    while True:
        actions = [-1, -1]
        while all(action not in list(range(5)) for action in actions):
            print(f"Actions: {list(enumerate(env.get_action_meanings()))}")
            actions = input("Action 0~3 input for both agents:")
            actions = [int(a) for a in actions.split()]

        observation, reward, done, info = env.step(actions)
        print(f"state: {env.agent_current_state}")
        print(f"position: {env.get_current_agent_position()}")
        # observation, reward, done, info = env.step(env.action_space.sample())
        print(f"rewards: {reward}, done: {done}, info: {info}")
        env.render()

    # location = np.where(env.init_world_map == 4)
    # print(type(location))
