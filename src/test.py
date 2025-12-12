import gym
from gym import spaces

class Test(gym.Env):

    def __init__(self):

        self.observation_space = spaces.Box(0, 10, shape=(2,), dtype=int)
        self.action_space = spaces.Discrete(3)
        self.timestep = 0
        self.done = False

    def _get_obs(self):
        return [1,2]


    def reset(self):
        # We need the following line to seed self.np_random
        # super().reset(seed=seed)

        # Choose the agent's location uniformly at random

        self.timestep = 0
        self.done = False

        return [1,2]

    def step(self, action):
        self.timestep +=1

        if action == 0:
            reward = 10
        else:
            reward = 2
        if (self.timestep == 5):
            self.done = True

        return [3,4], reward, self.done, {}

    def render(self):
        pass




