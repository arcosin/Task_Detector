
import random

import gym
from gym.envs.atari.atari_env import AtariEnv




'''
    An OpenAI Gym wrapper for multiple atari environments.
    Can reset to a specified environment or a random environment from the list.
        envs -- An array of the names of the environments to be used.
'''
class MultiEnv():
    def __init__(self, envs):
        self.envs = [AtariEnv(game = e, obs_type = "image", frameskip = 1, full_action_space = True) for e in envs]
        self.currEnv = 0
        self.actSpace = 18
        self.obsSpace = (210, 160, 3)
        self.iter = 0

    def getCurrEnv(self):
        return self.envs[currEnv]

    def getEnvList(self):
        return self.envs

    def reset(self, envNumber = None):
        if envNumber is not None:
            if envNumber >= 0 and envNumber < len(self.envs):
                raise ValueError("[MultiEnv]:  environment out of bounds.")
            env = envNumber
        else:
            env = random.choice(self.envs)
        self.currEnv = env
        return self.envs[env].reset()

    def render(self, mode = 'human'):
        return self.envs[self.currEnv].render(mode = mode)

    def step(self, action):
        return self.envs[self.currEnv].step(action)

    def seed(self, seed = None):
        for env in self.envs:
            env.seed(seed)

    def close(self):
        for env in self.envs:
            env.close()

    def get_action_meanings(self):
        return self.envs[self.currEnv].get_action_meanings()

    def get_keys_to_action(self):
        return self.envs[self.currEnv].get_keys_to_action()

    def clone_state(self):
        return self.envs[self.currEnv].clone_state()

    def clone_full_state(self):
        return self.envs[self.currEnv].clone_full_state()

#===============================================================================
