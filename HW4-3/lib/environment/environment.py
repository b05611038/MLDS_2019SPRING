import gym
import numpy as np

from lib.utils import *
from lib.environment.wrapper import make_wrap_atari


class TestEnvironment(object):
    def __init__(self, env_name, test = True):
        self.env = gym.make(env_name)

        self.env_name = env_name
        self.lives = 0
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def seed(self, seed):
        self.env.seed(seed)

    def reset(self):
        observation = self.env.reset()
        return np.array(observation)

    def step(self, action):
        if not self.env.action_space.contains(action):
            raise ValueError('Ivalid action!!')

        observation, reward, done, info = self.env.step(action)
        true_done = True if done else False

        if self.env_name == 'Breakout-v0':
            lives = self.env.unwrapped.ale.lives()
            if lives < self.lives and lives > 0:
                done = True

            self.lives = lives
        elif self.env_name == 'Pong-v0':
            pass

        return np.array(observation).astype('uint8'), reward, done, true_done, info

    def render(self):
        return self.env.render(mode = 'rgb_array')

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_random_action(self):
        return self.action_space.sample()


class Environment(object):
    def __init__(self, env_name, args, atari_wrapper = False, test = False):
        if atari_wrapper:
            clip_rewards = not test
            self.env = make_wrap_atari(env_name, clip_rewards)
        else:
            self.env = gym.make(env_name)

        self.env_name = env_name
        self.lives = 0
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        
    def seed(self, seed):
        '''
        Control the randomness of the environment
        '''
        self.env.seed(seed)

    def reset(self):
        '''
        When running dqn:
            observation: np.array
                stack 4 last frames, shape: (84, 84, 4)
        When running pg:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
        '''
        observation = self.env.reset()

        return np.array(observation)

    def step(self, action):
        '''
        When running dqn:
            observation: np.array
                stack 4 last preprocessed frames, shape: (84, 84, 4)
            reward: int
                wrapper clips the reward to {-1, 0, 1} by its sign
                we don't clip the reward when testing
            done: bool
                whether reach the end of the episode?
        When running pg:
            observation: np.array
                current RGB screen of game, shape: (210, 160, 3)
            reward: int
                if opponent wins, reward = +1 else -1
            done: bool
                whether reach the end of the episode?
        '''
        if not self.env.action_space.contains(action):
            raise ValueError('Ivalid action!!')

        observation, reward, done, info = self.env.step(action)

        if self.env_name == 'Breakout-v0':
            lives = self.env.unwrapped.ale.lives()
            if lives < self.lives and lives > 0:
                done = True

            self.lives = lives
        elif self.env_name == 'Pong-v0':
            pass

        return np.array(observation).astype('uint8'), reward, done, info

    def render(self):
        return self.env.render(mode = 'rgb_array')

    def get_action_space(self):
        return self.action_space

    def get_observation_space(self):
        return self.observation_space

    def get_random_action(self):
        return self.action_space.sample()

