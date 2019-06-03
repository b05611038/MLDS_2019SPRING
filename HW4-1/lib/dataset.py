import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from lib.utils import *

class ReplayBuffer(object):
    def __init__(self, env, maximum, preprocess_dict, gamma = 0.99):
        #only note the game environment
        #not the enviroment object
        self.env = env
        self.maximum = maximum
        self.preprocess_dict = preprocess_dict
        self.gamma = gamma
        self.eps = 10e-7

        #one elemnet in datalist is a training pair with three elements: observation, reward, action
        #the pair relationship -> model(observation) ==> action ==> reward
        self.data = []
        self.rewards = []
        self.__insert_lock = []

    def reset_maximum(self, new):
        self.maximum = new
        return None

    def new_episode(self):
        if len(self.rewards) > self.maximum:
            self.data = self.data[1: ]
            self.rewards = self.rewards[1: ]
            self.__insert_lock = self.__insert_lock[1: ]

        self.data.append([])
        self.rewards.append([])
        self.__insert_lock.append(False)
        return None

    def insert(self, observation, action):
        if self.__insert_lock[-1] != True:
            #not lock can append
            self.data[-1].append([observation.squeeze(), action])
        else:
            raise RuntimeError('Please use new_episode() before insert new episode information.')

        return None

    def insert_reward(self, reward, times, done):
        if self.__insert_lock[-1] != True:
            for i in range(times):
                if self.preprocess_dict['time_decay']:
                    decay_reward = reward * math.pow((self.gamma), (times - 1 - i))
                    self.rewards[-1].append(decay_reward)
                else:
                    self.rewards[-1].append(reward)

        else:
            raise RuntimeError('Please use new_episode() before insert new episode information.')

        if done:
            self.__insert_lock[-1] = True

        return None

    def trainable(self):
        #check the buffer is ready for training
        return True if len(self.rewards) >= self.maximum else False

    def getitem(self, episode_size):
        observation = None
        action = None
        reward = None
        for i in range(episode_size):
            select = random.randint(0, self.maximum - 1)
            dataset = EpisodeSet(self.data[select], self.rewards[select])
            dataloader = DataLoader(dataset, batch_size = len(self.data[select]), shuffle = False)
            for iter, (obs, act, rew) in enumerate(dataloader):
                if observation is None:
                    observation = obs.squeeze()
                else:
                    observation = torch.cat((observation, obs.squeeze()), dim = 0)

                if action is None:
                    action = act.squeeze()
                else:
                    action = torch.cat((action, act.squeeze()), dim = 0)

                if reward is None:
                    reward = rew
                else:
                    reward = torch.cat((reward, rew), dim = 0)

        if self.preprocess_dict['normalized']:
            mean = torch.mean(reward, dim = 0)
            std = torch.std(reward, dim = 0)
            reward = (reward - mean) / (std + self.eps)

        return observation.detach(), action.detach(), reward.detach()


class EpisodeSet(Dataset):
    def __init__(self, data, rewards):
        self.data = data
        self.rewards = rewards

    def __getitem__(self, index):
        #return observation, action
        reward = torch.tensor(self.rewards[index]).float()
        return self.data[index][0].float(), self.data[index][1].float(), reward

    def __len__(self):
        return len(self.data)


