import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from lib.utils import *

class ReplayBuffer(object):
    def __init__(self, env, maximum, time_decay, normalized, gamma = 0.99):
        #only note the game environment
        #not the enviroment object
        self.env = env
        self.maximum = maximum
        self.time_decay = time_decay
        self.gamma = gamma
        self.normalized = normalized

        #one elemnet in datalist is a training pair with three elements: observation, reward, action
        #the pair relationship -> model(observation) ==> action ==> reward
        self.data = []
        self.rewards = []
        self.__insert_lock = []

    def reset_maximum(self, new):
        self.maximum = new
        return None

    def new_episode(self):
        if len(self.data) >= self.maximum:
            self.data = self.data[1: ]
            self.__insert_lock = self.insert_lock[1: ]

        self.data.append([])
        self.rewards.append([])
        self.__insert_lock.append(False)
        return None

    def insert(self, observation, action, reward = None):
        if self.__insert_lock[-1] != True:
            #not lock can append
            pair = [observation, action]
            self.data[-1].append(pair)
            if reward is not None:
                self.rewards.append(reward)
                self.__insert_lock[-1] = True
        else:
            raise RuntimeError('Please use new_episode() before insert new episode information.')

        return None

    def trainable(self):
        #check the buffer is ready for training
        return False if len(self.rewards) < self.maximum else True

    def getitem(self, episode_size):
        observation = None
        action = None
        reward = None
        for i in range(episode_size):
            select = random.randint(0, self.maximum - 1)
            dataset = EpisodeSet(self.data[select])
            rew = torch.tensor([self.rewards[select]])
            rew = rew.repeat([len(self.data[select])])
            dataloader = DataLoader(dataset, batch_size = len(self.data[select]), shuffle = False)
            for iter, (obs, act) in enumerate(dataloader):
                if observation is None:
                    observation = obs
                else:
                    observation = torch.cat((observation, obs), dim = 0)

                if action is None:
                    action = act
                else:
                    action = torch.cat((action, act), dim = 0)

            if self.time_decay:
                time_step = torch.arange(rew.size(0))
                time_step = torch.flip(time_step, dims = [0]).float()
                decay = torch.pow(self.gamma, time_step)
                rew = torch.mul(decay, rew)

            if reward is None:
                reward = rew
            else:
                rew = torch.cat((reward, rew), dim = 0)

        if self.normalized:
            mean = torch.mean(reward, dim = 0)
            std = torch.std(reward, dim = 0)
            reward = (reward - mean) / std

        return observation, action, reward


class EpisodeSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        #return observation, action, reward
        return data[index][0], torch.tensor(data[index][1]).float()

    def __len__(self):
        return len(self.data)


