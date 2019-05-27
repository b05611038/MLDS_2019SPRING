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
        self.__insert_lock.append(False)
        return None

    def insert(self, observation, action, reward = None):
        if self.__insert_lock[-1] != True:
            #not lock can append
            self.data[-1].append([observation.squeeze(), action])
            if reward is not None:
                self.rewards.append(reward)
                self.__insert_lock[-1] = True
        else:
            raise RuntimeError('Please use new_episode() before insert new episode information.')

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
            dataset = EpisodeSet(self.data[select])
            rew = torch.tensor([self.rewards[select]])
            rew = rew.repeat([len(self.data[select])])
            dataloader = DataLoader(dataset, batch_size = len(self.data[select]), shuffle = False)
            for iter, (obs, act) in enumerate(dataloader):
                if observation is None:
                    observation = obs.squeeze()
                else:
                    observation = torch.cat((observation, obs.squeeze()), dim = 0)

                if action is None:
                    action = act.squeeze()
                else:
                    action = torch.cat((action, act.squeeze()), dim = 0)

            if self.preprocess_dict['time_decay']:
                time_step = torch.arange(rew.size(0))
                time_step = torch.flip(time_step, dims = [0]).float()
                decay = torch.pow(self.gamma, time_step)
                rew = torch.mul(decay, rew)

            if reward is None:
                reward = rew
            else:
                reward = torch.cat((reward, rew), dim = 0)

        if self.preprocess_dict['normalized']:
            mean = torch.mean(reward, dim = 0)
            std = torch.std(reward, dim = 0)
            reward = (reward - mean) / std

        return observation, action, reward


class EpisodeSet(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        #return observation, action
        return self.data[index][0].float(), self.data[index][1].float()

    def __len__(self):
        return len(self.data)

