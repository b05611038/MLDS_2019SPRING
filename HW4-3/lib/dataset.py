import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

from lib.utils import *


class ReplayBuffer(Dataset):
    def __init__(self, env, maximum, preprocess_dict, gamma = 0.99):
        #only note the game environment
        #not the enviroment object
        self.env = env
        self.maximum = maximum
        self.preprocess_dict = preprocess_dict
        self.length = 0
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

    def insert(self, observation, next_observation, action):
        if self.__insert_lock[-1] != True:
            #not lock can append
            self.episode_data[-1].append([observation, next_observation, action])
        else:
            raise RuntimeError('Please use new_episode() before insert new episode information.')

        return None

    def insert_reward(self, reward, times, done):
        if self.__insert_lock[-1] != True:
            if self.preprocess_dict['time_decay']:
                decay_reward = (reward * (np.power(self.gamma, np.flip(np.arange(times))) / \
                        np.sum(np.power(self.gamma, np.flip(np.arange(times)))))).tolist()

                self.rewards[-1][len(self.rewards[-1]): ] = decay_reward
            else:
                normal_reward = (reward * np.repeat(1.0, times) / np.sum(np.repeat(1.0, times))).tolist()
                self.rewards[-1][len(self.rewards[-1]): ] = normal_reward

        else:
            raise RuntimeError('Please use new_episode() before insert new episode information.')

        if done:
            self.__insert_lock[-1] = True

        return None

    def trainable(self):
        #check the buffer is ready for training
        return True if len(self.rewards) >= self.maximum else False

    def make(self, episode_size):
        self.observation = None
        self.next_observation = None
        self.action = None
        self.reward = None
        for i in range(episode_size):
            select = random.randint(0, self.maximum - 1)
            dataset = EpisodeSet(self.data[select], self.rewards[select])
            dataloader = DataLoader(dataset, batch_size = len(self.data[select]), shuffle = False)
            for iter, (obs, obs_next, act, rew) in enumerate(dataloader):
                if self.observation is None:
                    self.observation = obs.squeeze()
                else:
                    self.observation = torch.cat((self.observation, obs.squeeze()), dim = 0)

                if self.next_observation is None:
                    self.next_observation = next_observation
                else:
                    self.next_observation = torch.cat((self.next_observation, next_obs.squeeze()), dim = 0)

                if self.action is None:
                    self.action = act.squeeze()
                else:
                    self.action = torch.cat((self.action, act.squeeze()), dim = 0)

                if self.reward is None:
                    self.reward = rew
                else:
                    self.reward = torch.cat((self.reward, rew), dim = 0)

        if self.preprocess_dict['normalized']:
            mean = torch.mean(self.reward, dim = 0)
            std = torch.std(self.reward, dim = 0)
            self.reward = (self.reward - mean) / (std + self.eps)

        self.length = self.reward.size(0)
        return None

    def __getitem__(self, index):
        return self.observation[index].detach(), self.next_observation[index].detach(),
                self.action[index].detach(), self.reward[index].detach()

    def __len__(self):
        return self.length


class EpisodeSet(Dataset):
    def __init__(self, data, rewards):
        self.data = data
        self.rewards = rewards

    def __getitem__(self, index):
        #return observation, action
        reward = torch.tensor(self.rewards[index]).float()
        return self.data[index][0].float(), self.data[index][1].float(), self.data[index][2].long(), reward

    def __len__(self):
        return len(self.data)


