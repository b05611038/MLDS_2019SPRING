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
        self.gamma = gamma
        self.eps = 10e-7

        self.data = []
        self.rewards = []

    def reset_maximum(self, new):
        self.maximum = new
        return None

    def insert(self, observation, next_observation, action):
        if len(self.data) > self.maximum:
            self.data = self.data[1: ]
            self.rewards = self.rewards[1: ]

        self.data.append([observation, next_observation, action])

        return None

    def insert_reward(self, reward, times, done):
        if self.preprocess_dict['time_decay']:
            decay_reward = (reward * (np.power(self.gamma, np.flip(np.arange(times))) / \
                    np.sum(np.power(self.gamma, np.flip(np.arange(times)))))).tolist()

            self.rewards[len(self.rewards): ] = decay_reward
        else:
            normal_reward = (reward * np.repeat(1.0, times) / np.sum(np.repeat(1.0, times))).tolist()
            self.rewards[len(self.rewards): ] = normal_reward

        return None

    def trainable(self):
        #check the buffer is ready for training
        return True if len(self.rewards) > (self.maximum // 4) else False

    def __getitem__(self, index):
        select = random.randint(0, len(self.rewards) - 1)
        return self.data[select][0].squeeze(0).float().detach(), self.data[select][1].squeeze(0).float().detach(), \
                torch.tensor(self.data[select][2]).long(), torch.tensor(self.rewards[select]).float()

    def __len__(self):
        return self.maximum // 8


