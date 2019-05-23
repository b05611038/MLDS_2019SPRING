import random
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.utils import *

class ReplayBuffer(Dataset):
    def __init__(self, env, maximum, transform = None):
        #only note the game environment
        #not the enviroment object
        self.env = env
        self.maximum = maximum
        if transform is None:
            raise NotImplementedError('Please init trochvision.transform.Compose() in transform args.')
        self.transform = transform

        #one elemnet in datalist is a training pair with three elements: observation, reward, action
        #the pair relationship -> model(observation) ==> action ==> reward
        self.data = []

    def insert(self, observation, action, reward):
        pair = [observation, action, reward]
        if len(self.data) >= self.maximum:
            self.data = self.data[1: ]

        self.data.append(pair)
        return None

    def trainable(self):
        #check the buffer is ready for training
        return False if len(self.data) < self.maximum else True

    def __getitem__(self, index):
        #return
        select = random.randint(0, self.maximum - 1)
        select_pair = self.data[select]
        return self.transform(select_pair[0]), torch.tensor([select_pair[1]]), torch.tensor([select_pair[2]])

    def __len__(self);
        return self.maximum


