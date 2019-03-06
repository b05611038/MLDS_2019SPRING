import math
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class SampleFunctionDataset(Dataset):
    def __init__(self, val_split, func, domain, data_num, mode = None):
        self.mode = mode
        self.val_split = val_split
        self.func = func
        #the domain range of function
        self.domain = domain
        self.data_num = data_num

        data, label = self._generate_data(self.func, self.domain, self.data_num)
        self.train_x, self.test_x, self.train_y, self.test_y = self._split_data(data, label, val_split)

    def _generate_data(self, func, domain, data_num):
        data = np.random.random_sample((data_num, 1))
        data = (domain[1] - domain[0]) * data + domain[0]
        label = func(data)

        return data, label

    def _split_data(self, data, label, val_split):
        self.seed = random.randint(0, 10000)
        train_x, test_x, train_y, test_y = train_test_split(data, label, test_size = val_split, random_state = self.seed)

        return train_x, test_x, train_y, test_y

    def __getitem__(self, index):
        if self.mode == 'train':
            train_data = torch.tensor(self.train_x[index])
            train_label = torch.tensor(self.train_y[index])

            return train_data.float(), train_label.float()

        elif self.mode == 'test':
            test_data = torch.tensor(self.test_x[index])
            test_label = torch.tensor(self.test_y[index])

            return test_data.float(), test_label.float()

        else:
            raise RuntimeError('Please check the mode of dataset.')

    def __len__(self):
        if self.mode == 'train':
            return self.train_y.shape[0]
        elif self.mode == 'test':
            return self.test_y.shape[0]
        else:
            raise RuntimeError('Please check the mode of dataset.')

    def get_num(self):
        if self.mode == 'train':
            return self.train_y.shape[0]
        elif self.mode == 'test':
            return self.test_y.shape[0]
        else:
            raise RuntimeError('Please check the mode of dataset.')


