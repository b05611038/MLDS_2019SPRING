import pickle
import numpy as np
import torch

def save_object(fname, obj):
    #the function is used to save some data in class or object in .pkl file
    with open(fname, 'wb') as out_file:
        pickle.dump(obj, out_file)
    out_file.close()

def load_object(fname):
    #the function is used to read the data in .pkl file
    with open(fname, 'rb') as in_file:
        return pickle.load(in_file)

class RandomLabel():
    #to retain the same training label
    def __init__(self, ratio, mode = 'create'):
        self.ratio = ratio
        self.mode = mode
        self.label = []

    def grab(self, index):
        if index > len(self.label):
            raise IndexError('Please check the iteration of dataloader.')

        self.mode = 'stop'
        return self.label[index]

    def generate(self, data):
        if self.mode == 'create':
            new = self._random_sample(data)
            self.label.append(new)

            return new
        else:
            raise RuntimeError('Please check the RandomLabel api usage.')

    def _random_sample(self, data):
        size = int(data.size(0) * self.ratio)
        rand_arr = torch.randint(0, 10, data.size())
        _, index = torch.rand(data.size(0)).sort()
        for i in range(size):
            data[index[i]] = rand_arr[index[i]]

        return data


