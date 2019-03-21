import os
import sys
import time
import copy
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.utils import *
from lib.visualize import GenSenPlot

def LoadModel(model_name):
    return torch.load(model_name, map_location = 'cpu')

def ModelExchange(model_one, model_two, alpha):
    model = copy.deepcopy(model_one)
    for index, layer in enumerate(model.parameters()):
        for index_one, layer_one in enumerate(model_one.parameters()):
            for index_two, layer_two in enumerate(model_two.parameters()):
                if index_one == index_two:
                    layer.data = layer_one.data * alpha + layer_two.data * (1 - alpha)
                else:
                    pass

    return model    

def Eval(model_one, model_two):
    model_one = LoadModel(model_one)
    model_two = LoadModel(model_two)

    criterion = nn.CrossEntropyLoss()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
    test_set = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

    train_loader = DataLoader(train_set, batch_size = 1024, shuffle = False)
    test_loader = DataLoader(test_set, batch_size = 1024, shuffle = False)

    alpha = np.linspace(-1.5, 2.5, 50)
    train_acc = []
    train_loss = []
    test_acc = []
    test_loss = []
    for i in range(alpha.shape[0]):
        model = ModelExchange(model_one, model_two, alpha[i])
        model.float().eval()
        with torch.no_grad():
            temp_correct = 0
            temp_total = []
            temp_loss = []
            for iter, data in enumerate(train_loader):
                train_x, train_y = data
                train_x = train_x.float()
                train_y = train_y.long()

                train_out = model(train_x)
                _, prediction = torch.max(train_out.data, 1)
                temp_total.append(train_y.size(0))
                temp_correct += (prediction == train_y).sum().item()

                temp_loss.append(criterion(train_out, train_y).detach())

            temp_loss = torch.tensor(temp_loss)
            train_loss = torch.sum(temp_loss).
                
                

if __name__ == '__main__':
    model_one = LoadModel(sys.argv[1])
    model_two = LoadModel(sys.argv[2])
    model = ModelExchange(model_one, model_two, 0.5)
