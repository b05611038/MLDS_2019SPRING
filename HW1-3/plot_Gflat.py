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
from lib.model import SenCNN
from lib.visualize import GenSenPlot

def LoadModel(model_name):
    return torch.load(model_name, map_location = 'cpu')

def ModelExchange(model_one, model_two, alpha):
    para_one = model_one.named_parameters()
    para_two = model_two.named_parameters()

    dict_para = dict(para_two)

    for name_one, parameters in para_one:
        if name_one in dict_para:
            dict_para[name_one].data.copy_(alpha * parameters.data + (1 - alpha) * dict_para[name_one].data)

    model = SenCNN()
    return model.load_state_dict(dict_para)    

def Eval(model_one, model_two, point_num):
    model_one = LoadModel(model_one)
    model_two = LoadModel(model_two)

    criterion = nn.CrossEntropyLoss()
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train_set = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
    test_set = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

    train_loader = DataLoader(train_set, batch_size = 1024, shuffle = False)
    test_loader = DataLoader(test_set, batch_size = 1024, shuffle = False)

    alpha = np.linspace(-1.5, 2.5, point_num)
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
            temp_loss = torch.mul(temp_loss, temp_total).sum().numpy()
            train_loss.append((temp_loss / temp_total.sum()).numpy())
            train_acc.append(temp_correct / temp_total.sum().numpy())

            print('Alpha:', i, '| Train Loss: %.6f' % (temp_loss / temp_total.sum()).numpy(),
                    '| Test Acc.: %.6f' % temp_correct / temp_total.sum().numpy())

            temp_correct = 0
            temp_total = []
            temp_loss = []
            for iter, data in enumerate(test_loader):
                test_x, test_y = data
                test_x = test_x.float()
                test_y = test_y.long()

                test_out = model(test_x)
                _, prediction = torch.max(test_out.data, 1)
                temp_total.append(test_y.size(0))
                temp_correct += (prediction == test_y).sum().item()

                temp_loss.append(criterion(test_out, y).detach())

            temp_loss = torch.tensor(temp_loss)
            temp_loss = torch.mul(temp_loss, temp_total).sum().numpy()
            test_loss.append((temp_loss / temp_total.sum()).numpy())
            test_acc.append(temp_correct / temp_total.sum().numpy())

            print('Alpha:', i, '| Test Loss: %.6f' % (temp_loss / temp_total.sum()).numpy(),
                    '| Test Acc.: %.6f' % temp_correct / temp_total.sum().numpy())

    return alpha, np.array(train_loss), np.array(train_acc), np.array(test_loss), np.array(test_acc)

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: python3 plot_Gflat.py [image name] [point num] [model 1 path] [model 2 path]')
        exit(0)

    start_time = time.time()
    alpha, train_loss, train_acc, test_loss, test_acc = Eval(sys.argv[3], sys.argv[4], int(sys.argv[2]))

    GenSenPlot([alpha, [train_loss, test_loss], [train_acc, test_acc]], [['train', 'test'], ['train', 'test']],
            sys.argv[1], 'Flatness vs Generalization', ['alpha', 'cross_entropy', 'Accuracy'])    

    print('All process done, cause %s seconds.' % (time.time() - start_time))


