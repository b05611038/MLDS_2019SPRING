import sys
import csv
import time
import copy
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from lib.model import ANN
from lib.utils import *
#--------------------------------------------------------------------------------
#train_FT.py is to train the feed forward nerual network for function fitting in
#2019 MLDS HW1-2
#--------------------------------------------------------------------------------
def Grad_norm(model):
    grad_all = 0
    for p in model.parameters():
        grad = 0
        if p.grad is not None:
            grad = (p.grad.data ** 2).sum()

        grad_all += grad
        grad_all = grad_all ** 0.5
        grad_all.required_grad = True

    return grad_all

def TrainModel(saving_name, dataset, criterion, epochs, points, device, save = True):
    if device < 0:
        env = torch.device('cpu')
        print('Envirnment setting done, using device: cpu')
    else:
        torch.backends.cudnn.benchmark = True
        cuda.set_device(device)
        env = torch.device('cuda:' + str(device))
        print('Envirnment setting done, using device: CUDA_' + str(device))

    criterion.to(env)
    train_set = dataset
    train_set.mode = 'train'

    train_loader = DataLoader(train_set, batch_size = train_set.get_num(), shuffle = False)

    point = np.asarray([0, 0])
    #record high loss and small loss
    loss_list = []
    while (point[0] < points // 2 or point[1] < points // 2):
        model = ANN()
        model.float().to(env)
        optim = Adam(model.parameters())
        optim_grad = Adam(model.parameters(), lr = 0.0001)

        print('Model Structure:')
        print(model)

        for epoch in range(epochs[0]):
            for iter, data in enumerate(train_loader):
                train_x, train_y = data
                train_x = train_x.to(env)
                train_y = train_y.to(env)

                optim.zero_grad()

                out = model(train_x)

                loss = criterion(out, train_y)
                loss.backward()

                optim.step()

            train_loss = float(loss.cpu().detach().numpy())
            grad_norm = float(Grad_norm(model).cpu().detach().numpy())
            print('Model version:', point.sum() + 1, '| Epoch:', epoch + 1, '| Grad_norm: %6f' % grad_norm, '| Train loss: %6f' % train_loss)

        print('Changing loss function, continue training...')
        for epoch in range(epochs[1]):
            for iter, data in enumerate(train_loader):
                train_x, train_y = data
                train_x = train_x.to(env)
                train_y = train_y.to(env)

                optim_grad.zero_grad()

                out = model(train_x)
 
                norm_loss = criterion(out, train_y)
                grad_all = torch.autograd.grad(norm_loss, model.parameters(), create_graph = True)
                grad_norm = torch.zeros((1, 1)).to(env)
                for i, g in enumerate(grad_all):
                    grad_norm += (g ** 2).sum()

                grad_norm = grad_norm ** 0.5
                grad_norm.backward()

                optim_grad.step()

            norm_loss = float(norm_loss.cpu().detach().numpy())
            grad_norm = float(grad_norm.cpu().detach().numpy())
            print('Model version:', point.sum() + 1, '| Grad epoch:', epoch + 1, '| Grad_norm: %6f' % grad_norm, '| Grad norm loss: %6f' % norm_loss)

            if grad_norm < 0.5 and point[0] <= points // 2:
                loss_list.append(norm_loss)
                print('Save loss: %6f' % norm_loss)
                torch.save(model.state_dict(), saving_name + '_paras_ver' + str(point.sum() + 1) + '.pkl')
                point[0] += 1
                break
            elif grad_norm >= 0.5 and grad_norm < 1 and point[1] <= points // 2:
                loss_list.append(norm_loss)
                print('Save loss: %6f' % norm_loss)
                torch.save(model.state_dict(), saving_name + '_paras_ver' + str(point.sum() + 1) + '.pkl')
                point[1] += 1
                break

        print('Model version', point.sum(), 'saving done.')

    loss_list = np.asarray(loss_list)
    np.save('loss.npy', loss_list)
    save_object(saving_name + '_dataset.pkl', train_set)
    print('All training process done.')

def target_function(x):
    y = 4 * np.sin(np.exp(x / 8)) - 5 * np.cos(x) + np.exp(x / 8)
    return y

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: python3 train_NZ.py [model name] [normal epoch] [grad epoch] [record points] [device]')
        exit(0)

    start_time = time.time()
    dataset = SampleFunctionDataset(0.1, target_function, [0, 20], 100000)
    TrainModel(sys.argv[1], dataset, nn.MSELoss(), [int(sys.argv[2]), int(sys.argv[3])], int(sys.argv[4]), int(sys.argv[5]))
    print('All process done, cause %s seconds.' % (time.time() - start_time))

