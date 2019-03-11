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
from lib.utils import SampleFunctionDataset
#--------------------------------------------------------------------------------
#train_FT.py is to train the feed forward nerual network for function fitting in
#2019 MLDS HW1-2
#--------------------------------------------------------------------------------
def Acquire_grad_norm(model):
    grad_all = 0
    for p in model.parameters():
        grad = 0
        if p.grad is not None:
            grad = (p.grad.cpu().data.numpy() ** 2).sum()

        grad_all += grad

    return float(grad_all ** 0.5)

def TrainModel(model, saving_name, dataset, criterion, interval, device, save = True):
    if device < 0:
        env = torch.device('cpu')
        print('Envirnment setting done, using device: cpu')
    else:
        torch.backends.cudnn.benchmark = True
        cuda.set_device(device)
        env = torch.device('cuda:' + str(device))
        print('Envirnment setting done, using device: CUDA_' + str(device))

    model.float().to(env)
    criterion.to(env)
    optim = Adam(model.parameters())

    train_set = dataset
    train_set.mode = 'train'
    test_set = copy.deepcopy(train_set)
    test_set.mode = 'test'

    train_loader = DataLoader(train_set, batch_size = interval, shuffle = False)
    test_loader = DataLoader(test_set, batch_size = test_set.get_num(), shuffle = False)

    print('Model Structure:')
    print(model)
    print('Model Parameter numbers: ',sum(p.numel() for p in model.parameters() if p.requires_grad))

    history = ['Iteration,Grad_norm,Train loss,Test loss\n']
    for iter, data in enumerate(train_loader):
        train_x, train_y = data
        train_x = train_x.to(env)
        train_y = train_y.to(env)

        optim.zero_grad()

        out = model(train_x)

        loss = criterion(out, train_y)
        loss.backward()

        optim.step()

        train_loss = loss.detach() 
        grad_norm = Acquire_grad_norm(model)
        with torch.no_grad():
            for _iter, test_data in enumerate(test_loader):
                test_x, test_y = test_data
                test_x = test_x.to(env)
                test_y = test_y.to(env)

                test_out = model(test_x)

                test_loss = criterion(test_out, test_y)

            print('iter:', (iter + 1) * interval, '| grad norm: %6f' % grad_norm, '| train_loss: %6f' % train_loss, '| test_loss: %6f' % test_loss)
            history.append(str((iter + 1) * interval) + ',' + str(grad_norm) + ',' + str(float(train_loss.detach())) + ',' + str(float(test_loss.detach())) + '\n')
            train_loss = 0
            if save:
                torch.save(model, saving_name + '_I' + str((iter + 1) * interval) + '.pkl')

    print('Training process done.\nStart recording all history...')
    f = open(saving_name + '.csv', 'w')
    f.writelines(history)
    f.close()

    if save:
        torch.save(model, saving_name + '.pkl')

    print('History file saving done.')


def target_function(x):
    y = 4 * np.sin(np.exp(x / 8)) - 5 * np.cos(x) + np.exp(x / 8)

    return y

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python3 train_FT.py [model name] [sample point] [save interval] [device]')
        exit(0)

    start_time = time.time()
    model = ANN()
    dataset = SampleFunctionDataset(0.1, target_function, [0, 20], int(sys.argv[2]))
    TrainModel(model, sys.argv[1], dataset, nn.MSELoss(), int(sys.argv[3]), int(sys.argv[4]))
    print('All process done, cause %s seconds.' % (time.time() - start_time))