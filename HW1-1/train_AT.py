import sys
import csv
import time
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from lib.model import ANN_cifar10, CNN
#--------------------------------------------------------------------------------
#train_AT is to test the performance of different deep learning model whcih has 
#same amount of parameters but different depth
#--------------------------------------------------------------------------------
def TrainModel(model, saving_name, criterion, epochs, batch_size, device, save = True):
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

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
    train_eval_set = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
    test_set = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = False)
    train_eval_loader = DataLoader(train_eval_set, batch_size = batch_size, shuffle = False)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)

    print('Model Structure:')
    print(model)

    history = ['Epoch,Train Loss,Train Acc.,Test Loss,Test Acc.\n']
    for epoch in range(epochs):
        print('Start training epoch: ', epoch + 1)
        for iter, data in enumerate(train_loader):
            train_x, train_y = data
            train_x = train_x.float().to(env)
            train_y = train_y.long().to(env)

            optim.zero_grad()

            out = model(train_x)

            loss = criterion(out, train_y)
            loss.backward()

            optim.step()

            if iter != 0 and iter % 10 == 0:
                print('Iter: ', iter, ' | Loss: %6f' % loss.detach())

        train_loss = []
        train_total = []
        train_correct = 0
        with torch.no_grad():
            for _iter, train_data in enumerate(train_eval_loader):
                train_x, train_y = train_data
                train_x = train_x.float().to(env)
                train_y = train_y.long().to(env)

                train_out = model(train_x)
                _, prediction = torch.max(train_out.data, 1)
                train_total.append(train_y.size(0))
                train_correct += (prediction == train_y).sum().item()

                train_loss.append(criterion(train_out, train_y).detach())

        train_loss = torch.tensor(train_loss)
        train_total = torch.tensor(train_total)

        test_loss = []
        test_total = []
        test_correct = 0
        with torch.no_grad():
            for _iter, test_data in enumerate(test_loader):
                test_x, test_y = test_data
                test_x = test_x.float().to(env)
                test_y = test_y.long().to(env)

                test_out = model(test_x)
                _, prediction = torch.max(test_out.data, 1)
                test_total.append(test_y.size(0))
                test_correct += (prediction == test_y).sum().item()

                test_loss.append(criterion(test_out, test_y).detach())

        test_loss = torch.tensor(test_loss)
        test_total = torch.tensor(test_total)

        train_loss = (torch.sum(torch.mul(train_loss, train_total.float())) / torch.sum(train_total)).detach()
        train_acc = (100 * train_correct / torch.sum(train_total)).detach()
        test_loss = (torch.sum(torch.mul(test_loss, test_total.float())) / torch.sum(test_total)).detach()
        test_acc = (100 * test_correct / torch.sum(test_total)).detach()

        history.append(str(epoch + 1) + ',' + str(train_loss) + ',' + str(train_acc) + ','
                + str(test_loss) + ',' + str(test_acc) + '\n')
        print('\nEpoch: ', epoch + 1, '| Train loss: %6f' % train_loss, '| Train Acc.: %2f' % train_acc,
                '| Test loss: %6f' % test_loss, '| Test Acc.: %2f' % test_acc, '\n')

    f = open(saving_name + '.csv', 'w')
    f.writelines(history)
    f.close()

    if save:
        torch.save(model, saving_name + '.pkl')

    print('All process done.')

if __name__ == '__main__':
    if len(sys.argv) < 7:
        print('Usage: python3 train_AT.py [CNN or ANN] [model depth] [model unit] [model name] [epochs] [batch size] [device]')
        exit(0)

    start_time = time.time()
    if sys.argv[1] == 'ANN':
        model = ANN_cifar10(depth = int(sys.argv[2]), unit = int(sys.argv[3]))
    elif sys.argv[1] == 'CNN':
        model = CNN(depth =  int(sys.argv[2]), channel = int(sys.argv[3]))
    else:
        print('Please check the model selection args.')
        exit(0)

    TrainModel(model, sys.argv[4], nn.CrossEntropyLoss(), int(sys.argv[5]), int(sys.argv[6]), int(sys.argv[7]))
    print('All process done, cause %s seconds.' % (time.time() - start_time))


