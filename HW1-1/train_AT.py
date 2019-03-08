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

from lib.model import ANN, CNN
#--------------------------------------------------------------------------------
#train_AT is to test the performance of different deep learning model whcih has 
#same amount of parameters but different depth
#--------------------------------------------------------------------------------
def TrainModel(model, saving_name, criterion, epochs, device, save = False):
    if device < 0:
        env = torch.device('cpu')
    else:
        torch.backends.cudnn.benchmark = True
        cuda.set_device(device)
        env = torch.device('cuda:' + str(device))

    print('Envirnment setting done, using device: ' + str(torch.device))

    model.float().to(env)
    criterion.to(env)
    optim = Adam(model.parameters())

    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_set = torchvision.datasets.CIFAR10(root = './data', train = True, download = True, transform = transform)
    test_set = torchvision.datasets.CIFAR10(root = './data', train = False, download = True, transform = transform)

    train_loader = DataLoader(train_set, batch_size = 100, shuffle = False)
    test_loader = DataLoader(test_set, batch_size = 100, shuffle = False)

    print('Model Structure:')
    print(model)

    history = ['Epoch,Loss,Acc.\n']
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

            if iter != 0 and iter % 2000 == 0:
                print('Iter: ', iter, '| Loss: %6f' % loss)

        test_loss = 0
        test_count = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for _iter, test_data in enumerate(test_loader):
                test_x, test_y = test_data
                test_x = test_x.float().to(env)
                test_y = test_y.long().to(env)

                test_out = model(test_x)
                _, prediction = torch.max(test_out.data, 1)
                total += test_y.size(0)
                correct += (prediction == test_y).sum().item()

                test_loss += criterion(test_out, test_y)
                test_count += 1

            test_loss = test_loss / test_count
            history.append(str(epoch + 1) + ',' + str(test_loss) + ',' + str(100 * correct / total) + '\n')
            print('\nEpoch: ', epoch + 1, '| Testing loss: %6f' % test_loss, '| Testing Acc.: %2f' % (100 * correct / total), '\n')

    f = open(saving_name + '.csv', 'w')
    f.writelines(history)
    f.close()

    if save:
        torch.save(model, saving_name + '.pkl')

    print('All process done.')

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('Usage: python3 train_AT.py [CNN or ANN] [model depth] [model unit] [model name] [epochs] [device]')
        exit(0)

    start_time = time.time()
    if sys.argv[1] == 'ANN':
        model = ANN(depth = int(sys.argv[2]), unit = int(sys.argv[3]))
    elif sys.argv[1] == 'CNN':
        model = CNN(depth =  int(sys.argv[2]), channel = int(sys.argv[3]))
    else:
        print('Please check the model selection args.')
        exit(0)

    TrainModel(model, sys.argv[4], nn.CrossEntropyLoss(), int(sys.argv[5]), int(sys.argv[6]))
    print('All process done, cause %s seconds.' % (time.time() - start_time))


