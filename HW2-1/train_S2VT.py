import os
import sys
import csv
import time
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from lib.utils import *
from lib.model import S2VT
from lib.dataset import VCDataSet
from lib.word2vec import Word2vec

def TrainModel(model, word2vec, saving_name, epochs, batch_size, device, save = True):
    model.float().to(env)
    criterion = nn.CrossEntropyLoss()
    criterion.to(env)
    optim = Adam(model.parameters())

    train_set = VCDataSet('./data/training_data/feat', './training_label_dict.pkl',
            word2vec.seq_max, mark = 'train', mode = 'fix')
    test_set =  VCDataSet('./data/testing_data/feat', './testing_label_dict.pkl',
            word2vec.seq_max, mark = 'test', mode = 'fix')

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)
    test_loader = DataLoader(test_set, batch_size = batch_size, shuffle = False)

    print('Model Structure:')
    print(model)

    history = ['Epoch,Train Loss,Test Loss\n']
    for epoch in range(epochs):
        print('Start training epoch:', epoch + 1)
        model = model.train()
        train_loss = []
        for iter, data in enumerate(train_loader):
            train_video, train_seq, train_mask, train_label, train_label_mask = data
            train_video = Pack_seq(train_video).to(env)
            train_seq = Pack_seq(train_seq).long().to(env)
            train_mask = Pack_seq(train_mask).byte().to(env)
            train_label = train_label.long().to(env)
            train_label_mask = train_label_mask.byte().to(env)

            train_label = Label_mask(train_label, train_label_mask)

            optim.zero_grad()

            out = model(train_video, train_seq, train_mask)

            loss = criterion(out.view(-1, word2vec.seq_max), train_label.view(-1, ))
            loss.backward()
            train_loss.append(loss.detach())

            optim.step()

            if iter != 0 and iter % 10 == 0:
                print('Iter: ', iter, ' | Loss: %6f' % loss.detach())

        test_loss = []
        model = model.eval()
        with torch.no_grad():
            for _iter, test_data in enumerate(test_loader):
                test_video, test_seq, test_label = data
                test_video = Pack_seq(test_video).to(env)
                test_seq = Pack_seq(test_seq).long().to(env)
                test_mask = Pack_seq(test_mask).byte().to(env)
                test_label = test_label.to(env)
                test_label_mask = test_label_mask.to(env)

                test_label = Label_mask(test_label, test_label, mask)

                test_out = model(test_video, test_seq)
                _, prediction = torch.max(test_out.data, 1)
                test_loss.append(criterion(test_out, test_label).detach())

        train_loss = torch.tensor(train_loss).mean().item()
        test_loss = torch.tensor(test_loss).mean().item()

        history.append(str(epoch + 1) + ',' + str(train_loss) + ',' + str(test_loss) + '\n')
        print('\nEpoch: ', epoch + 1, '| Train loss: %6f' % train_loss, '| Test loss: %6f' % test_loss, '\n')

    f = open(saving_name + '.csv', 'w')
    f.writelines(history)
    f.close()

    if save:
        torch.save(model, saving_name + '.pkl')

    print('All training process done.')

def Label_mask(label, mask):
    return torch.masked_select(label, mask)

def Pack_seq(tensor):
    new = torch.empty(tensor.size(1), tensor.size(0), tensor.size(2))
    for i in range(tensor.size(0)):
        new[:, i, :] = tensor[i, :, :]

    return new

def Een_setting(device):
    if device < 0:
        env = torch.device('cpu')
        print('Envirnment setting done, using device: cpu')
    else:
        torch.backends.cudnn.benchmark = True
        cuda.set_device(device)
        env = torch.device('cuda:' + str(device))
        print('Envirnment setting done, using device: CUDA_' + str(device))

    return env

def LoadModel(name, out_size, env):
    model = None
    for file in os.listdir('./'):
        if file == name + '.pkl':
            model = torch.load(name + '.pkl')
            print('Load model', name, 'success.')
            break

    if model == None:
        return S2VT(out_size, env)
    else:
        return model

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: python3 train_S2VT.py [model name] [epoch] [batch_size] [device]')
        exit(0)

    start_time = time.time()
    w2v = load_object('./word2vec.pkl')
    env = Een_setting(int(sys.argv[4]))
    model = LoadModel(sys.argv[1], w2v.seq_max, env)
    TrainModel(model, w2v, sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), env)
    print('All process done, cause %s seconds.' % (time.time() - start_time))


