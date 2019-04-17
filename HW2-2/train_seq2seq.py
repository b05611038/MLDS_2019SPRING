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
from lib.model import Seq2seq
from lib.dataset import CBDataset
from lib.word2vec import Word2vec

def TrainModel(model, word2vec, saving_name, epochs, batch_size, device, save = True):
    model.float().to(env)
    criterion = nn.CrossEntropyLoss()
    criterion.to(env)
    optim = Adam(model.parameters())

    train_set = CBDataset(sentence_pair_path = './data/sentence_pair.pkl', mask_max = word2vec.seq_max,
            seq_length_max = word2vec.seq_length_max)

    train_loader = DataLoader(train_set, batch_size = batch_size, shuffle = True)

    print('Model Structure:')
    print(model)

    history = ['Epoch,Train Loss,Train Acc.\n']
    for epoch in range(epochs):
        print('Start training epoch:', epoch + 1)
        model = model.train()
        train_total = 0
        train_right = 0
        train_loss = []
        for iter, data in enumerate(train_loader):
            sentence, guided_seq, guided_mask, label_seq, label_mask = data
            sentence = Pack_seq(sentence).long().to(env)
            guided_seq = Pack_seq(guided_seq).long().to(env)
            guided_mask = Pack_seq(guided_mask).byte().to(env)
            label_seq = label_seq.long().to(env)
            label_mask = label_mask.byte().to(env)

            label = Label_mask(label_seq, label_mask)

            optim.zero_grad()

            out, _hid = model(sentence, guided_seq, guided_mask)

            loss = criterion(out.view(-1, word2vec.seq_max), label.view(-1, ))
            loss.backward()

            _max, max_index = torch.max(out.view(-1, word2vec.seq_max), 1)
            train_total += label.view(-1, ).size(0)
            train_right += (max_index == label.view(-1, )).sum().item()

            _ = torch.nn.utils.clip_grad_norm_(model.parameters(), 0.01)

            train_loss.append(loss.detach())

            optim.step()

            del _hid
            if iter % 5 == 0 and iter != 0:
                print('Iter: ', iter, ' | Loss: %6f' % loss.detach())

        train_acc = train_right / train_total * 100
        train_loss = torch.tensor(train_loss).mean().item()

        if epoch % 10 == 0 and epoch != 0:
            model.probability *= 1.02

        history.append(str(epoch + 1) + ',' + str(train_loss) + ',' + str(test_loss) + '\n')
        print('\nEpoch: ', epoch + 1, '| Train loss: %6f' % train_loss, '| Train Acc. %.4f' % train_acc)

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

def LoadModel(name, out_size, env, max_seq_length, hidden_size, bidrectional, attention, mode = 'self', probability = 0.2):
    model = None
    for file in os.listdir('./'):
        if file == name + '.pkl':
            model = torch.load(name + '.pkl')
            print('Load model', name, 'success.')
            break

    if model == None:
        return Seq2seq(out_size, env, max_seq_length, hidden_size, bidrectional, attention, mode, probability)
    else:
        return model

if __name__ == '__main__':
    if len(sys.argv) < 8:
        print('Usage: python3 train_seq2seq.py [model name] [hidden_size] [bidrectional] [attention] [epoch] [batch_size] [device]')
        exit(0)

    start_time = time.time()
    w2v = load_object('./word2vec.pkl')
    env = Een_setting(int(sys.argv[7]))
    model = LoadModel(sys.argv[1], w2v.seq_max, env, w2v.seq_length_max,
            int(sys.argv[2]), bool(sys.argv[3]), sys.argv[4])
    TrainModel(model, w2v, sys.argv[1], int(sys.argv[5]), int(sys.argv[6]), env)
    print('All process done, cause %s seconds.' % (time.time() - start_time))


