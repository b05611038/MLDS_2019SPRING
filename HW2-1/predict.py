import os
import sys
import csv
import time
import numpy as np
import torch
import torch.cuda as cuda

from lib.utils import *
from lib.beamsearch import BeamSearch

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

def Grab_test(path):
    test_set = []
    for file in os.listdir(path):
        if file.endswith('.npy'):
            video = np.load(path + '/' + file)
            test_set.append([video, file.replace('.npy', '')])

    print('Testing set building done.')
    return test_set

def Predict(model, w2v, test_set, env, k):
    searcher = BeamSearch(model, w2v, env, k = k)
    w2v = load_object(w2v)
    outcome = []
    for i in range(len(test_set)):
        video = test_set[i][0]
        video = torch.tensor(video).view(80, 1, 4096).float().to(env)
        seq = searcher(video).to('cpu')

        no_title = True
        sentence = test_set[i][1] + ','
        for j in range(seq.size(0)):
            word = w2v.v2w(torch.tensor([seq[j]]))
            word, is_word = Clean(word)
            if is_word and no_title:
                word = word.title()
                no_title = False

            if is_word:
                sentence = sentence + word + ' '
            else:
                pass

        sentence += '\n'
        outcome.append(sentence)
        if i != 0 and (i + 1) % 5 == 0:
            print('Progress:', i + 1, '/', len(test_set))

    return outcome

def In_file(outcome):
    f = open('output.txt', 'w')
    f.writelines(outcome)
    f.close()
    print('Writing outcome to ./output.txt done.')

def Clean(word):
    word = word.replace('(', '')
    word = word.replace(')', '')
    if word == '<eos>':
        return '', False
    elif word == '<bos>':
        return '', False 
    elif word == '<padding>':
        return '', False
    else:
        return word, True

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python3 predict.py [model] [word2vec] [search beam] [device]')
        exit(0)

    start_time = time.time()
    env = Een_setting(int(sys.argv[4]))
    test_set = Grab_test('./data/testing_data/feat')
    outcome = Predict(sys.argv[1], sys.argv[2], test_set, env, int(sys.argv[3]))
    In_file(outcome)
    print('All process done, cause %s seconds.' % (time.time() - start_time))
