import os
import re
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

def Grab_input(path, word2vec):
    test_set = []
    f = open(path, 'r')
    text = f.readlines()
    f.close()

    for i in range(len(text)):
        sentence = text[i].replace('\n', '').split(' ')
        sentence = [w for w in sentence if not re.match(r'[A-Z]+', w, re.I)] #may change
        sentence = [re.sub('[0-9]', '', w) for w in sentence] #may change

        arr_sentence = np.empty(len(sentence), )
        for word in range(len(sentence)):
            arr_sentence[word] = word2vec.w2v(sentence[word])

        test_set.append(arr_sentence)

    return test_set

def Predict(model, word2vec, test_set, env, k):
    searcher = BeamSearch(model, word2vec, env, k = k)
    outcome = []
    for index in range(len(test_set)):
        sentence = torch.tensor(test_set[index]).view(-1, 1, 1).long().to(env)
        seq = searcher(sentence).to('cpu')

        sentence = ''
        for word in range(seq.size(0)):
            sentence += Clean(word2vec.v2w(torch.tensor([seq[word]])))

        sentence += '\n'
        outcome.append(sentence)
        if index != 0 and (index + 1) % 5 == 0:
            print('Progress:', index + 1, '/', len(test_set))

    return outcome

def Clean(word):
    clean_token = ['<padding>', '<unknown>', '<bos>', '<eos>', '<padding>']
    #clean_token = ['<padding>', '<unknown>', '<bos>', '<eos>', '<padding>',
    #        '．', '〞', '◎', '∫', '♪', '』', '『']
    if word in clean_token:
        return ''
    else:
        return word

def In_file(outcome, model):
    f = open('output_' + model.replace('.pkl', '') + '.txt', 'w')
    f.writelines(outcome)
    f.close()
    print('Writing outcome to ./output.txt done.')

if __name__ == '__main__':
    if len(sys.argv) < 6:
        print('Usage: python3 predict.py [in file] [model] [word2vec] [search beam] [device]')
        exit(0)

    start_time = time.time()
    env = Een_setting(int(sys.argv[5]))
    w2v = load_object(sys.argv[3])
    test_set = Grab_input(sys.argv[1], w2v)
    outcome = Predict(sys.argv[2], w2v, test_set, env, int(sys.argv[4]))
    In_file(outcome, sys.argv[2])
    print('All process done, cause %s seconds.' % (time.time() - start_time))

        
