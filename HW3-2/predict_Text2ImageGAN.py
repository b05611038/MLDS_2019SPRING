import sys
import numpy as np
import torch
import torch.cuda as cuda

from lib.model import *
from lib.utils import *
from lib.visualize import *

def Load_model(path, device):
    model = torch.load(path, map_location = 'cpu')
    model.device = device
    model.to(device)

    return model

def Load_test_input(path):
    eyes = ['aqua', 'black', 'blue', 'brown', 'green', 'orange', 'pink', 'purple', 'red', 'yellow']
    hairs = ['aqua', 'gray', 'green', 'orange', 'red', 'white', 'black', 'blonde', 'blue', 'brown',
            'pink', 'purple']

    f = open(path, 'r')
    text = f.readlines()
    f.close()

    tags = {}
    for line in text:
        context = line.replace('\n', '').split(',')
        index = int(context[0])
        feature = context[1].split(' ')
        hair = feature[0]
        eye = feature[2]
        for i in range(len(hairs)):
            if hair == hairs[i]:
                hair = i
                break

        for i in range(len(eyes)):
            if eye == eyes[i]:
                eye = i
                break

        tags[index] = hair * len(eyes) + eye

    tag_tensor = torch.zeros(len(text), len(hairs) * len(eyes))
    iter_index = 0
    for key in list(tags.keys()):
        tag_tensor[iter_index, key] = 1
        iter_index += 1

    return tag_tensor

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

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 predict_GAN.py [model] [device]')
        exit(0)

    env = Een_setting(int(sys.argv[2]))
    model = Load_model(sys.argv[1], env)
    tags = Load_test_input('./data/testing_tags.txt')
    Save_imgs(model, 'output_' + sys.argv[1].split('/')[-1: ][0].replace('.pkl', ''), tags)


