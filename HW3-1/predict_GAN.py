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
    Save_imgs(model, 'output_' + sys.argv[1].split('/')[-1: ][0].replace('.pkl', '')) 


