import os
import time
import numpy as np
import torch
import torch.cuda as cuda
import torchvision
import torchvision.transforms as tfs

from torch.optim import SGD, Adam
from torch.utils.data import DataLoader

from lib.utils import *
from lib.dataset import ReplayBuffer
from lib.environment.environment import Environment
from lib.agent.agent import PGAgent

class PGTrainer(object):
    def __init__(self, model_type, model_name, policy, device):
        self.device = self._device_setting(device)

        self.model_type = model_type
        self.model_name = model_name
        #self.agent = 
        self.save_dir = self._create_dir(model_name)

        self.policy = policy

    def _create_dir(self, model_name):
        #information saving directory
        #save model checkpoint and episode history
        if not os.path.exists('./output'):
            os.makedirs('./output')

        save_dir = os.path.join('./output', model_name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        print('All training output would save in', save_dir)
        return save_dir

    def _device_setting(self, device):
        #select training environment and device
        print('Init training device and environment ...')
        if device < 0:
            training_device = torch.device('cpu')
            print('Envirnment setting done, using device: cpu')
        else:
            torch.backends.cudnn.benchmark = True
            cuda.set_device(device)
            training_device = torch.device('cuda:' + str(device))
            print('Envirnment setting done, using device: cuda:' + str(device))

        return training_device


