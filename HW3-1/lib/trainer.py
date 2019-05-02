import os
import csv
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.data import DataLoader

from lib.utils import *
from lib.dataset import *
from lib.model import *
from lib.loss import *
from lib.visualize import *

class GANTrainer():
    def __init__(self, model_type, model_name, distribution, device):
        if distribution not in ['uniform', 'normal']:
            raise ValueError('Please input correct sample distribution. [uniform or normal]')

        self.distribution = distribution
        self.env = self._env_setting(device)

        self.model_type = model_type
        self.model_name = model_name
        if _check_continue_training(model_name):
            self.model = self._load_model(model_name)
        else:
            self.model = self._select_model(model_type)

    def train(self, name, epoch, batch_size, save = True):
        if batch_size == 1:
            raise RuntimeError("For training batchnorm layer, batch can't set as 1.")
        pass

    def _epoch(self, batch_size):
        pass

    def _select_model(self, model_select):
        if model_select not in ['GAN', 'DCGAN', 'WGAN', 'WGAN_GP']:
            raise ValueError('Please select correct GAN model. [GAN, DCGAN, WGAN, WGAN_GP]')

        if model_select == 'GAN':
            model = GAN()
        elif model_select == 'DCGAN':
            model = DCGAN()
        elif model_select == 'WGAN':
            model = WGAN()
        elif model_select == 'WGAN_GP':
            model = WGAN_GP()

        return model

    def _load_model(self, model_name):
        model = torch.load(model_name)
        print('Load model', name, 'success.')
        return model

    def _check_continue_training(self, model_name):
        path = model_name.split('/')

        if len(path) == 1:
            files = os.listdir('./')
        else:
            real_path = ''
            for i in range(len(path) - 1):
                if i == len(path) - 2:
                    real_path = real_path + path[i]
                else:
                    real_path = real_path + path[i] + '/'
            files = os.listdir(real_path)

        if model_name in files:
            return True
        else:
            return False

    def _env_setting(device):
        if device < 0:
            env = torch.device('cpu')
            print('Envirnment setting done, using device: cpu')
        else:
            torch.backends.cudnn.benchmark = True
            cuda.set_device(device)
            env = torch.device('cuda:' + str(device))
            print('Envirnment setting done, using device: cuda:' + str(device))

        return env


