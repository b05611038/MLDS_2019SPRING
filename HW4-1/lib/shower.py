import os
import time
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import *
from lib.environment.environment import Environment
from lib.agent.agent import PGAgent

class PGShower(object):
    def __init__(self, model_type, model_name, observation_preprocess, reward_preprocess, device, env = 'Pong-v0'):
        self.device = self._device_setting(device)

        self.model_type = model_type
        self.model_name = model_name

        self.env = Environment(env, None)
        self.observation_preprocess = observation_preprocess
        self.valid_action = self._valid_action(env)
        pass


    def _get_model_checkpoint(self, model_name):
        if os.path.isdir(os.path.join('./output', model_name)):
            save_dir = os.path.join('./output', model_name)
            files = os.listdir(save_dir)
            model_list = []
            for file in files:
                if file.endswith('.pth'):
                    model_list.append(os.path.join(save_dir, file))

            return model_list

        else:
            print('There is not any checkpoint cna used for showing.')
            exit(0)

    def _valid_action(self, env):
        if env == 'Pong-v0':
            #only need up and down
            return [2, 3]

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


