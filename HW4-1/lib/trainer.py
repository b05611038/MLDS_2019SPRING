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
    def __init__(self, model_type, model_name, observation_preprocess, valid_action, device, optimizer = 'Adam', policy = 'PPO', env = 'Pong-v0'):
        self.device = self._device_setting(device)

        self.model_type = model_type
        self.model_name = model_name

        self.env = Environment(env, None)
        self.observation_preprocess = observation_preprocess
        self.valid_action = valid_action
        self.agent = PGAgent(model_name, model_type, self.device, observation_preprocess, 1, valid_action)
        self.state = self._continue_training(model_name)
        self.save_dir = self._create_dir(model_name)

        self.dataset = ReplayBuffer(env = env, maximum = 128)
        self.policy = policy

    def play(self, max_state, episode_size, save_interval):
        state = self.state
        max_state += self.state
        while(state <= max_state):
            self._collect_data(self.agent, self.env, batch_size)
            if dataset.trainable():
               pass
            else:
               self._collect_data(self.agent, self.env, batch_size)


    def _collect_data(self, agent, times):
        reward = []
        for i in range(times):
            done = False
            observation = self.env.reset()
            self.agent.insert_memory(observation)
            time_step = 0
            while(!done):
                action = self.agent.make_action(observation)
                observation_next, reward, done, _ = self.env.step(action)
                self.dataset.insert(observation, action, reward)
                observation = observation_next
                time_step += 1

        return reward

    def _select_optimizer(self, select):
        if select == 'SGD':
            self.optim = SGD(self.agent.model.parameters(), lr = 0.01, momentum = 0.9)
        elif select == 'Adam':
            self.optim = Adam(self.agent.model.parameters(), lr = 0.01)
        else:
            raise ValueError(select, 'is not valid option in choosing optimizer.')

        return None

    def _save_checkpoint(self, state, mode = 'episode'):
        #save the state of the model
        print('Start saving model checkpoint ...')
        if mode == 'episode':
            save_dir = os.path.join(self.save_dir, ('model_episode_' + str(state) + '.pth'))
            self.agent.save(save_dir)
        elif mode == 'iteration':
            save_dir = os.path.join(self.save_dir, ('model_iteration_' + str(state) + '.pth'))
            self.agent.save(save_dir)

        print('Model:', self.model_name, 'checkpoint', state, 'saving done.')
        return None

    def _continue_training(self, model_name):
        if os.path.isdir(os.path.join('./output', model_name)):
            save_dir = os.path.join('./output', model_name)
            files = os.listdir(save_dir)
            model_list = []
            for file in files:
                if file.endswith('.pth'):
                    model_list.append(file)

            if len(model_list) > 0:
                model_list.sort()
                model_state_path = os.path.join(save_dir, model_list[-1])
                training_state = int(model_list[-1].replace('.pth', '').split('_')[2])

                #load_model
                self.agent.load(model_state_path)
                return training_state
            else:
                return 0
        else:
            return 0

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


