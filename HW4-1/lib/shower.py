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
from lib.visualize import VideoMaker, PGPlotMaker 

class PGShower(object):
    def __init__(self, model_type, model_name, observation_preprocess, device, env = 'Pong-v0'):
        self.device = self._device_setting(device)

        self.model_type = model_type
        self.model_name = model_name

        self.env = Environment(env, None)
        self.observation_preprocess = observation_preprocess
        self.valid_action = self._valid_action(env)
        self.agent = PGAgent(model_name, model_type, self.device, observation_preprocess, 1, self.valid_action)
        self.models = self._get_model_checkpoint(model_name)
        self.save_path = os.path.join('./output', model_name)
        self.ploter = PGPlotMaker(model_name = model_name)

    def show(self, sample_times):
        print('Plot training history ...')
        self.ploter.plot_all()

        print('Start make checkpoint models interact with environmnet ...')
        maker = VideoMaker(self.model_name)
        
        for iter in range(len(self.models)):
            self.agent.load(self.models[iter])
            self.agent.model.eval()
            scores, videos = self._play_game(sample_times)
            index = self._max_score(scores)
            maker.insert_video(videos[index])
            maker.make(self.save_path, self.models[iter].split('/')[-1].replace('.pth', ''))

            print('Progress:', iter + 1, '/', len(self.models))

        print('All video saving done.')
        return None

    def _play_game(self, times):
        scores = []
        videos = []
        for i in range(times):
            done = False
            observation = self.env.reset()
            videos.append([])
            self.agent.insert_memory(observation)
            scores.append(0.0)

            while not done:
                action, _processed, _model_out = self.agent.make_action(observation)
                observation_next, reward, done, _ = self.env.step(action)
                scores[i] += reward
                videos[i].append(observation)
                observation = observation_next

        return scores, np.asarray(videos)

    def _max_score(self, scores):
        max_score = -1
        record = -1
        for i in range(len(scores)):
            if scores[i] > max_score:
                record = i
                max_score = scores[i]

        return record

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
            print('There is not any checkpoint can used for showing.')
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


