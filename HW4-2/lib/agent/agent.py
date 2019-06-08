import random
import numpy as np
import torch
import torch.cuda as cuda

from lib.agent.base import Agent
from lib.agent.preprocess import Transform
from lib.agent.model import BaselineModel

class QAgent(Agent):
    def __init__(self, name, model_select, device,
            observation_preprocess, max_memory_size, valid_action):
        super(QAgent, self).__init__()

        self.name = name
        self.device = device

        self.observation_preprocess = observation_preprocess
        self.transform = Transform(observation_preprocess, device)

        self.max_memory_size = max_memory_size
        self.valid_action = valid_action

        self.model_select = model_select
        self.model = self._init_model(model_select, self.transform.image_size(), len(valid_action))
        self.memory = None

    def training_model_out(self, observation):
        self.model = self.model.train()
        return self.model(observation.to(self.device))

    def make_action(self, observation, mode = 'mix', p = None):
        #return processed model observation and action
        if self.observation_preprocess['minus_observation'] == True:
            if self.memory is None:
                raise RuntimeError('Please insert init memory before playing a game.')

        self.model = self.model.eval()
        processed = self._preprocess(observation)
        processed = processed.to(self.device)
        input_processed = processed.unsqueeze(0)
        output = self.model(input_processed)
        self.insert_memory(observation)
        action = self._decode_model_output(output, mode, p)

        return action, processed.cpu().detach()

    def random_action(self):
        return self.valid_action[random.randint(0, len(self.valid_action) - 1)]

    def insert_memory(self, observation):
        self.memory = self.transform(observation)
        return None

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        return None

    def load(self, path):
        self.model.cpu()
        self.model.load_state_dict(torch.load(path, map_location = 'cpu'))
        self.model.to(self.device)
        return None

    def _decode_model_output(self, output, mode, p):
        if mode == 'argmax':
            _, action = torch.max(output, 1)
            action_index = action.cpu().detach().numpy()[0]
            action = self.valid_action[action_index]
            return action
        elif mode == 'mix':
            # p is rnadom probability, if 1.0 means all action is random
            if p is None:
                raise ValueError('Please set argument p in make_action')

            if random.random() < p:
                #means random action
                action_index = random.randint(0, len(self.valid_action) - 1)
                action = self.valid_action[action_index]
                return action
            else:
                _, action = torch.max(output, 1)
                action_index = action.cpu().detach().numpy()[0]
                action = self.valid_action[action_index]
                return action

    def _preprocess(self, observation):
        return self.transform(observation, self.memory)

    def _check_memory(self):
        if len(self.memory) > self.max_memory_size:
            self.memory = self.memory[-self.max_memory_size: ]
        return None

    def _init_model(self, model_select, observation_size, action_size):
        if model_select == 'baseline':
            model = BaselineModel(image_size = observation_size, action_selection = action_size)
            model = model.to(self.device)
            return model
        else:
            raise ValueError(model_select, 'is not in implemented model.')


