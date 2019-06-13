import random
import numpy as np
import torch
import torch.cuda as cuda
from torch.distributions import Categorical

from lib.agent.base import Agent
from lib.agent.preprocess import Transform
from lib.agent.model import Baseline

class ACAgent(Agent):
    def __init__(self, name, model_select, env_id, device,
            observation_preprocess, max_memory_size, valid_action):
        super(ACAgent, self).__init__()

        self.name = name
        self.env_id = env_id
        self.device = device

        self.observation_preprocess = observation_preprocess
        self.transform = Transform(observation_preprocess, env_id, device)

        self.max_memory_size = max_memory_size
        self.valid_action = valid_action

        self.model_select = model_select
        self.model = self._init_model(model_select, self.transform.image_size(), len(valid_action))
        self.memory = None

    def training_model_out(self, observation):
        self.model = self.model.train()
        return self.model(observation.to(self.device))

    def make_action(self, observation, mode = 'sample', p = None):
        #return processed model observation and action
        if self.observation_preprocess['minus_observation'] == True:
            if self.memory is None:
                raise RuntimeError('Please insert init memory before playing a game.')

        self.model = self.model.eval()
        processed = self._preprocess(observation)
        processed = processed.to(self.device)
        input_processed = processed.unsqueeze(0)
        output, _ = self.model(input_processed)
        self.insert_memory(observation)
        action, action_index = self._decode_model_output(output, mode, p)

        return action, action_index, output, processed.cpu().detach()

    def init_action(self):
        if self.env_id == 'Breakout-v0':
            return 1
        elif self.env_id == 'Pong-v0':
            select = random.randint(0, len(self.valid_action) - 1)
            return self.valid_action[select]
        else:
            raise NotImplementedError(self.env_id, 'is not in implemented environment.')

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
            if p is None:
                _, action = torch.max(output, 1)
                action_index = action.cpu().detach().numpy()[0]
                return self.valid_action[action_index], action_index
            else:
                if random.random() < p:
                    select = random.randint(0, len(self.valid_action))
                    return self.valid_action[select], action_index
                else:
                    _, action = torch.max(output, 1)
                    action_index = action.cpu().detach().numpy()[0]
                    return self.valid_action[action_index], action_index
        elif mode == 'sample':
            if p is None:
                try:
                    output = output.detach().squeeze().cpu()
                    m = Categorical(output)
                    action_index = m.sample().numpy()
                    return self.valid_action[action_index], action_index
                except RuntimeError:
                    #one numbers in  probability distribution is zero
                    _, action = torch.max(output, 0)
                    action_index = action.cpu().detach().numpy()[0]
                    return self.valid_action[action_index], action_index
            else:
                if random.random() < p:
                    select = random.randint(0, len(self.valid_action))
                    return self.valid_action[select], action_index
                else:
                    try:
                        output = output.detach().squeeze().cpu()
                        m = Categorical(output)
                        action_index = m.sample().numpy()
                        return self.valid_action[action_index], action_index
                    except RuntimeError:
                        _, action = torch.max(output, 0)
                        action_index = action.cpu().detach().numpy()[0]
                        return self.valid_action[action_index], action_index

    def _preprocess(self, observation):
        return self.transform(observation, self.memory)

    def _check_memory(self):
        if len(self.memory) > self.max_memory_size:
            self.memory = self.memory[-self.max_memory_size: ]
        return None

    def _init_model(self, model_select, observation_size, action_size):
        if model_select == 'baseline':
            model = Baseline(image_size = observation_size, action_selection = action_size)
            model = model.to(self.device)
            return model
        else:
            raise ValueError(model_select, 'is not in implemented model.')


