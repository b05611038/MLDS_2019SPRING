import numpy as np
import torch
import torch.cuda as cuda

from lib.agent.base import Agent
from lib.agent.preprocess import Transform
from lib.agent.model import BaselineModel

class PGAgent(Agent):
    def __init__(self, name, model_select, device,
            observation_preprocess, max_memory_size, valid_action):
        super(PGAgent, self).__init__()

        self.name = name
        self.device = device

        self.observation_preprocess = observation_preprocess
        self.transform = Transform(observation_preprocess)

        self.max_memory_size = max_memory_size
        self.valid_action = valid_action

        self.model_select = model_select
        self.model = self._init_model(model_select, self.transform.image_size(), action_size)
        self.memory = None

    def make_action(self, observation):
        #return processed model observation and action
        processed = self._preprocess(observation).to(self.device)
        output = self.model(processed)
        _, action = torch.max(output, 1)
        return action.cpu().detach().numpy()[0]

    def insert_memory(self, observation):
        observation = self._preprocess(observation)
        self.memory = observation
        return None

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        return None

    def load(self, path):
        self.model.cpu()
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        return None

    def _preprocess(self, observation):
        return self.transform(observation, self.memory)

    def _check_memory(self):
        if len(self.memory) > self.max_memory_size:
            self.memory = self.memory[-self.max_memory_size: ]
        return None

    def _init_model(self, model_select, observation_size, action_size):
        if model_select == 'baseline':
            model = BaselineModel(image_size = np.prod(observation_size), action_selection = action_size)
            model = model.to(self.device)
            return model
        else:
            raise ValueError(model_select, 'is not in implemented model.')


