import os
import torch
import torchvision
import torchvision.transforms as tfs

from lib.agent.base import Agent
from lib.agent.model import BaselineModel

class PGAgent(Agent):
    def __init__(self, name, device, observation_size, action_size, model_select, model_path = None,
            policy = 'Policy gradient'):

        self.name = name
        self.device = device
        self.observation_size = observation_size
        self.action_size = action_size
        self.model_select = model_select
        self.model_path = model_path
        self.model = self._init_model(model_select, model_path, observation_size, action_size)

    def make_action(self, observation, mode = 'env'):
        if mode == 'env':
            pass
        elif mode == 'train':
            pass
        else:
            raise ValueError(mode, 'is not in agent action mode setting.')

    def save(self, path):
        save_path = os.path.join(path, (self.name + '.pth'))
        torch.save(self.model.state_dict(), model)

    def _preprocess(self, observation):
        pass

    def _init_model(self, model_select, model_path, observation_size, action_size):
        if model_path is not None:
            if model_select == 'baseline':
                model = BaselineModel(image_size = observation_size, action_selection = action_size)
                model = model.to(self.device)
                return model
            else:
                raise ValueError(model_select, 'is not in implemented model.')

        else:
            if model_select == 'baseline':
                model = BaselineModel(image_size = observation_size, action_selection = action_size)
                model = model.load_state_dict(torch.load(model_path))
                model = model.to(self.device)

                return model


