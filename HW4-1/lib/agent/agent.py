import torch
import torch.cuda as cuda
import torchvision
import torchvision.transforms as tfs

from lib.agent.base import Agent
from lib.agent.model import BaselineModel

class PGAgent(Agent):
    def __init__(self, name, model_select, device, observation_preprocess, valid_action):

        self.name = name
        self.device = device
        self.observation_size = observation_size
        self.action_size = action_size
        self.preprocess = preprocess
        self.model_select = model_select
        self.model = self._init_model(model_select, observation_size, action_size)

    def make_action(self, observation, mode = 'env'):
        if mode == 'env':
            pass
        elif mode == 'train':
            pass
        else:
            raise ValueError(mode, 'is not in agent action mode setting.')

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        return None

    def load(self, path):
        self.model.cpu()
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        return None

    def _preprocess(self, observation):
        pass

    def _init_model(self, model_select, observation_size, action_size):
        if model_select == 'baseline':
            model = BaselineModel(image_size = observation_size, action_selection = action_size)
            model = model.to(self.device)
            return model
        else:
            raise ValueError(model_select, 'is not in implemented model.')


