import numpy as np
import torch
import torch.cuda as cuda
from torch.distributions import Categorical

from lib.agent.base import Agent
from lib.agent.preprocess import Transform
from lib.agent.model import BaselineModel

class QAgent(Agent):
    def __init__(self, name, model_select, device,
            observation_preprocess, max_memory_size, valid_action):
        super(PGAgent, self).__init__()

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

    def make_action(self, observation):
        #return processed model observation and action
        if self.observation_preprocess['minus_observation'] == True:
            if self.memory is None:
                raise RuntimeError('Please insert init memory before playing a game.')

        #self.model = self.model.eval()
        #processed = self._preprocess(observation)
        #processed = processed.to(self.device)
        #input_processed = processed.unsqueeze(0)
        #output = self.model(input_processed)
        #self.insert_memory(observation)
        #action = self._decode_model_output(output)
        #return action, processed.cpu().detach(), output.cpu().detach()

    def insert_memory(self, observation):
        observation = self._preprocess(observation, mode = 'init')
        self.memory = observation.to(self.device)
        return None

    def save(self, path):
        torch.save(self.model.state_dict(), path)
        return None

    def load(self, path):
        self.model.cpu()
        self.model.load_state_dict(torch.load(path))
        self.model.to(self.device)
        return None

    def _decode_model_output(self, output, mode = 'sample'):
        '''
        if mode == 'argmax':
            _, action = torch.max(output, 1)
            action_index = action.cpu().detach().numpy()[0]
            action = self.valid_action[action_index]
            return action
        elif mode == 'sample':
            try:
                output = output.detach().squeeze().cpu()
                m = Categorical(output)
                action_index = m.sample().numpy()
                action = self.valid_action[action_index]
                return action
            except RuntimeError:
                #one numbers in  probability distribution is zero
                _, action = torch.max(output, 0)
                action_index = action.cpu().detach().numpy()
                action = self.valid_action[action_index]
                return action
        '''

    def _preprocess(self, observation, mode = 'normal'):
        if mode == 'normal':
            return self.transform(observation, self.memory)
        elif mode == 'init':
            return self.transform.insert_init_memory(observation)

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


