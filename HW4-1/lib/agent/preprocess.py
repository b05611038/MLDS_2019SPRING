import numpy as np
import torch
import torchvision
import torchvision.transforms as T

#transform dictionary format pass in the class
# {'implenmented string': True}

class Transform(object):
    def __init__(self, preprocess_dict):
        self.implemented_list = self.implenmented()
        self.preprocess_dict = preprocess_dict
        keys = preprocess_dict.keys()
        for key in keys:
            if key not in self.implemented_list:
                raise KeyError(key, 'is not the implemented observation preprocess method.')

        self.transform = self._init_torchvision_method(preprocess_dict)

    def __call__(self, observation, memory = None):
        if preprocess_dict['slice_scoreboard'] == True:
            observation = self._slice_scoreboard(observation)

        if preprocess_dict['minus_observation'] == True:
            observation = self._minus_observation(observation, memory)

        observation = self.transform(observation)
        return observation

    def _init_torchvision_method(self, preprocess_dict):
        method = []
        if preprocess_dict['gray_scale'] == True:
            method.append(T.Grayscale())

        method.append(T.ToTensor())
        return T.Compose(method)

    def _minus_observation(self, observation, memory):
        return observation - memory

    def _slice_scoreboard(self, image):
        image = image[24:, :, :]
        return image

    def image_size(self):
        height = 210
        length = 160
        channel = 3
        if self.preprocess_dict['slice_scoreboard'] == True:
            height = 186
        if self.preprocess_dict['gray_scale'] == True:
            channel = 1
        return (height, length, channel)

    def implenmented(self):
        implemented_list = ['slice_scoreboard', 'gray_scale', 'minus_observation']
        return implemented_list


