import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import *


class BaselineModel(nn.Module):
    def __init__(self, image_size, action_selection):
        super(BaselineModel, self).__init__()

        self.image_size = image_size
        self.action_selection = action_selection

        #two layer linear model
        self.main = nn.Sequential(
                nn.Linear(np.prod(image_size), 256, bias = True),
                nn.Dropout(p = 0.2),
                nn.ReLU()
                )

        self.actor = nn.Sequential(
                nn.Linear(256, action_selection, bias = True),
                nn.Softmax(dim = -1)
                )

        self.critic = nn.Linear(256, 1)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.main(x)
        action = self.actor(x)
        value = self.critic(x)

        return action, value


