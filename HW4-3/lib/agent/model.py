import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class Baseline(nn.Module):
    def __init__(self, image_size, action_selection):
        super(Baseline, self).__init__()

        self.groups = 2
        self.image_size = image_size
        self.action_selection = action_selection

        self.conv = nn.Sequential(
                nn.Conv2d(image_size[2], 16 * self.groups, 5, stride = 2, padding = 2, bias = False, groups = self.groups),
                nn.BatchNorm2d(16 * self.groups),
                nn.ReLU(),
                nn.Conv2d(16 * self.groups, 32 * self.groups, 5, stride = 2, padding = 2, bias = False, groups = self.groups),
                nn.BatchNorm2d(32 *  self.groups),
                nn.ReLU(),
                nn.Conv2d(32 * self.groups, 64, 5, stride = 2, bias = False),
                nn.BatchNorm2d(64),
                nn.ReLU(),
                )

        self.actor = nn.Linear(64 * 9 * 9, action_selection)
        self.softmax = nn.Softmax(dim = -1)

        self.critic = nn.Linear(64 * 9 * 9, 1)

    def forward(self, x):
        x = self.conv(x)

        x = x.view(x.size(0), -1)
        actor = self.actor(x)
        actor = self.softmax(actor)
        state_value = self.critic(x)
        return actor, state_value 

class PongLinear(nn.Module):
    def __init__(self, image_size, action_selection):
        super(PongLinear, self).__init__()

        self.image_size = image_size
        self.action_selection = action_selection

        self.main = nn.Sequential(
                nn.Linear(np.prod(image_size), 256, bias = True),
                nn.Dropout(p = 0.2),
                nn.ReLU(),
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
        state_value = self.critic(x)

        return action, state_value


