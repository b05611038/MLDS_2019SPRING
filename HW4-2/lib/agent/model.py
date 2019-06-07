import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaselineModel(nn.Module):
    def __init__(self, image_size, action_selection):
        super(BaselineModel, self).__init__()

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

        self.linear = nn.Linear(64 * 9 * 9, action_selection)

    def forward(self, x):
        x = self.conv(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


