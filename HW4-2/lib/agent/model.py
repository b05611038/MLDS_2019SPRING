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

        print(image_size)
        self.conv = nn.Sequential(
                nn.Conv2d(image_size[2], 16 * self.groups, 5, stride = 2, padding = 2, bias = False, groups = self.groups),
                nn.ReLU(),
                nn.Conv2d(16 * self.groups, 32, 5, stride = 2, padding = 2, bias = False),
                nn.ReLU()
                )

        self.linear = nn.Linear(int(32 * image_size[0] / 4 * image_size[1] / 4), action_selection)

    def forward(self, x):
        x = self.conv(x)

        x = x.view(x.size(0), -1)
        x = self.linear(x)
        return x


