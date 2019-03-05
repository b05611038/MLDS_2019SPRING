import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self, depth, unit):
        super(ANN, self).__init__()

        self.depth = depth
        self.unit = unit

        self.denses = nn.Sequential()
        self.denses.add_module('Dense_1', nn.Linear(1, unit))
        for i in range(depth - 2):
            self.denses.add_module('Dense_' + str(i + 2), nn.Linear(unit, unit))

        self.denses.add_module('Dense_' + str(depth), nn.Linear(unit, 1))

    def forward(self, data):
        out = self.denses(data)

        return out


