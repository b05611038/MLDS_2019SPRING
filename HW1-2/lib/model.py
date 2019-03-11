import torch
import torch.nn as nn
import torch.nn.functional as F

class ANN(nn.Module):
    def __init__(self, depth = 2, unit = 50):
        super(ANN, self).__init__()

        self.depth = depth
        self.unit = unit

        self.denses = nn.Sequential()
        self.denses.add_module('Dense_1', nn.Linear(1, unit, bias = False))
        self.denses.add_module('ReLU_1', nn.ReLU())
        for i in range(depth - 2):
            self.denses.add_module('Dense_' + str(i + 2), nn.Linear(unit, unit, bias = False))
            self.denses.add_module('ReLU_' + str(i + 2), nn.ReLU())

        self.denses.add_module('Dense_' + str(depth), nn.Linear(unit, 1, bias = False))

    def forward(self, data):
        x = self.denses(data)

        return x

class CNN(nn.Module):
    def __init__(self, depth = 2, channel = 5):
        super(CNN, self).__init__()

        self.depth = depth
        self.channel = channel

        self.conv = nn.Sequential()
        self.conv.add_module('Conv_1', nn.Conv2d(3, channel, 3, padding = 1, bias = False))
        self.conv.add_module('ReLU_1', nn.ReLU())
        for i in range(depth - 2):
            self.conv.add_module('Conv_' + str(i + 2), nn.Conv2d(channel, channel, 3, padding = 1, bias = False))
            self.conv.add_module('ReLU_' + str(i + 2), nn.ReLU())

        self.conv.add_module('Conv_' + str(depth), nn.Conv2d(channel, 10, 3, padding = 1, bias = False))

        self.avg = nn.AvgPool2d((32, 32), stride = (1, 1))

    def forward(self, data):
        x = self.conv(data)
        x = self.avg(x)

        x = x.view(-1, 10)

        return x


