import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, channel, depth = 10):
        super(CNN, self).__init__()

        self.channel = channel
        self.depth = depth

        self.conv = nn.Sequential()
        self.conv.add_module('conv_1', nn.Conv2d(3, channel, 3, padding = 1, bias = False))
        self.conv.add_module('bn_1', nn.BatchNorm2d(channel))
        self.conv.add_module('relu_1', nn.ReLU(inplace = True))
        for i in range(1, depth - 1):
            self.conv.add_module('conv_' + str(i + 1), nn.Conv2d(channel, channel, 3, padding = 1, bias = False))
            self.conv.add_module('bn_' + str(i + 1), nn.BatchNorm2d(channel))
            self.conv.add_module('relu_' + str(i + 1), nn.ReLU(inplace = True))

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dense = nn.Linear(channel, 10)

    def forward(self, data):
        x = self.conv(data)
        x = self.avg_pool(x)

        x = x.view(x.size(0), -1)
        x = self.dense(x)

        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride = 1, short = None):
        super(BasicBlock, self).__init__()

        self.in_channel = in_channel
        self.out_channel = out_channel
        self.stride = stride
        self.short = short

        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size = 3, stride = stride, padding = 1, bias = False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size = 3, stride = 1, padding = 1, bias = False)
        self.bn2 = nn.BatchNorm2d(out_channel)

        self.relu = nn.ReLU(inplace = True)

    def forward(self, data):
        residual = data

        x = self.conv1(data)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)

        if self.short is not None:
            residual = self.short(data)

        x += residual
        x = self.relu(x)

        return x

class ResNet(nn.Module):
    def __init__(self, num_class = 10):
        super(ResNet, self).__init__()

        self.in_channel = 64
        self.num_class = num_class

        self.lead_conv = nn.Conv2d(3, 64, kernel_size = 7, stride = 2, padding = 3, bias = False)
        self.lead_bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace = True)

        self.layer1 = self._make_layer(64, 2, stride = 1)
        self.layer2 = self._make_layer(128, 2, stride = 2)
        self.layer3 = self._make_layer(256, 2, stride = 2)
        self.layer4 = self._make_layer(512, 2, stride = 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

        self.dense = nn.Linear(512, num_class)

    def forward(self, data):
        x = self.lead_conv(data)
        x = self.lead_bn(x)
        x = self.relu(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.dense(x)

        return x

    def _make_layer(self, out_channel, num_block, stride):
        short = None
        if stride != 1 or self.in_channel != out_channel:
            short = nn.Sequential(
                    nn.Conv2d(self.in_channel, out_channel, kernel_size = 1, stride = stride, bias = False),
                    nn.BatchNorm2d(out_channel),
                    )

        layers = []
        layers.append(BasicBlock(self.in_channel, out_channel, stride, short = short))
        self.in_channel = out_channel
        for _ in range(1, num_block):
            layers.append(BasicBlock(self.in_channel, out_channel))

        return nn.Sequential(*layers)


