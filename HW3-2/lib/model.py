import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import *

def weights_init(model):
    class_name = model.__class__.__name__
    if class_name.find('Conv') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
    elif class_name.find('Linear') != -1:
        nn.init.normal_(model.weight.data, 0.0, 0.02)
        nn.init.normal_(model.bias.data, 0.0, 0.02)
    elif class_name.find('BatchNorm') != -1:
        nn.init.normal_(model.weight.data, 1.0, 0.02)
        nn.init.constant_(model.bias.data, 0.0)

class Text2ImageGAN(nn.Module):
    def __init__(self, text_length, device, distribution = 'torch', noise_length = 100, out_channel = 3, channel = 64, sigmoid_used = True, init_weight = True):
        super(Text2ImageGAN, self).__init__()

        self.text_length = text_length
        self.device = device

        self.distribution = distribution
        self.noise_length = noise_length
        self.out_channel = out_channel
        self.channel = channel
        self.sigmoid_used = sigmoid_used
        self.init_weight = init_weight

        self.generator = T2IGenerator(channel, out_channel, noise_length, text_length)
        self.discriminator = T2IDiscriminator(out_channel, channel, text_length, sigmoid_used)

        if init_weight:
            self.generator.apply(weights_init)
            self.discriminator.apply(weights_init)

        self.generator = self.generator.float().to(self.device)
        self.discriminator = self.discriminator.float().to(self.device)

    def forward(self, feed, mode = 'generate'):
        # feed is list for [text, image]
        if mode not in ['generate', 'discriminate']:
            raise ValueError('Please check the model mode, [generate or discrimiate].')

        if mode == 'generate':
            latent_vector = self._latent_random(feed[0].size(0), self.distribution)
            return self.generator(feed[0], latent_vector)
        elif mode == 'discriminate':
            return self.discriminator(feed[1], feed[0])
        else:
            raise RuntimeError('Please check the model mode.')

    def _latent_random(self, numbers, distribution):
        if distribution == 'uniform':
            latent = np.random.uniform(-1, 1, (numbers, self.noise_length))
            return torch.tensor(latent).float().to(self.device)
        elif distribution == 'normal':
            latent = np.random.normal(0, 1, (numbers, self.noise_length))
            return torch.tensor(latent).float().to(self.device)
        elif distribution == 'torch':
            latent = torch.randn(numbers, self.noise_length)
            return latent.float().to(self.device)
        else:
            raise RuntimeError("Can't generate random latent vector.")


class T2IGenerator(nn.Module):
    def __init__(self, channel, out_channel, noise_length, text_length):
        super(T2IGenerator, self).__init__()

        self.channel = channel
        self.out_channel = out_channel
        self.noise_length = noise_length
        self.text_length = text_length

        self.embedding = nn.Linear(text_length, channel * 4)
        self.dense = nn.Linear(channel * 4 + noise_length, 4 * 4 * channel * 8)

        self.main = nn.Sequential(
                # size: [batch * (channel * 8) * 4 * 4]
                nn.ConvTranspose2d(channel * 8, channel * 4, kernel_size = 5, stride = 2, padding = 2, output_padding = 1, bias = False),
                nn.BatchNorm2d(channel * 4, momentum = 0.9),
                nn.ReLU(inplace = True),
                # size [batch * (channel * 4) * 8 * 8]
                nn.ConvTranspose2d(channel * 4, channel * 2, kernel_size = 5, stride = 2, padding = 2, output_padding = 1, bias = False),
                nn.BatchNorm2d(channel * 2, momentum = 0.9),
                nn.ReLU(inplace = True),
                # size [batch * (channel * 4) * 16 * 16]
                nn.ConvTranspose2d(channel * 2, channel, kernel_size = 5, stride = 2, padding = 2, output_padding = 1, bias = False),
                nn.BatchNorm2d(channel, momentum = 0.9),
                nn.ReLU(inplace = True),
                # size [batch * (channel * 4) * 32 * 32]
                nn.ConvTranspose2d(channel, out_channel, kernel_size = 5, stride = 2, padding = 2, output_padding = 1, bias = False),
                nn.Tanh()
                # size: [batch * out_channel * 64 * 64]
                )

    def forward(self, text, noise):
        text = self.embedding(text)
        x = torch.cat((noise, text), dim = 1)
        x = self.dense(x)
        x = x.view(x.size(0), self.channel * 8, 4, 4)

        return self.main(x)


class T2IDiscriminator(nn.Module):
    def __init__(self, in_channel, channel, text_length, sigmoid_used):
        super( T2IDiscriminator, self).__init__()

        self.in_channel = in_channel
        self.channel = channel
        self.text_length = text_length
        self.sigmoid_used = sigmoid_used

        self.embedding = nn.Linear(text_length, channel * 4)
        self.main = nn.Sequential(
                nn.Conv2d(in_channel, channel, kernel_size = 5, stride = 2, padding = 2, bias = False),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(channel, channel * 2, kernel_size = 5, stride = 2, padding = 2, bias = False),
                nn.BatchNorm2d(channel * 2),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(channel * 2, channel * 4, kernel_size = 5, stride = 2, padding = 2, bias = False),
                nn.BatchNorm2d(channel * 4),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(channel * 4, channel * 8, kernel_size = 5, stride = 2, padding = 2, bias = False),
                nn.BatchNorm2d(channel * 8),
                nn.LeakyReLU(0.2, inplace = True),
                )

        self.conv = nn.Conv2d((channel * 8 + channel * 4), channel * 8, 
                kernel_size = 1, stride = 1, padding = 0, bias = False)
        self.lrelu = nn.LeakyReLU(0.2, inplace = True)
        self.dense = nn.Linear(channel * 8 * 4 * 4, 1)

    def forward(self, image, text):
        text_embedding = self.embedding(text)
        text_embedding = text_embedding.view(-1, 256, 1, 1).repeat(1, 1, 4, 4)

        x = self.main(image)
        x = torch.cat((x, text_embedding), dim = 1)
        x = self.conv(x)
        x = self.lrelu(x).view(x.size(0), -1)
        x = self.dense(x)

        if self.sigmoid_used:
            return torch.sigmoid(x)
        else:
            return x


