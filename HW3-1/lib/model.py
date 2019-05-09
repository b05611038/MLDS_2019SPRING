import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import *


__all__ = ['GAN', 'DCGAN', 'WGAN', 'WGAN_GP']


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

class originalGAN(nn.Module):
    def __init__(self, distribution, device, latent_length = 100, image_size = (3, 64, 64), sigmoid_use = True, init_weight = True):
        super(originalGAN, self).__init__()

        self.distribution = distribution
        self.device = device
        self.latent_length = latent_length
        self.image_size = image_size
        self.sigmoid_use = sigmoid_use
        self.init_weight = init_weight

        self.generator = Generator(latent_length, image_size)
        self.discriminator = Discriminator(image_size, sigmoid_use)

        if init_weight:
            self.generator.apply(weights_init)
            self.discriminator.apply(weights_init)

        self.generator = self.generator.float().to(self.device)
        self.discriminator = self.discriminator.float().to(self.device)

    def forward(self, feed, mode = 'generate'):
        if mode not in ['generate', 'discriminate']:
            raise ValueError('Please check the model mode, [generate or discrimiate].')

        if mode == 'generate':
            latent_vector = self._latent_random(feed, self.distribution)
            return self.generator(latent_vector)
        elif mode == 'discriminate':
            return self.discriminator(feed)
        else:
            raise RuntimeError('Please check the model mode.')

    def _latent_random(self, numbers, distribution):
        if distribution == 'uniform':
            latent = np.random.uniform(-1, 1, (numbers, self.latent_length))
            return torch.tensor(latent).float().to(self.device)
        elif distribution == 'normal':
            latent = np.random.normal(0, 1, (numbers, self.latent_length))
            return torch.tensor(latent).float().to(self.device)
        elif distribution == 'oth_normal':
            latent = np.random.normal(0, 1, (numbers, self.latent_length))
            latent = orthogonal(latent)
            return torch.tensor(latent).float().to(self.device)
        elif distribution == 'torch':
            latent = torch.randn(numbers, self.latent_length)
            return latent.float().to(self.device)
        else:
            raise RuntimeError("Can't generate random latent vector.")

class Discriminator(nn.Module):
    def __init__(self, image_size, sigmoid_use):
        super(Discriminator, self).__init__()

        self.image_size = image_size
        self.sigmoid_use = sigmoid_use

        self.main = nn.Sequential(
            nn.Linear(int(np.prod(image_size)), 512),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace = True),
            nn.Linear(256, 1)
        )

        if sigmoid_use:
            self.main.add_module('5', nn.Sigmoid())

    def forward(self, image):
        image = image.view(image.size(0), -1)
        score = self.main(image)

        return score

class Generator(nn.Module):
    def __init__(self, latent_length, image_size):
        super(Generator, self).__init__()

        self.latent_length = latent_length
        self.image_size = image_size

        def block(in_feat, out_feat, normalize = True):
            layers = [nn.Linear(in_feat, out_feat)]
            if normalize:
                layers.append(nn.BatchNorm1d(out_feat, 0.8))

            layers.append(nn.LeakyReLU(0.2, inplace = True))
            return layers

        self.main = nn.Sequential(
            *block(self.latent_length, 128, normalize = False),
            *block(128, 256),
            *block(256, 512),
            *block(512, 1024),
            nn.Linear(1024, int(np.prod(self.image_size))),
            nn.Tanh()
        )

    def forward(self, feed):
        image = self.main(feed)
        image = image.view(image.size(0), *self.image_size)
        return image

class dcGAN(nn.Module):
    def __init__(self, distribution, device, latent_length = 100, in_channel = 3, channel = 128, init_weight = True):
        super(dcGAN, self).__init__()

        self.distribution = distribution
        self.device = device
        self.latent_length = latent_length
        self.in_channel = in_channel
        self.channel = channel
        self.init_weight = init_weight

        self.generator = DCGenerator(channel, in_channel, latent_length)
        self.discriminator = DCDiscriminator(in_channel, channel)

        if init_weight:
            self.generator.apply(weights_init)
            self.discriminator.apply(weights_init)

        self.generator = self.generator.float().to(self.device)
        self.discriminator = self.discriminator.float().to(self.device)

    def forward(self, feed, mode = 'generate'):
        if mode not in ['generate', 'discriminate']:
            raise ValueError('Please check the model mode, [generate or discrimiate].')

        if mode == 'generate':
            latent_vector = self._latent_random(feed, self.distribution)
            return self.generator(latent_vector)
        elif mode == 'discriminate':
            return self.discriminator(feed)
        else:
            raise RuntimeError('Please check the model mode.')

    def _latent_random(self, numbers, distribution):
        if distribution == 'uniform':
            latent = np.random.uniform(-1, 1, (numbers, self.latent_length, 1, 1))
            return torch.tensor(latent).float().to(self.device)
        elif distribution == 'normal':
            latent = np.random.normal(0, 1, (numbers, self.latent_length, 1, 1))
            return torch.tensor(latent).float().to(self.device)
        elif distribution == 'oth_normal':
            latent = np.random.normal(0, 1, (numbers, self.latent_length, 1, 1))
            latent = orthogonal(latent)
            return torch.tensor(latent).float().to(self.device)
        elif distribution == 'torch':
            latent = torch.randn(numbers, self.latent_length, 1, 1)
            return latent.float().to(self.device)
        else:
            raise RuntimeError("Can't generate random latent vector.")


class DCGenerator(nn.Module):
    def __init__(self, channel, out_channel, latent_length):
        super(DCGenerator, self).__init__()

        self.channel = channel
        self.out_channel = out_channel
        self.latent_length = latent_length

        self.main = nn.Sequential(
                # size: [batch * latent_length * 1 * 1]
                nn.ConvTranspose2d(latent_length, channel * 8, kernel_size = 4, stride = 1, padding = 0, bias = False),
                nn.BatchNorm2d(channel * 8),
                nn.ReLU(inplace = True),
                # size: [batch * (channel * 8) * 4 * 4]
                nn.ConvTranspose2d(channel * 8, channel * 4, kernel_size = 4, stride = 2, padding = 1, bias = False),
                nn.BatchNorm2d(channel * 4),
                nn.ReLU(inplace = True),
                # size: [batch * (channel * 4) * 8 * 8]
                nn.ConvTranspose2d(channel * 4, channel * 2, kernel_size = 4, stride = 2, padding = 1, bias = False),
                nn.BatchNorm2d(channel * 2),
                nn.ReLU(inplace = True),
                # size: [batch * (channel * 2) * 16 * 16]
                nn.ConvTranspose2d(channel * 2, channel, kernel_size = 4, stride = 2, padding = 1, bias = False),
                nn.BatchNorm2d(channel),
                nn.ReLU(inplace = True),
                # size: [batch * channel * 32 * 32]
                nn.ConvTranspose2d(channel, out_channel, kernel_size = 4, stride = 2, padding = 1, bias = False),
                nn.Tanh()
                # size: [batch * out_channel * 64 * 64]
                )

    def forward(self, latent_vector):
        return self.main(latent_vector)

class DCDiscriminator(nn.Module):
    def __init__(self, in_channel, channel):
        super(DCDiscriminator, self).__init__()

        self.in_channel = in_channel
        self.channel = channel

        self.main = nn.Sequential(
                nn.Conv2d(in_channel, channel, kernel_size = 4, stride = 2, padding = 1, bias = False),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(channel, channel * 2, kernel_size = 4, stride = 2, padding = 1, bias = False),
                nn.BatchNorm2d(channel * 2),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(channel * 2, channel * 4, kernel_size = 4, stride = 2, padding = 1, bias = False),
                nn.BatchNorm2d(channel * 4),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(channel * 4, channel * 8, kernel_size = 4, stride = 2, padding = 1, bias = False),
                nn.BatchNorm2d(channel * 8),
                nn.LeakyReLU(0.2, inplace = True),
                nn.Conv2d(channel * 8, 1, kernel_size = 4, stride = 1, padding = 0, bias = False),
                nn.Sigmoid()
                )

    def forward(self, image):
        return self.main(image).view(image.size(0), -1)

def _gan(arch, distribution, device, **kwargs):
    model_list = ['GAN', 'WGAN', 'WGAN_GP']
    if arch not in model_list:
        raise ValueError('Please choose correct GAN structure. [DCGAN, WGAN, WGAN_GP]')

    model = originalGAN(distribution, device, **kwargs)

    return model

def GAN(distribution, device):
    return _gan('GAN', distribution, device, latent_length = 100, image_size = (3, 64, 64), sigmoid_use = True, init_weight = True)

def DCGAN(distribution, device):
    return dcGAN(distribution, device, latent_length = 100, in_channel = 3, channel = 64, init_weight = True)

def WGAN(distribution, device):
    return _gan('WGAN', distribution, device, latent_length = 100, image_size = (3, 64, 64), sigmoid_use = False, init_weight = True)

def WGAN_GP(distribution, device):
    return _gan('WGAN_GP', distribution, device, latent_length = 100, image_size = (3, 64, 64), sigmoid_use = False, init_weight = True)


