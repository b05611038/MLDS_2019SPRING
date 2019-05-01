import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['GAN', 'DCGAN', 'WGAN', 'WGAN_GP']


class GAN(nn.Module):
    def __init__(self, latent_length = 100, in_channel = 3, channel = 128, sigmoid_use = True, init_weight = True):
        super(DCGAN, self).__init__()

        self.latent_length = latent_length
        self.in_channel = in_channel
        self.channel = channel
        self.sigmoid_use = sigmoid_use
        self.init_weight = init_weight

        self.generator = Generator(channel, in_channel, latent_length)
        self.discriminator = Discriminator(in_channel, channel, sigmoid_use)

        if init_weight:
            self._weights_init(self.generator)
            self._weights_init(self.discriminator)

    def forward(self, input, mode = 'generate'):
        if mode not in ['generate', 'discriminate']:
            raise ValueError('Please check the model mode, [generate or discrimiate].')

        if mode == 'generate':
            return self.generator(input)
        elif mode == 'discriminate':
            return self.discriminator(input)
        else:
            raise RuntimeError('Please check the model mode.')

    def _weights_init(model):
        class_name = model.__class__.__name__
        if class_name.find('Conv') != -1:
            nn.init.normal_(model.weight.data, 0.0, 0.02)
        elif class_name.find('BatchNorm') != 1:
            nn.init.normal_(model.weight.data, 1.0, 0.02)
            nn.init.constant_(model.bias.data, 0)


class Generator(nn.Module):
    def __init__(self, channel, out_channel, latent_length):
        super(Generator, self).__init__()

        self.channel = channel
        self.out_channel = out_channel
        sefl.latent_length = latent_length

        self.gen = nn.Sequential(
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
                nn.BatchNorm2d(channel * 2),
                nn.ReLU(inplace = True),
                # size: [batch * channel * 32 * 32]
                nn.ConvTranspose2d(channel, out_channel, kernel_size = 4, stride = 2, padding = 1, bias = False),
                nn.Tanh()
                # size: [batch * out_channel * 64 * 64]
                )

        def forward(self, latent_vector):
            return self.gen(latent_vector)


class Discriminator(nn.Module):
    def __init__(self, in_channel, channel, sigmoid_use):
        super(Discriminator, self).__init__()

        self.in_channel = in_channel
        self.channel = channel

        if sigmoid_used:
            self.dis = nn.Sequential(
                    nn.Conv2d(in_channel, channel, kernel_size = 4, stride = 2, padding = 1, bias = False),
                    nn.BatchNorm2d(channel),
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
        else:
            self.dis = nn.Sequential(
                    nn.Conv2d(out_channel, channel, kernel_size = 4, stride = 2, padding = 1, bias = False),
                    nn.BatchNorm2d(channel),
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
                    )

        def forward(self, image):
            return self.dis(image).squeeze()

def _gan(arch, **kwargs):
    model_list = ['DCGAN', 'WGAN', 'WGAN_GP']
    if arch not in model_list:
        raise ValueError('Please choose correct GAN structure. [DCGAN, WGAN, WGAN_GP]')
    model = GAN(**kwargs)

    return model

def DCGAN(**kwargs):
    return _gan('DCGAN', latent_length = 100, in_channel = 3, channel = 128, sigmoid_use = True, init_weight = True)

def WGAN(**kwargs):
    return _gan('DCGAN', latent_length = 100, in_channel = 3, channel = 128, sigmoid_use = False, init_weight = True)

def WGAN_GP(**kwargs):
    return _gan('DCGAN', latent_length = 100, in_channel = 3, channel = 128, sigmoid_use = False, init_weight = True)


