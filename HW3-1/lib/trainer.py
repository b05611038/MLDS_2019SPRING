import os
import csv
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd

from torch.optim import Adam, RMSprop
from torch.utils.data import DataLoader

from lib.utils import *
from lib.dataset import *
from lib.model import *
from lib.loss import *
from lib.visualize import *


class GANTrainer():
    def __init__(self, model_type, model_name, distribution, dataset_mode, switch_ratio, device):
        if distribution not in ['uniform', 'normal']:
            raise ValueError('Please input correct sample distribution. [uniform or normal]')

        self.distribution = distribution
        self.env = self._env_setting(device)

        self.model_type = model_type
        self.model_name = model_name
        self.switch_ratio = switch_ratio
        self.loss_layer = None
        if _check_continue_training(model_name):
            self.model = self._load_model(model_name)
        else:
            self.model = self._select_model(model_type)

        self._select_optim(model_type)
        self.dataset_mode = dataset_mode
        self.dataset = AnimeFaceDataset(mode = dataset_mode)

    def train(self, name, epochs, batch_size, save = True):
        if batch_size == 1:
            raise RuntimeError("For training batchnorm layer, batch can't set as 1.")

        self.dataloader = DataLoader(self.dataset, batch_size = batch_size, shuffle = False)
        self.history = ['Epoch,Generator Loss,Discriminator Loss\n']
        for epoch in epochs:
            self._epoch(batch_size, epoch)
            pass

    def _epoch(self, batch_size, epoch_iter):
        dis_loss = []
        dis_total = []
        gen_loss = []
        gen_total = []
        for iter, (image, label) in enumerate(self.dataloader):
            #discriminate
            self.optim_D.zero_grad()

            self.model.generator = self.model.generator.eval()
            self.model.discriminator = self.model.discriminator.train()

            image = image.float().to(self.device)
            label = label.float().to(self.device)

            fake_image = self.model(batch_size)
            fake_label = torch.zeros_like(label).to(self.device)

            real_dis = self.model(image, mode = 'discriminate')
            fake_dis = self.model(fake_image, mode = 'discriminate')

            d_loss = self._calculate_loss([real_dis, fake_dis], [label, fake_label], [image, fake_image],
                    self.model.discriminator, self.model_select, mode = 'discriminate')

            d_loss.backward()
            dis_loss.append(d_loss.detach() * image.size(0))
            dis_total.append(image.size(0))
            self.optim_D.step()

            if iter % self.switch_ratio == 0:
                #generate
                self.optim_G.zero_grad()

                self.model.generator = self.model.generator.train()
                self.model.discriminator = self.model.discriminator.eval()

                fake_image = self.model(batch_size)
                fake_dis = self.model(fake_image, mode = 'discriminate')

                g_loss = self._calculate_loss([None, fake_dis], [label, None], [None, None],
                    self.model.discriminator, self.model_select, mode = 'generate')

                g_loss.backward()
                gen_loss.append(g_loss.deatch() * image.size(0))
                gen_tatal.append(image.size(0))
                self.optim_G.step()

            if iter % (self.switch_ratio * 10) == 0:
                print('Epoch %d |' % epoch_iter + 1, 'Iter %d |' % iter + 1, 
                        'Generator loss: %.6f |' % g_loss.detach(),
                        'Discriminator loss: %.6f' % d_loss.detach())

        gen_loss = torch.tensor(gen_loss).sum() / torch.tensor(gen_total).sum()
        dis_loss = torch.tensor(gen_loss).sum() / torch.tensor(gen_total).sum()
        self.history.append([str(epoch_iter + 1) + ',' + str(gen_loss) + ',' + str(dis_loss) + '\n'])
        print('-' * 120 + '\n')
        print('Epoch %d |' % epoch_iter + 1, 'Generator loss: %.6f |' % gen_loss, 'Discriminator loss: %.6f' % dis_loss)
        print('\n' + '-' * 120)

    def _calculate_loss(self, input_data, target, images, D, model_select, mode):
        #image and target are list object for [discriminator, generator]
        if mode == 'discriminate':
            #input_data: [discriminator out(real), discriminator out(fake)]
            #target: [real_label, fake_label]
            #images: [real_image, fake_image]
            if model_select == 'GAN':
                loss = self.loss_layer(input_data[0], target[0]) + self.loss_layer(input_data[1], target[1]) / 2
            elif model_select == 'DCGAN':
                loss = self.loss_layer(input_data[0], target[0]) + self.loss_layer(input_data[1], target[1]) / 2
            elif model_select == 'WGAN':
                loss = -torch.mean(input_data[0]) + torch.mean(input_data[1])
            elif model_select == 'WGAN_GP':
                gradient_penalty = (D, images[0], images[1])
                loss = -torch.mean(input_data[0]) + torch.mean(input_data[1]) + self.lambda_gp * gradient_penalty

            return loss

        elif mode == 'generate':
            if model_select == 'GAN':
                loss = self.loss_layer(input_data[1], target[0])
            elif model_select == 'DCGAN':
                loss = self.loss_layer(input_data[1], target[0])
            elif model_select == 'WGAN':
                loss = -torch.mean(input_data[1])
            elif model_select == 'WGAN_GP':
                loss = -torch.mean(input_data[1])

            return loss

        else:
            raise RuntimeError('Please check source code for mode selection.')

    def _gradient_penalty(self, D, real_image, fake_image):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.tensor(np.andom.random((real_image.size(0), 1, 1, 1))).float().to(self.device)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_image + ((1 - alpha) * fake_image)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.tensor(real_image.size(0), 1).fill_(1.0).float().to(self.device).requires_grad_(False)
        gradients = autograd.grad(
                outputs = d_interpolates,
                inputs = interpolates,
                grad_outputs = fake,
                create_graph = True,
                retain_graph = True,
                only_inputs = True
                )[0]
        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim = 1) - 1) ** 2).mean()
        return gradient_penalty

    def _select_optim(self, model_type):
        if model_select == 'GAN':
            self.optim_G = Adam(self.model.generator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
            self.optim_D = Adam(self.model.discriminator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
        elif model_select == 'DCGAN':
            self.optim_G = Adam(self.model.generator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
            self.optim_D = Adam(self.model.discriminator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
        elif model_select == 'WGAN':
            self.optim_G = RMSprop(self.model.generator.parameters(), lr = 0.00005)
            self.optim_D = RMSprop(self.model.discriminator.parameters(), lr = 0.00005)
        elif model_select == 'WGAN_GP':
            self.optim_G = Adam(self.model.generator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
            self.optim_D = Adam(self.model.discriminator.parameters(), lr = 0.0002, betas = (0.5, 0.999))

    def _select_model(self, model_select):
        if model_select not in ['GAN', 'DCGAN', 'WGAN', 'WGAN_GP']:
            raise ValueError('Please select correct GAN model. [GAN, DCGAN, WGAN, WGAN_GP]')

        if model_select == 'GAN':
            model = GAN()
            self.loss_layer = nn.BCELoss()
        elif model_select == 'DCGAN':
            model = DCGAN()
            self.loss_layer = nn.BCELoss()
        elif model_select == 'WGAN':
            model = WGAN()
            self.loss_layer = 'Check GANTrainer._calculate_loss().'
        elif model_select == 'WGAN_GP':
            model = WGAN_GP()
            self.lambda_gp = 10
            self.loss_layer = 'Check GANTrainer._calculate_loss().'

        return model

    def _load_model(self, model_name):
        model = torch.load(model_name)
        print('Load model', name, 'success.')
        return model

    def _check_continue_training(self, model_name):
        path = model_name.split('/')

        if len(path) == 1:
            files = os.listdir('./')
        else:
            real_path = ''
            for i in range(len(path) - 1):
                if i == len(path) - 2:
                    real_path = real_path + path[i]
                else:
                    real_path = real_path + path[i] + '/'
            files = os.listdir(real_path)

        if model_name in files:
            return True
        else:
            return False

    def _env_setting(device):
        if device < 0:
            env = torch.device('cpu')
            print('Envirnment setting done, using device: cpu')
        else:
            torch.backends.cudnn.benchmark = True
            cuda.set_device(device)
            env = torch.device('cuda:' + str(device))
            print('Envirnment setting done, using device: cuda:' + str(device))

        return env


