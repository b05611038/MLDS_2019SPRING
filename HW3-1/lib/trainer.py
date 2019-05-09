import os
import csv
import numpy as np
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import torchvision
import torchvision.transforms as tfs

from torch.optim import SGD, Adam, RMSprop
from torch.utils.data import DataLoader

from lib.utils import *
from lib.dataset import *
from lib.model import *
from lib.visualize import *

class GANTrainer():
    def __init__(self, model_type, model_name, distribution, dataset_mode, switch_ratio, device, img_path = './image'):
        if distribution not in ['uniform', 'normal', 'oth_normal', 'torch']:
            raise ValueError('Please input correct sample distribution. [uniform, normal, torch]')

        self.distribution = distribution
        self.env = self._env_setting(device)

        self.model_type = model_type
        self.model_name = model_name
        self.switch_ratio = switch_ratio
        self.img_path = img_path
        self.loss_layer = None
        if self._check_continue_training(model_name):
            self.model = self._select_model(model_type)
            self.model = self._load_model(model_name)
        else:
            self.model = self._select_model(model_type)

        self._select_optim(model_type)
        self.dataset_mode = dataset_mode
        self.dataset = AnimeFaceDataset(mode = dataset_mode)

    def train(self, epochs, batch_size, save = True):
        if batch_size == 1:
            raise RuntimeError("For training batchnorm layer, batch can't set as 1.")

        self.dataloader = DataLoader(self.dataset, batch_size = batch_size, shuffle = False)
        self.history = ['Epoch,Generator Loss,Discriminator Loss\n']
        self.model.to(self.env)
        self.model.device = self.env
        print('Model Structure:')
        print(self.model)

        for epoch in range(epochs):
            self._epoch(batch_size, epoch)

        self._save_history()
        torch.save(self.model, self.model_name + '.pkl')

        print('All training process done.')

    def _epoch(self, batch_size, epoch_iter):
        dis_loss = []
        dis_total = []
        gen_loss = []
        gen_total = []
        self.model = self.model.train()
        for iter, (image, label) in enumerate(self.dataloader):
            #discriminate
            self.optim_D.zero_grad()

            image = image.float().to(self.env)
            label = label.float().to(self.env)

            fake_image = self.model(image.size(0))
            fake_image = fake_image.detach().to(self.env)
            fake_label = torch.zeros_like(label).to(self.env)

            real_dis = self.model(image, mode = 'discriminate')
            fake_dis = self.model(fake_image, mode = 'discriminate')

            d_loss = self._calculate_loss([real_dis, fake_dis], [label, fake_label], [image, fake_image],
                    self.model.discriminator, self.model_type, mode = 'discriminate')

            d_loss.backward()
            self.model.generator.zero_grad()
            dis_loss.append(d_loss.detach() * image.size(0))
            dis_total.append(image.size(0))

            self.optim_D.step()

            if self.model_type == 'WGAN':
                for p in self.model.discriminator.parameters():
                    p.data.clamp_(-0.01, 0.01)

            if iter % self.switch_ratio == 0:
                #generate
                self.optim_G.zero_grad()

                new_label = torch.ones_like(label).float().to(self.env)
                fake_img = self.model(image.size(0))
                fake_dis = self.model(fake_img, mode = 'discriminate')

                g_loss = self._calculate_loss([None, fake_dis], [new_label, None], [None, None],
                    self.model.discriminator, self.model_type, mode = 'generate')

                g_loss.backward()
                self.model.discriminator.zero_grad()
                gen_loss.append(g_loss.detach() * image.size(0))
                gen_total.append(image.size(0))
                self.optim_G.step()

            if iter % 10 == 0:
                print('Epoch', epoch_iter + 1, '| Iter', iter, 
                        '| Generator loss: %.6f' % g_loss.detach(),
                        '| Discriminator loss: %.6f' % d_loss.detach())
        if epoch_iter % 5 == 4:
            self.model = self.model.eval()
            img_tensor = self.model(64).cpu()
            GeneratorImage(img_tensor, self.img_path + '/' + self.model_name + '_E' + str(epoch_iter + 1) + '.png',
                    show = False, save = True)

        gen_loss = torch.tensor(gen_loss).sum() / torch.tensor(gen_total).sum()
        dis_loss = torch.tensor(dis_loss).sum() / torch.tensor(dis_total).sum()
        self.history.append(str(epoch_iter + 1) + ',' + str(gen_loss.detach().numpy()) +
                ',' + str(dis_loss.detach().numpy()) + '\n')

        print('-' * 120 + '\n')
        print('Epoch', epoch_iter + 1, '| Generator loss: %.6f |' % gen_loss, 'Discriminator loss: %.6f' % dis_loss)
        print('\n' + '-' * 120)

    def _save_history(self):
        f = open(self.model_name + '.csv', 'w')
        f.writelines(self.history)
        f.close()

    def _calculate_loss(self, input_data, target, images, D, model_type, mode):
        #image and target are list object for [discriminator, generator]
        if mode == 'discriminate':
            #input_data: [discriminator out(real), discriminator out(fake)]
            #target: [real_label, fake_label]
            #images: [real_image, fake_image]
            if model_type == 'GAN':
                loss = (self.loss_layer(input_data[0], target[0]) + self.loss_layer(input_data[1], target[1])) / 2
            elif model_type == 'DCGAN':
                loss = (self.loss_layer(input_data[0], target[0]) + self.loss_layer(input_data[1], target[1])) / 2
            elif model_type == 'WGAN':
                loss = -torch.mean(input_data[0]) + torch.mean(input_data[1])
            elif model_type == 'WGAN_GP':
                gradient_penalty = self._gradient_penalty(D, images[0], images[1])
                loss = -torch.mean(input_data[0]) + torch.mean(input_data[1]) + self.lambda_gp * gradient_penalty

            return loss

        elif mode == 'generate':
            if model_type == 'GAN':
                loss = self.loss_layer(input_data[1], target[0])
            elif model_type == 'DCGAN':
                loss = self.loss_layer(input_data[1], target[0])
            elif model_type == 'WGAN':
                loss = -torch.mean(input_data[1])
            elif model_type == 'WGAN_GP':
                loss = -torch.mean(input_data[1])

            return loss

        else:
            raise RuntimeError('Please check source code for mode selection.')

    def _gradient_penalty(self, D, real_image, fake_image):
        # Random weight term for interpolation between real and fake samples
        alpha = torch.tensor(np.random.random((real_image.size(0), 1, 1, 1))).float().to(self.env)
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_image + ((1 - alpha) * fake_image)).requires_grad_(True)
        d_interpolates = D(interpolates)
        fake = torch.ones(real_image.size(0), 1).float().requires_grad_(False).to(self.env)
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
        if model_type == 'GAN':
            self.optim_G = Adam(self.model.generator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
            self.optim_D = Adam(self.model.discriminator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
        elif model_type == 'DCGAN':
            self.optim_G = Adam(self.model.generator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
            self.optim_D = Adam(self.model.discriminator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
            #self.optim_D = SGD(self.model.discriminator.parameters(), lr = 0.00005)
        elif model_type == 'WGAN':
            self.optim_G = RMSprop(self.model.generator.parameters(), lr = 0.00005)
            self.optim_D = RMSprop(self.model.discriminator.parameters(), lr = 0.00005)
        elif model_type == 'WGAN_GP':
            self.optim_G = Adam(self.model.generator.parameters(), lr = 0.0002, betas = (0.5, 0.999))
            self.optim_D = Adam(self.model.discriminator.parameters(), lr = 0.0002, betas = (0.5, 0.999))

    def _select_model(self, model_type):
        if model_type not in ['GAN', 'DCGAN', 'WGAN', 'WGAN_GP']:
            raise ValueError('Please select correct GAN model. [GAN, DCGAN, WGAN, WGAN_GP]')

        if model_type == 'GAN':
            model = GAN(self.distribution, self.env)
            self.loss_layer = nn.BCELoss()
        elif model_type == 'DCGAN':
            model = DCGAN(self.distribution, self.env)
            self.loss_layer = nn.BCELoss()
        elif model_type == 'WGAN':
            model = WGAN(self.distribution, self.env)
            self.loss_layer = 'Check GANTrainer._calculate_loss().'
        elif model_type == 'WGAN_GP':
            model = WGAN_GP(self.distribution, self.env)
            self.lambda_gp = 10
            self.loss_layer = 'Check GANTrainer._calculate_loss().'

        return model

    def _load_model(self, model_name):
        model = torch.load(model_name + '.pkl')
        print('Load model', model_name, 'success.')
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

        if (model_name + '.pkl') in files:
            return True
        else:
            return False

    def _env_setting(self, device):
        if device < 0:
            env = torch.device('cpu')
            print('Envirnment setting done, using device: cpu')
        else:
            torch.backends.cudnn.benchmark = True
            cuda.set_device(device)
            env = torch.device('cuda:' + str(device))
            print('Envirnment setting done, using device: cuda:' + str(device))

        return env


