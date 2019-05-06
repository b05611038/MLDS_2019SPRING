import os
import time
import argparse
from lib.utils import *
from lib.trainer import GANTrainer

def init_parser(main):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type = str, help = 'Four kinds of GAN can choose. [GAN, DCGAN, WGAN, WGAN_GP]')
    parser.add_argument('model_name', type = str, help = 'model saving name.')
    parser.add_argument('distribution', type = str, help = 'distribution of generator use [normal or uniform].')
    parser.add_argument('device', type = int, help = 'device choosing for training. [-1 is cpu]')
    parser.add_argument('--dataset_mode', type = str, default = 'sample', help = 'how dataset grab data. [sample or batch]')
    parser.add_argument('--switch_ratio', type = int, default = 1, help = 'the switch ratio of training generator and discriminator')
    parser.add_argument('--epochs', type = int, default = 2000, help = 'number of epochs of training.')
    parser.add_argument('--batch_size', type = int, default = 256, help = 'size of the batches.')
    parser.add_argument('--latent_dim', type = int, default = 100, help = 'dimensionality of the latent space.')
    opt = parser.parse_args()
    print(opt)

    return opt

if __name__ == '__main__':
    start_time = time.time()

    opt = init_parser(__name__)
    trainer = GANTrainer(opt.model_type, opt.model_name, opt.distribution,
            opt.dataset_mode, opt.switch_ratio, opt.device)
    trainer.train(opt.epochs, opt.batch_size)
    print('All process done, cause %s seconds.' % (time.time() - start_time))


