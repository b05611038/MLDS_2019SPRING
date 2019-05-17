import os
import time
import argparse
from lib.utils import *
from lib.trainer import Text2ImageGANTrainer

def init_parser(main):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type = str, help = 'Four kinds of GAN can choose. [GAN, WGAN, WGAN_GP]')
    parser.add_argument('model_name', type = str, help = 'model saving name.')
    parser.add_argument('device', type = int, help = 'device choosing for training. [-1 is cpu]')
    parser.add_argument('--distribution', type = str, default = 'torch', help = 'distribution of generator use [normal or uniform].')
    parser.add_argument('--dataset_mode', type = str, default = 'sample', help = 'how dataset grab data. [sample or batch]')
    parser.add_argument('--dataset', type = str, default = 'old', help = 'choosing dataset for training. [old or new]')
    parser.add_argument('--switch_ratio', type = int, default = 1, help = 'the switch ratio of training generator and discriminator')
    parser.add_argument('--epochs', type = int, default = 200, help = 'number of epochs of training.')
    parser.add_argument('--batch_size', type = int, default = 128, help = 'size of the batches.')
    parser.add_argument('--noise_length', type = int, default = 100, help = 'dimensionality of the latent space.')
    opt = parser.parse_args()
    print(opt)

    return opt

if __name__ == '__main__':
    start_time = time.time()

    opt = init_parser(__name__)
    trainer = Text2ImageGANTrainer(opt.model_type, opt.model_name, opt.distribution, opt.noise_length, 
            opt.dataset, opt.dataset_mode, opt.switch_ratio, opt.device)
    trainer.train(opt.epochs, opt.batch_size)
    print('All process done, cause %s seconds.' % (time.time() - start_time))


