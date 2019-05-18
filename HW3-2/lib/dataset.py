import os
import csv
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as tfs
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

from lib.utils import *

class Text2ImageDataset(Dataset):
    def __init__(self, transform = None, mode = 'sample', dataset = 'old'):
        # path of data is the directory with sub-directory images/ for training images
        # also have the tags of pairing image ./path/tags
        if transform is None:
            self.transform = tfs.Compose([tfs.ToTensor(), tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transform = transform

        if mode not in ['sample', 'batch']:
            raise ValueError('Please input correct dataset mode. [sample or batch]')

        self.mode = mode
        if dataset == 'old':
            self.dataset = dataset
            self.data_path = './data/images'
        elif dataset == 'new':
            self.dataset = dataset
            self.data_path = './data/new_images'
        else:
            raise ValueError('Please input correct dataset, [old or new]')

        eyes, hairs = self._tags_info()
        self.text_length = len(eyes) * len(hairs)
        self.images, self.tags = self._build(self.data_path, dataset, eyes, hairs)
        self.key_list = list(self.tags)

    def __getitem__(self, index):
        #real image
        if self.mode == 'sample':
            select = random.randint(0, len(self.tags) - 1)
            select = self.key_list[select]
        elif self.mode == 'batch':
            select = self.key_list[index]

        image = self.transform(self.images[select])
        label_index = self.tags[select]
        label = torch.zeros(self.text_length)
        label[label_index] = 1

        return image, label

    def __len__(self):
        return len(self.tags)

    def _build(self, data_path, dataset, eyes, hairs):
        if dataset == 'old':
            f = open('./data/tags.csv', 'r')
        elif dataset == 'new':
            f = open('./data/new_tags.txt', 'r')

        text = f.readlines()
        f.close()

        tags = {}
        images = {}
        for line in text:
            if dataset == 'old':
                context = line.replace('\n', '').split(',')
            elif dataset == 'new':
                context = line.replace('\n', '').split('|')

            if Path(data_path + '/' + context[0] + '.jpg').exists():
                index = int(context[0])
                images[index] = Image.open(data_path + '/' + context[0] + '.jpg').resize((64, 64), Image.BILINEAR)

                if dataset == 'old':
                    feature = context[1].split(' ')
                    hair = feature[0]
                    eye = feature[2]
                elif dataset == 'new':
                    eye = context[4].split(' ')[0]
                    hair = context[5].split(' ')[0]

                for i in range(len(hairs)):
                    if hair == hairs[i]:
                        hair = i
                        break

                for i in range(len(eyes)):
                    if eye == eyes[i]:
                        eye = i
                        break

                try:
                    tags[index] = hair * len(eyes) + eye
                except TypeError:
                    continue
            else:
                continue

        return images, tags

    def _tags_info(self):
        eyes = ['aqua', 'black', 'blue', 'brown', 'green', 'orange', 'pink', 'purple', 'red', 'yellow']
        hairs = ['aqua', 'gray', 'green', 'orange', 'red', 'white', 'black', 'blonde', 'blue', 'brown',
                'pink', 'purple']

        return eyes, hairs


