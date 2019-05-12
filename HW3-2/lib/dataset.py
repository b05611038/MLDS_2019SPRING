import os
import csv
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as tfs
from torch.utils.data import Dataset
from PIL import Image

from lib.utils import *

class Text2ImageDataset(Dataset):
    def __init__(self, transform = None, mode = 'sample', data_path = './data'):
        # path of data is the directory with sub-directory images/ for training images
        # also have the tags of pairing image ./path/tags
        if transform is None:
            self.transform = tfs.Compose([tfs.ToTensor(), tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transform = transform

        if mode not in ['sample', 'batch']:
            raise ValueError('Please input correct dataset mode. [sample or batch]')

        self.mode = mode
        self.data_path = data_path
        self.images = self._grab_images(data_path)
        eyes, hairs = self._tags_info()
        self.tags = self._grab_tags(data_path, eyes, hairs)

    def __getitem__(self, index):
        #real image
        if self.mode == 'sample':
            select = random.randint(0, len(self.images) - 1)
            image = self.transform(self.images[select])
            label = self.tags[select]
        elif self.mode == 'batch':
            image = self.transform(self.images[index])
            label = self.tags[index]

        return image, label

    def __len__(self):
        return len(self.images)

    def _grab_tags(self, path, eyes, hairs):
        f = open(path + '/tags.csv', 'r')
        text = f.readlines()
        f.close()

        tags = {}
        for line in text:
            context = line.replace('\n', '')split(',')
            index = int(context[0])
            feature = context[1].split(' ')
            hair = feature[0]
            eye = feature[2]
            for i in range(len(hairs)):
                if hair == hairs[i]:
                    hair = i
                    break

            for i in range(len(eyes)):
                if eye == eyes[i]:
                    eye = i
                    break

            tags[index] = hair * len(eyes) + eye

        return tags
            
    def _grab_images(self, path):
        path = path + '/images'
        name_list = []
        files = os.listdir(path)
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                name_list.append(path + '/' + file)

        images = {}
        for image in name_list:
            images[int(image.split('/')[-1: ][0].replace('.jpg', ''))] = \
                    Image.open(image).resize((64, 64), Image.BILINEAR)

        return images

    def _tags_info(self):
        eyes = ['aqua', 'black', 'blue', 'brown', 'green', 'orange', 'pink', 'purple', 'red', 'yellow']
        hairs = ['aqua', 'gray', 'green', 'orange', 'red', 'white', 'black', 'blonde', 'blue', 'brown',
                'pink', 'purple']

        return eyes, hairs


