import os
import random
import numpy as np
import torch
import torchvision
import torchvision.transforms as tfs
from torch.utils.data import Dataset
from PIL import Image

from lib.utils import *

class AnimeFaceDataset(Dataset):
    def __init__(self, transform = None, mode = 'sample', data_path = './data/faces'):
        if transform is None:
            self.transform = tfs.Compose([tfs.ToTensor(), tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        else:
            self.transform = transform

        if mode not in ['sample', 'batch']:
            raise ValueError('Please input correct dataset mode. [sample or batch]')

        self.mode = mode
        self.data_path = data_path
        self.images = self._grab_image(data_path)

    def __getitem__(self, index):
        #real image
        if self.mode == 'sample':
            select = random.randint(0, len(self.images) - 1)
            image = self.transform(self.images[select])
        elif self.mode == 'batch':
            image = self.transform(self.images[index])

        label = torch.tensor([1]).float()

        return image, label

    def __len__(self):
        return len(self.images)

    def _grab_image(self, path):
        name_list = []
        files = os.listdir(path)
        for file in files:
            if file.endswith('.jpg') or file.endswith('.png'):
                name_list.append(path + '/' + file)

        images = []
        for image in name_list:
            images.append(Image.open(image).resize((64, 64), Image.BILINEAR))

        return images


