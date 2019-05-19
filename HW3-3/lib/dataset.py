"""
Copyright (C) 2018 NVIDIA Corporation.  All rights reserved.
Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
"""
import torch.utils.data as data
import os.path

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath = line.strip()
            imlist.append(impath)

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        impath = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)

        return img

    def __len__(self):
        return len(self.imlist)


class ImageLabelFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(os.path.join(self.root, flist))
        self.transform = transform
        self.loader = loader
        self.classes = sorted(list(set([path.split('/')[0] for path in self.imlist])))
        self.class_to_idx = {self.classes[i]: i for i in range(len(self.classes))}
        self.imgs = [(impath, self.class_to_idx[impath.split('/')[0]]) for impath in self.imlist]

    def __getitem__(self, index):
        impath, label = self.imgs[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)

    return images


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = sorted(make_dataset(root))
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in: " + root + "\n"
                               "Supported image extensions are: " +
                               ",".join(IMG_EXTENSIONS)))

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.return_paths:
            return img, path
        else:
            return img

    def __len__(self):
        return len(self.imgs)

###############################################################################
# Code by self
# for checking different features in dataset
###############################################################################

import numpy as np
from sklearn.model_selection import train_test_split

class Generate_dataset_list():

    def __init__(self, image_path = './data/images', feature_path = './data/features.txt'):

        self.dataset_feature = ['5_o_Clock_Shadow', 'Arched_Eyebrows', 'Attractive', 'Bags_Under_Eyes', 'Bald',
                'Bangs', 'Big_Lips', 'Big_Nose', 'Black_Hair', 'Blond_Hair',
                'Blurry', 'Brown_Hair', 'Bushy_Eyebrows', 'Chubby', 'Double_Chin',
                'Eyeglasses', 'Goatee', 'Gray_Hair', 'Heavy_Makeup', 'High_Cheekbones',
                'Male', 'Mouth_Slightly_Open', 'Mustache', 'Narrow_Eyes', 'No_Beard',
                'Oval_Face', 'Pale_Skin', 'Pointy_Nose', 'Receding_Hairline', 'Rosy_Cheeks',
                'Sideburns', 'Smiling', 'Straight_Hair', 'Wavy_Hair', 'Wearing_Earrings',
                'Wearing_Hat', 'Wearing_Lipstick', 'Wearing_Necklace', 'Wearing_Necktie', 'Young']

        self.image_path = image_path
        self.feature_path = feature_path
        self.num_lines, self.feature_text = self._grab_image_features(feature_path)

    def build_dataset(self, feature, split_ratio = 0.001, save = './data/dataset'):
        if feature not in self.dataset_feature:
            raise ValueError('Please input correct feature.')

        for i in range(len(self.dataset_feature)):
            if self.dataset_feature[i] == feature:
                index = i
                break

        exist = []
        non_exist = []
        for i in range(len(self.feature_text)):
            content = self.feature_text[i].replace('\n', '').split(' ')
            content = [i for i in content if i != '']

            if content[index + 1] == '1':
                exist.append(self.image_path + '/' + content[0] + '\n')
            elif content[index + 1] == '-1':
                non_exist.append(self.image_path + '/' + content[0] + '\n')
            else:
                raise RuntimeError("Can't grab feature from feature file.")

        train_a, test_a = train_test_split(exist, test_size = split_ratio, shuffle = False)
        train_b, test_b = train_test_split(non_exist, test_size = split_ratio, shuffle = False)

        file_name = [save + '/' + feature + '_' + 'list_trainA.txt', save + '/' + feature + '_' + 'list_testA.txt', 
                save + '/' + feature + '_' + 'list_trainB.txt', save + '/' + feature + '_' + 'list_testB.txt']
        dataset = [train_a, test_a, train_b, test_b]

        for i in range(len(dataset)):
            f = open(file_name[i], 'w')
            f.writelines(dataset[i])
            f.close()

        print('All process done')

    def _grab_image_features(self, path):
        f = open(path, 'r')
        text = f.readlines()
        f.close()

        num_lines = int(text[0])
        text = text[2: ]
        if num_lines != len(text):
            raise RuntimeError('Please check the data feature file.')

        return num_lines, text


