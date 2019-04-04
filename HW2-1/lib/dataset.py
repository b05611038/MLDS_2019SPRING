import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.utils import *

class VCDataSet(Dataset):
    def __init__(self, data_path, label_path, mask_max, mark, mode = 'fix', save_path = './data/dataset'):
        self.data_path = data_path
        self.label_path = label_path
        self.mask_max = mask_max
        self.mark = mark
        self.mode = mode
        self.save_path = save_path

        self.feature = self._get_feature(data_path)
        self.label = load_object(label_path)
        self.index_list = list(self.label.keys())
        self.data, self.guided_seq, self.mask, self.label, self.mask_label = self._build(self.index_list, save_path)

    def __getitem__(self, index):
        data = torch.tensor(self.data[:, index, :])
        out_seq = torch.tensor(self.guided_seq[:, index, :])
        mask = torch.tensor(self.mask[index, :, :]).byte()
        label = torch.tensor(self.label[index, :])
        mask_label = torch.tensor(self.mask_label[index, :]).byte()

        return data, out_seq, mask, label, mask_label

    def __len__(self):
        return len(self.label)

    def _build(self, index_list, save_path):
        forward_label = []
        forward_seq = []
        mask_index = []
        max_seq = 0
        data = np.empty((80, len(index_list), 4096))
        for i in range(len(index_list)):
            seq = self._index_choose(self.mode, self.label[index_list[i]])
            mask_index.append(seq.shape[0])
            if max_seq < seq.shape[0]:
                max_seq = seq.shape[0]

            forward_seq.append(seq[0: seq.shape[0] - 1])
            forward_label.append(seq[1: seq.shape[0]])
            data[:, i: i + 1, :] = self.feature[index_list[i]].reshape(80, 1, 4096)

        mask = np.ones((len(forward_seq), max_seq - 1, self.mask_max))
        final_seq = np.zeros((max_seq - 1, len(forward_seq), 1))
        mask_label = np.ones((len(forward_seq), max_seq - 1))
        final_label = np.zeros((len(forward_seq), max_seq - 1))
        for i in range(len(forward_seq)):
            final_seq[0: forward_seq[i].shape[0], i, 0] = forward_seq[i]
            final_label[i, 0: forward_label[i].shape[0]] = forward_label[i]
            mask[i, mask_index[i]:, :] = 0
            mask_label[i, mask_index[i]: ] = 0

        return data, final_seq, mask, final_label, mask_label
            
    def _index_choose(self, mode, label_list):
        if mode == 'fix':
            return label_list[0]
        elif mode == 'random':
            select = random.randint(0, len(label_list) - 1)
            return label_list[select]
        else:
            raise RuntimeError('Please select correct mode of the label random choosing, fix or random.')

    def _get_feature(self, feature_path):
        feature = {}
        for file in os.listdir(feature_path):
            if file.endswith('.npy'):
                feature[file.replace('.npy', '')] = np.load(feature_path + '/' + file)

        return feature


