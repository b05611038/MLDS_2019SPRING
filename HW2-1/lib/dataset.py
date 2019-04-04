import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.utils import *

class VCDataSet(Dataset):
    def __init__(self, data_path, label_path, mark, mode = 'fix', save_path = './data/dataset'):
        self.data_path = data_path
        self.label_path = label_path
        self.mark = mark
        self.mode = mode
        self.save_path = save_path

        self.feature = self._get_feature(data_path)
        self.label = load_object(label_path)
        self.index_list = list(self.label.keys())
        self.data, self.out_seq, self.mask, self.label = self._build_matrix(self.index_list, save_path)

    def __getitem__(self, index):
        data = self.data[:, index, :]
        out_seq = self.out_seq[:, index, :]
        mask = self.mask[index]
        label = self.label[index]

        return torch.tensor(data), torch.tensor(out_seq), torch.tensor(mask), torch.tensor(label) 

    def _check_exist(self, save_path):
        check = [False, False, False, False]
        for file in os.listdir(save_path):
            if file == 'data.npy':
                check[0] = True
            if file == 'out_seq.npy':
                check[1] = True
            if file == 'mask.npy':
                check[2] = True
            if file == 'label.npy':
                check[3] = True

        if check[0] and check[1] and check[2] and check[3]:
            return True
        else:
            return False

    def _build_matrix(self, index_list, save_path):
        if self._check_exist(save_path):
            return np.load(save_path + '/data.npy'), np.load(save_path + '/out_seq.npy'), np.load(save_path + '/mask.npy'), np.load(save_path + '/label.npy')
        else:
            data = None
            label = []
            mask = []
            guide_seq = []
            max_seq = 0
            for i in range(len(index_list)):
                seq = self._index_choose(self.mode, self.label[index_list[i]])
                if max_seq < seq.shape[0]:
                    max_seq = seq.shape[0]

                for time_state in range(1, seq.shape[0]):
                    guide_seq.append(seq)
                    label.append(seq[time_state])
                    mask.append(time_state - 1)

                data_temp = self.feature[index_list[i]].reshape(80, 1, 4096)
                for j in range(seq.shape[0] - 1):
                    if data is None:
                        data = data_temp
                    else:
                        data = np.concatenate((data, data_temp), axis = 1)

            mask = np.array(mask)
            label = np.array(label)
            out_seq = np.zeros((max_seq, len(guide_seq)))
            for i in range(len(guide_seq)):
                out_seq[0: guide_seq[i].shape[0], i] = guide_seq[i]

            np.save(save_path + '/data.npy', data)
            np.save(save_path + '/out_seq.npy', out_seq)
            np.save(save_path + '/mask.npy', mask)
            np.save(save_path + '/label.npy', label)

            return data, out_seq, mask, label

    def __len__(self):
        return len(self.label)

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


