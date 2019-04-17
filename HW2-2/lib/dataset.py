import os
import random
import numpy as np
import torch
from torch.utils.data import Dataset

from lib.utils import *

class CBDataset(Dataset):
    def __init__(self, sentence_pair_path, mask_max, seq_length_max, save_path = './data/dataset'):
        self.sentence_pair_path = sentence_pair_path
        self.mask_max = mask_max
        self.save_path = save_path

        if not os.path.exists(save_path):
            os.makedirs(save_path)

        self.sentence_pair = load_object(self.sentence_pair_path)  #list object [last sentence, next sentnece]
        self.data, self.guided_seq, self.guided_mask, self.label_seq, self.label_mask = self._build(self.sentence_pair)

    def __getitem__(self, index):
        data = torch.tensor(self.data[:, index, :])
        guided_seq = torch.tensor(self.guided_seq[:, index, :])
        guided_mask = torch.tensor(self.guided_mask[index, :])
        guided_mask = guided_mask.unsqueeze(1).repeat([1, self.mask_max]).byte()
        label_seq = torch.tensor(self.label_seq[index, :])
        label_mask = torch.tensor(self.label_mask[index, :]).byte()

        return data, guided_seq, guided_mask, label_seq, label_mask

    def __len__(self):
        return len(self.sentence_pair)

    def _check_init_his(self, save_path):
        check = [False, False, False, False, False]
        if os.path.isfile(save_path + '/data.npy'):
            check[0] = True
        if os.path.isfile(save_path + '/guided_seq.npy'):
            check[1] = True
        if os.path.isfile(save_path + '/guided_mask.npy'):
            check[2] = True
        if os.path.isfile(save_path + '/label_seq.npy'):
            check[3] = True
        if os.path.isfile(save_path + '/label_mask.npy'):
            check[4] = True

        if check[0] and check[2] and check[2] and check[3] and check[4]:
            return True, [np.load(save_path + '/data.npy'), np.load(save_path + '/guided_seq.npy'), np.load(save_path + '/guided_mask.npy'), np.load(save_path + '/label_seq.npy'), np.load(save_path + '/label_mask.npy')]
        else:
            return False, []

    def _build(self, sentence_pair, save = True):
        check, arr = self._check_init_his(self.save_path)
        if check:
            data = arr[0]
            guided_seq = arr[1]
            guided_mask = arr[2]
            label_seq = arr[3]
            label_mask = arr[4]
        else:
            mask_index = []
            forward_guided = []
            forward_label = []
            max_data_seq = 0
            max_guided_seq = 0
            for i in range(len(sentence_pair)):
                last = sentence_pair[i][0]
                then = sentence_pair[i][1]
                mask_index.append(then.shape[0] - 1)

                if last.shape[0] > max_data_seq:
                    max_data_seq = last.shape[0]

                if then.shape[0] > max_guided_seq:
                    max_guided_seq = then.shape[0]

                forward_guided.append(last[0: then.shape[0] - 1])
                forward_label.append(last[1: then.shape[0]])

            data = np.empty((max_data_seq, len(sentence_pair), 1))
            guided_seq = np.zeros((max_guided_seq - 1, len(sentence_pair), 1)) #<bos>, ... (no <eos>)
            # guided_mask = np.ones((len(sentence_pair), max_guided_seq - 1, self.mask_max)) 
            # batch * seq * output_length
            # too big in the origin data, extend the data in __getitem__
            guided_mask = np.ones((len(sentence_pair), max_guided_seq - 1))
            label_seq = np.zeros((len(sentence_pair), max_guided_seq - 1)) # for alreadt time flatten data
            label_mask = np.ones((len(sentence_pair), max_guided_seq - 1))
            for i in range(len(sentence_pair)):
                data[0: sentence_pair[i][0].shape[0], i, 0] = sentence_pair[i][0]
                guided_seq[0: forward_guided[i].shape[0], i, 0] = forward_guided[i]
                label_seq[i, 0: forward_label[i].shape[0]] = forward_label[i]
                guided_mask[i, mask_index[i]:] = 0
                label_mask[i, mask_index[i]: ] = 0

            if save:
                np.save(self.save_path + '/data.npy', data)
                np.save(self.save_path + '/guided_seq.npy', guided_seq)
                np.save(self.save_path + '/guided_mask.npy', guided_mask)
                np.save(self.save_path + '/label_seq.npy', label_seq)
                np.save(self.save_path + '/label_mask.npy', label_mask)

        return data, guided_seq, guided_mask, label_seq, label_mask


