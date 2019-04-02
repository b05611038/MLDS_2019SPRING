import numpy as np
import torch
from torch.utils.data import Dataset

from lib.utils import *

class VCdataset(Dataset):
    def __init__(self, data_path, label, mark):
        self.data_path = data_path
        self.label_path = label_path
        self.mark = mark

    def __getitim__(self, index):
        pass

    def __len__(self):
        pass

class VCtesting(Dataset):
    def __init__(self, data_path = './data/testing_data/feat', label_path = './data/testing_label.json', mark = 'test'):
        self.data_path = data_path
        self.label_path = label_path
        self.mark = mark

    def __getitim__(self, index):
        pass

    def __len__(self):
        pass
