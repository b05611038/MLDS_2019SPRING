import numpy as np
import torch
from torch.utils.data import Dataset

class VCdataset(Dataset):
    def __init__(self, data, label, mark):
        self.data = data
        self.label = label
        self.mark = mark

    def __getitim__(self, index):
        pass

    def __len__(self):
        pass


