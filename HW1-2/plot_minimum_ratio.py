import sys
import time
import random
import numpy as np
import torch

from lib.model import ANN
from lib.utils import SampleFunctionDataset
from lib.visualize import TrainHistoryPlot

def gernerate_name(name, points = 100):
    model_name = []
    for i in range(1, points + 1):
        model_name.append(name + '_paras_ver' + str(i) + '.pkl')

    return model_name

def minimum_ratio(model_name, loss_list, dataset, criterion, sample_num, device):
    if device < 0:
        env = torch.device('cpu')
        print('Envirnment setting done, using device: cpu')
    else:
        torch.backends.cudnn.benchmark = True
        cuda.set_device(device)
        env = torch.device('cuda:' + str(device))
        print('Envirnment setting done, using device: CUDA_' + str(device))

    dataset.mode = 'train'
    data_loader = DataLoader(train_set, batch_size = dataset.get_num(), shuffle = False)

    for ver in range(len(model_name)):
        model = ANN()
        model.to(env)
        model.load_state_dict(torch.load(model_name[ver]))

        for index, layer in enumerate(model.parameters()):
            noise = (random.random() - 0.5) * 2
            layer.data.add_(noise)

        prediction = model()
