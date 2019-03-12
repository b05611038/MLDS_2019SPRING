import sys
import time
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from lib.model import ANN
from lib.utils import *
from lib.visualize import MinimumRatioPlot

def gernerate_name(name, points = 100):
    model_name = []
    for i in range(1, points + 1):
        model_name.append(name + '_paras_ver' + str(i) + '.pkl')

    return model_name

def calculate_minimum_ratio(model_name, loss_list, dataset, criterion, sample_num, device):
    if device < 0:
        env = torch.device('cpu')
        print('Envirnment setting done, using device: cpu')
    else:
        torch.backends.cudnn.benchmark = True
        cuda.set_device(device)
        env = torch.device('cuda:' + str(device))
        print('Envirnment setting done, using device: CUDA_' + str(device))

    data_loader = DataLoader(dataset, batch_size = dataset.get_num(), shuffle = False)

    minimum_ratio = []
    for ver in range(len(model_name)):
        model = ANN()
        model.to(env)
        model.load_state_dict(torch.load(model_name[ver]))

        minimum_num = 0
        for sample in range(sample_num):
            for index, layer in enumerate(model.parameters()):
                noise = (random.random() - 0.5) * 2
                layer.data.add_(noise)

            loss = 0
            for iter, data in enumerate(data_loader):
                x, y = data
                x = x.to(env)
                y = y.to(env)

                out = model(x)

                loss += criterion(out, y)

            loss = float(loss.cpu().detach().numpy())

            if loss > loss_list[ver]:
                minimum_num += 1
            if sample % 100 == 0 and sample != 0:
                print('Model version: ', ver + 1, '| Minimum ratio: %4f' % (minimum_num / sample), '| Progress:', sample, '/', sample_num)

        minimum_ratio.append(float(minimum_num / sample_num))

    minimum_ratio = np.asarray(minimum_ratio)

    return minimum_ratio

def target_function(x):
    y = 4 * np.sin(np.exp(x / 8)) - 5 * np.cos(x) + np.exp(x / 8)

    return y

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python3 plot_minimum_ratio.py [image name] [model name] [sample point] [device]')
        exit(0)

    start_time = time.time()
    model_name = gernerate_name(sys.argv[2])
    minimum_ratio = calculate_minimum_ratio(model_name, np.load('loss.npy'), load_object(sys.argv[2] + '_dataset.pkl'), nn.MSELoss(), int(sys.argv[3]), int(sys.argv[4]))
    np.save('minimum_ratio.npy', minimum_ratio)
    MinimumRatioPlot(minimum_ratio, np.load('loss.npy'), sys.argv[1])
    print('All process done, cause %s seconds.' % (time.time() - start_time))


