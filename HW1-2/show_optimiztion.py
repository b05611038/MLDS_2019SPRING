import sys
import numpy as np
import torch

from sklearn.decomposition import PCA

from lib.visualize import ModelWeightPlot

def grab_weight(model, mode):
    weight = torch.tensor([])
    if mode == 'whole':
        for p in model.parameters():
            weight = torch.cat((weight, p.view(-1, )), dim = 0)
    elif mode == 'first':
        for p in model.parameters():
            weight = torch.cat((weight, p.view(-1, )), dim = 0)
            break

    return weight.detach().numpy().reshape(1, -1)

def collect_name(model_name, epoch = 100, interval = 10):
    sub_name = []
    for record in range(interval, epoch + 1, interval):
        sub_name.append(model_name + '_E' + str(record) + '.pkl')

    return sub_name

def gernerate_name(name, amount = 8):
    model_name = []
    for i in range(amount):
        model_name.append(name + '_' + str(i + 1))

    return model_name

def reduced_weight(model_name, mode):
    weight = None
    count = []
    for i in range(len(model_name)):
        sub_name = collect_name(model_name[i])
        count.append(len(sub_name))
        for j in range(len(sub_name)):
            model = torch.load(sub_name[j], map_location = 'cpu')
            model.eval()
            if weight is None:
                weight = grab_weight(model, mode)
            else:
                weight = np.concatenate((weight, grab_weight(model, mode)), 0)

    pca_weight = PCA(n_components = 2, whiten = True).fit_transform(weight)
    reduced_weight = []
    index_iter = 0
    for i in range(len(count)):
        reduced_weight.append(weight[index_iter: index_iter + count[i], :])
        index_iter += count[i]

    return reduced_weight

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage python3 show_optimization.py [image name] [model save title] [grab range]')
        exit(0)

    model_name = gernerate_name(sys.argv[2])
    weight = reduced_weight(model_name, sys.argv[3])
    ModelWeightPlot(weight, model_name, sys.argv[1], 'optimization process by PCA')
    print('All process done.')


