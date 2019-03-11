import csv
import numpy as np
import torch

import matplotlib.pyplot as plt

def TrainHistoryPlot(his, his_label, save_name, title, axis_name, save = True):
    #history must be input as list[0]: iter or epoch
    #and otehr of history list is the acc or loss of different model
    plt.figure(figsize = (10, 6))
    for i in range(1, len(his)):
        plt.plot(his[0], his[i])

    plt.title(title)
    plt.xlabel(axis_name[0])
    plt.ylabel(axis_name[1])
    plt.legend(his_label, loc = 'upper left')
    if save:
        plt.savefig(save_name + '.png')
        print('Picture: ' + save_name + '.png done.')
    else:
        plt.show()

def ModelPredictFunctionPlot(model, model_name, func, save_name, title, save = True):
    #to plot the demo of the model in funciton fitting
    if len(model) != len(model_name):
        raise RuntimeError('Please check the model list.')

    plt.figure(figsize = (10, 6))
    domain = np.arange(0, 20, 0.01)
    outcome = [func(domain)]
    plt.plot(domain, outcome[0], label = 'Ground truth')
    for i in range(1, len(model) + 1):
        outcome.append(model[i - 1](torch.tensor(domain).view(-1, 1).float()))
        outcome[i] = outcome[i].view(-1, ).detach().numpy()

        plt.plot(domain, outcome[i], label = model_name[i - 1])

    plt.title(title)
    plt.xlabel('input')
    plt.ylabel('output')
    plt.legend(loc = 'upper right')
    if save:
        plt.savefig(save_name + '.png')
        print('Picture: ' + save_name + '.png done.')
    else:
        plt.show()


