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

def GenParaPlot(paras, y, save_name, title, axis_name, save = True):
    if paras.shape[0] != y.shape[0]:
        raise RuntimeError('Please check the model parameters and loss(Acc.) array.')

    plt.figure(figsize = (10, 6))
    plt.scatter(paras, y)
    plt.title(title)
    plt.xlabel('parameters')
    plt.ylabel(axis_name)
    if save:
        plt.savefig(save_name + '.png')
        print('Picture: ' + save_name + '.png done.')
    else:
        plt.show()


