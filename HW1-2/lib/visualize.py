import csv
import numpy as np
import torch

import matplotlib.pyplot as plt

color_bar = ['black', 'red', 'blue', 'green', 'brown', 'yellow', 'cyan', 'magenta']

def TrainHistoryPlot(his, his_label, save_name, title, axis_name, save = True):
    #history must be input as list[0]: iter or epoch
    #and otehr of history list is the acc or loss of different model
    plt.figure(figsize = (10, 4))
    for i in range(1, len(his)):
        plt.plot(his[0], his[i])

    plt.title(title)
    plt.xlabel(axis_name[0])
    plt.ylabel(axis_name[1])
    plt.legend(his_label, loc = 'upper right')
    if save:
        plt.savefig(save_name + '.png')
        print('Picture: ' + save_name + '.png done.')
    else:
        plt.show()

def ModelWeightPlot(reduced_weight, model_name, save_name, title, save = True):
    #the plot of the reduced dimensional weight vector
    if len(reduced_weight) > 8:
        raise IndexError('Default colors are up to eight colors.')

    if len(reduced_weight) != len(model_name):
        raise RuntimeError('Please check the list of model and model_name.')

    plt.figure(figsize = (10, 8))
    for weight in range(len(reduced_weight)):
        for state in range(reduced_weight[weight].shape[0]):
            if state == 0:
                plt.scatter(reduced_weight[weight][state, 0], reduced_weight[weight][state, 1], c = color_bar[weight], label = model_name[weight])
            else:
                plt.scatter(reduced_weight[weight][state, 0], reduced_weight[weight][state, 1], c = color_bar[weight])

    plt.title(title)
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.legend(loc = 'upper right')
    if save:
        plt.savefig(save_name + '.png')
        print('Picture: ' + save_name + '.png done.')
    else:
        plt.show()

def MinimumRatioPlot(minimum_ratio, loss, save_name, save = True):
    #the plot will plot the scatter plot of minimum ratio vs loss
    if minimum_ratio.shape[0] != loss.shape[0]:
        raise RuntimeError('Please check the loss and minimum ratio array.')

    plt.figure(figsize = (10, 8))
    plt.scatter(minimum_ratio, loss)
    #for i in range(minimum_ratio.shape[0]):
    #    plt.scatter(minimum_ratio[i], loss[i])

    plt.xlabel('minimum_ratio')
    plt.ylabel('loss')
    plt.ylim(28.38825, 28.38925)
    if save:
        plt.savefig(save_name + '.png')
        print('Picture: ' + save_name + '.png done.')
    else:
        plt.show()


