import csv
import numpy as np
import torch

import matplotlib.pyplot as plt

def TrainHistoryPlot(his, his_label, save_name, title, axis_name, save = True):
    #history must be input as list[0]: iter or epoch
    #and otehr of history list is the acc or loss of different model
    plt.figure(figsize = (10, 6))
    for i in range(1, len(his)):
        plt.plot(his[0][0: 100], his[i][0: 100])

    plt.title(title)
    plt.xlabel(axis_name[0])
    plt.ylabel(axis_name[1])
    plt.legend(his_label, loc = 'upper right')
    if save:
        plt.savefig(save_name + '.png')
        print('Picture: ' + save_name + '.png done.')
    else:
        plt.show()

def GenParaPlot(paras, y, label, save_name, title, axis_name, save = True):
    color = ['blue', 'orange']
    plt.figure(figsize = (10, 6))
    for i in range(len(y)):
        plt.scatter(paras, y[i], label = label[i], c = color[i])
    plt.title(title)
    plt.xlabel('parameters')
    plt.ylabel(axis_name)
    plt.legend(loc = 'upper right')
    if save:
        plt.savefig(save_name + '.png')
        print('Picture: ' + save_name + '.png done.')
    else:
        plt.show()

def GenSenPlot(plot_paras, label, save_name, title, axis_name, save = True):
    #plot_paras is the list of [x_axis, [left axis list], [right axis list]]
    #label is the list of [[left label], [right label]]
    #axis_name is the list of [x_axis, y_left, y_right]
    #max of the record data sublist is two
    if len(plot_paras[1]) > 2 or len(plot_paras[2]) > 2:
        raise IndexError('Please check the list of the plot_paras.')
    if len(axis_name) > 3:
        raise IndexError('Please check the list of the axis_name.')
    #if len(label[0]) != len(plot_paras[1]) or len(label[1]) != len(plot_paras[2]):
    #    raise IndexError('Please check the label and the plot_paras in same shape.')

    color = ['red', 'blue']
    style = ['-', '--']
    fig, ax1 = plt.subplots()

    #left plot of the figure
    ax1.set_xlabel(axis_name[0])
    ax1.set_ylabel(axis_name[1])
    for i in range(len(plot_paras[1])):
        ax1.plot(plot_paras[0], plot_paras[1][i], label = label[0][i], color = color[0], linestyle = style[i])

    ax1.tick_params(axis = 'y', labelcolor = color[0])
    plt.legend(loc = 'upper right')

    #right plot of the figure
    ax2 = ax1.twinx()
    ax2.set_ylabel(axis_name[2])
    for i in range(len(plot_paras[2])):
        ax2.plot(plot_paras[0], plot_paras[2][i], label = label[1][i], color = color[1], linestyle = style[i])

    ax2.tick_params(axis = 'y', labelcolor = color[1])
    plt.legend(loc = 'upper left')

    fig.tight_layout()

    if save:
        plt.savefig(save_name + '.png')
        print('Picture: ' + save_name + '.png done.')
    else:
        plt.show()


