import os
import sys
import time
import numpy as np
import pandas as pd
import torch

from lib.utils import *
from lib.visualize import GenSenPlot

def GrabDataframe(file_list):
    dataframe = []
    for i in range(len(file_list)):
        dataframe.append(pd.read_csv(file_list[i]))

    return dataframe

def FromDataframe(dataframe):
    train_loss = dataframe.iloc[len(dataframe) - 1, 1]
    train_acc = dataframe.iloc[len(dataframe) - 1, 2]
    sen = dataframe.iloc[len(dataframe) - 1, 3]
    test_loss = dataframe.iloc[len(dataframe) - 1, 4]
    test_acc = dataframe.iloc[len(dataframe) - 1, 5]

    return [sen, train_loss, train_acc, test_loss, test_acc]

def OutcomeConcatenate(dataframe_list):
    sen = []
    train_loss = []
    train_acc = []
    test_loss = []
    test_acc = []

    for i in range(len(argv)):
        temp = FromDataframe(dataframe_list[i])
        sen.append(temp[0])
        train_loss.append(temp[1])
        train_acc.append(temp[2])
        test_loss.append(temp[3])
        test_acc.append(temp[4])

    sen = np.array(sen)
    train_loss = np.array(train_loss)
    train_acc = np.array(train_acc)
    test_loss = np.array(test_loss)
    test_acc = np.array(test_acc)

    return sen, train_loss, train_acc, test_loss, test_acc

def GetXAxis(argv, sep):
    argv.sort(key = lambda argv: float(argv.split('.csv')[0].split(sep)[1]))
    axis = []
    for i in range(len(argv)):
        axis.append(float(argv[i].split('.csv')[0].split(sep)[1]))

    return argv, np.array(axis)

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: python3 plot_Gsen.py [image name] [sep letter] [HP name] [csv file] ...')
        exit(0)

    start_time = time.time()
    argv = []
    for i in range(4, len(sys.argv)):
        argv.append(sys.argv[i])

    argv, x_axis = GetXAxis(argv, sys.argv[2])
    dataframe_list = GrabDataframe(argv)
    sen, train_loss, train_acc, test_loss, test_acc = OutcomeConcatenate(dataframe_list)
    GenSenPlot([np.log10(x_axis), [sen], [train_loss, test_loss]],
            [['sensitivity'], ['train_loss', 'test_loss']],
            sys.argv[1] + '_loss', 'Sensitivity vs Loss', [sys.argv[3] + '(log_scale)', 'Sensitivity', 'Loss'])

    GenSenPlot([np.log10(x_axis), [sen], [train_acc, test_acc]],
            [['sensitivity'], ['train_acc', 'test_acc']],
            sys.argv[1] + '_acc', 'Sensitivity vs Acc.', [sys.argv[3] + '(log_scale)', 'Sensitivity', 'Acc.'])

    print('All process done, cause %s seconds.' % (time.time() - start_time))


