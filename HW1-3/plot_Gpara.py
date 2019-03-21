import sys
import time
import numpy as np
import pandas as pd
import torch

from lib.utils import *
from lib.visualize import GenParaPlot

def GrabDataframe(file_list):
    dataframe = []
    for i in range(len(file_list)):
        dataframe.append(pd.read_csv(file_list[i]))

    return dataframe

def OutcomeConcatenate(dataframe_list, mode):
    paras = []
    outcome = [[], []]
    if mode == 'loss':
        target = [1, 3]
        temp = [float('Inf'), float('Inf')] #train, loss
    elif mode == 'acc':
        target = [2, 4]
        temp = [0, 0] #train, loss
    else:
        raise RuntimeError('Please check the mode of function: OutcomeConcatenate.')

    for i in range(len(dataframe_list)):
        paras.append(dataframe_list[i].iloc[len(dataframe_list[i]) - 1, 0])
        for epoch in range(len(dataframe_list[i]) - 1):
            #train
            if mode == 'loss' and dataframe_list[i].iloc[epoch, target[0]] < temp[0]:
                temp[0] = dataframe_list[i].iloc[epoch, target[0]]
            elif mode == 'acc' and dataframe_list[i].iloc[epoch, target[0]] > temp[0]:
                temp[0] = dataframe_list[i].iloc[epoch, target[0]] 

            #test
            if mode == 'loss' and dataframe_list[i].iloc[epoch, target[1]] < temp[1]:
                temp[1] = dataframe_list[i].iloc[epoch, target[1]]
            elif mode == 'acc' and dataframe_list[i].iloc[epoch, target[1]] > temp[1]:
                temp[1] = dataframe_list[i].iloc[epoch, target[1]]

        outcome[0].append(temp[0]) #trian
        outcome[1].append(temp[1]) #test
        
    paras = np.array(paras)
    for i in range(len(outcome)):
        outcome[i] = np.array(outcome[i])

    return paras, outcome

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 plot_Gpara.py [image name] [csv file] ...')
        exit(0)

    start_time = time.time()
    argv = []
    for i in range(2, len(sys.argv)):
        argv.append(sys.argv[i])

    argv.sort(key = lambda argv: int(argv.split('.csv')[0].split('c')[1]))
    dataframe_list = GrabDataframe(argv)
    paras, outcome_loss = OutcomeConcatenate(dataframe_list, 'loss')
    paras, outcome_acc = OutcomeConcatenate(dataframe_list, 'acc')

    GenParaPlot(paras, outcome_loss, ['train', 'test'],sys.argv[1] + '_loss', 'paras vs loss', 'loss')
    GenParaPlot(paras, outcome_acc, ['train', 'test'],sys.argv[1] + '_acc', 'paras vs accuracy', 'Acc.')

    print('All process done, cause %s seconds.' % (time.time() - start_time))


