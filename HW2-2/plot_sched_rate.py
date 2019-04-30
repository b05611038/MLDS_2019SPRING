import sys
import csv
import copy
import numpy as np
import pandas as pd

from lib.visualize import TrainHistoryPlot

def from_dataframe(dataframe):
    #only return numpy array
    seq = []
    for i in range(len(dataframe)):
        seq.append(dataframe.iloc[i])

    return np.asarray(seq)

def grab_rate(dataframe_list, sys_argv):
    if len(dataframe_list) != 1:
        raise RuntimeError('Please check the args for only one file.')

    rate = []
    name = []

    rate.append(from_dataframe(dataframe_list[0].iloc[:, 0]))
    rate.append(from_dataframe(dataframe_list[0].iloc[:, 1]))

    name.append('sched_rate')

    return rate, name

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: python3 plot_sched_rate.py [image name] [csv file]')
        exit(0)

    dataframe_list = []
    args = []
    dataframe_list.append(pd.read_csv(sys.argv[2]))
    args.append(sys.argv[2])

    rate, name = grab_rate(dataframe_list, args)
    TrainHistoryPlot(rate, name, 'Sched_sampling_rate', 'Rate', ['epoch', 'rate'])
    print('All process done.')


