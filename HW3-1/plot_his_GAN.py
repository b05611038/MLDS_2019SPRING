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

def grab_his(dataframe_list, sys_argv, mode):
    his = []
    for i in range(len(dataframe_list)):
        if i == 0 and mode == 'loss':
            his.append(from_dataframe(dataframe_list[i].iloc[:, 0]))
            his.append(from_dataframe(dataframe_list[i].iloc[:, 1]))
            his.append(from_dataframe(dataframe_list[i].iloc[:, 2]))
        elif i != 0 and mode == 'loss':
            if (from_dataframe(dataframe_list[i].iloc[:, 0]) == his[0]).all():
                his.append(from_dataframe(dataframe_list[i].iloc[:, 1]))
                his.append(from_dataframe(dataframe_list[i].iloc[:, 2]))
            else:
                raise RuntimeError('Please check the file or all trained by same training setting.')
        else:
            raise RuntimeError('Please check the args of python command line.')

        his_label = [sys_argv[i].replace('his_', '').replace('.csv', '') + '_generator',
                sys_argv[i].replace('his_', '').replace('.csv', '') + '_discriminator']

    return his, his_label

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 plot_his_GAN.py [image name] [csv file 1] [csv file 2] ...')
        exit(0)

    dataframe_list = []
    args = []
    for i in range(2, len(sys.argv)):
        dataframe_list.append(pd.read_csv(sys.argv[i]))
        args.append(sys.argv[i])

    his, his_label = grab_his(dataframe_list, args, 'loss')
    TrainHistoryPlot(his, his_label, sys.argv[1] + '_loss', 'Loss', ['epoch', 'loss'])
    print('All process done.')


