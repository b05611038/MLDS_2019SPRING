import sys
import numpy as np
import torch

from lib.visualize import ModelPredictFunctionPlot

def load_model(model_name):
    if type(model_name) is not list:
        raise TypeError('Please check the innput model')

    model = []
    for i in range(len(model_name)):
        model.append(torch.load(model_name[i]))

    return model

def target_function(x):
    y = 4 * np.sin(np.exp(x / 8)) - 5 * np.cos(x) + np.exp(x / 8)

    return y

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('Usage: python3 demo.py [image name] [model 1 path] [model 2 path] ...')
        exit(0)

    model_name = []
    for i in range(2, len(sys.argv)):
        model_name.append(sys.argv[i])

    model = load_model(model_name)
    ModelPredictFunctionPlot(model, model_name, target_function, sys.argv[1], 'Model demo')
    print('All process done.')
