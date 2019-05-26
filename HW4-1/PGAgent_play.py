import time
import argparse

from lib.utils import *
from lib.trainer import PGTrainer

def init_parser(main):
    parser = argparse.ArgumentParser()

    parser.add_argument('model_type', type = str, help = 'One model can choose. [baseline]')
    parser.add_argument('model_name', type = str, help = 'Model name of the model.')
    parser.add_argument('Algorithm', type = str, help = 'Implemented Policy gradient base algorithm. [PPO]')
    parser.add_argument('device', type = int, help = 'device choosing for training. [-1 is cpu]')

    parser.add_argument('--optimizer', type = str, default = 'Adam', help = 'The optimizer you can choose.')
    parser.add_argument('--slice_scoreboard', type = bool, default = True,
            help = 'Method of image preprocess, if true, the scoreboard part of image would not feed into model.')
    parser.add_argument('--gray_scale', type = bool, default = True,
            help = 'Method of image preprocess, if true, the input image would from RGB -> Gray scale.')
    parser.add_argument('--minus_observation',  type = bool, default = True,
            help = 'Method of image preprocess, if true, input image would become the last state - now state.')
    parser.add_argument('--reward_normalize', type = bool, default = True,
            help = 'Method of reward process, if true, reward would be normalize by batch.')
    parser.add_argument('--decay_by_time', type = bool, default = True,
            help = 'Method of reward process, if true, reward would decay by time step.')

    opt = parser.parse_args()
    print(opt)

    return opt

if __name__ == '__main__':
    start_time = time.time()
    opt = init_parser(__name__)

    print('All process done, cause %s seconds.' % (time.time() - start_time))
