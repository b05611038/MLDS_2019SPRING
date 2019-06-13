import time
import argparse

from lib.utils import *
from lib.shower import QShower

def str2bool(v):
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def construct_observation_preprocess_dict(args):
    preprocess_dict = {}
    preprocess_dict['slice_scoreboard'] = args[0]
    preprocess_dict['gray_scale'] = args[1]
    preprocess_dict['minus_observation'] = args[2]
    return preprocess_dict

def init_parser(main):
    parser = argparse.ArgumentParser()

    parser.add_argument('model_type', type = str, help = 'One model can choose. [baseline]')
    parser.add_argument('model_name', type = str, help = 'Model name of the model.')
    parser.add_argument('device', type = int, help = 'device choosing for training. [-1 is cpu]')

    parser.add_argument('--sample_times', type = int, default = 4, help = 'How many games to check a checkpoint model.')
    parser.add_argument('--slice_scoreboard', type = str2bool, default = True,
            help = 'Method of image preprocess, if true, the scoreboard part of image would not feed into model.')
    parser.add_argument('--gray_scale', type = str2bool, default = True,
            help = 'Method of image preprocess, if true, the input image would from RGB -> Gray scale.')
    parser.add_argument('--minus_observation',  type = str2bool, default = True,
            help = 'Method of image preprocess, if true, input image would become the last state - now state.')

    opt = parser.parse_args()
    print(opt)

    return opt

if __name__ == '__main__':
    start_time = time.time()
    opt = init_parser(__name__)
    observation_dict = construct_observation_preprocess_dict([opt.slice_scoreboard, opt.gray_scale, opt.minus_observation])
    shower = QShower(opt.model_type, opt.model_name, observation_dict, opt.device)
    shower.show(opt.sample_times)
    print('All process done, cause %s seconds.' % (time.time() - start_time))


