import time
import argparse

from lib.utils import *
from lib.trainer import PGTrainer

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

def construct_reward_preprocess_dict(args):
    preprocess_dict = {}
    preprocess_dict['time_decay'] = args[0]
    preprocess_dict['normalized'] = args[1]
    return preprocess_dict

def init_parser(main):
    parser = argparse.ArgumentParser()

    parser.add_argument('model_type', type = str, help = 'One model can choose. [baseline]')
    parser.add_argument('model_name', type = str, help = 'Model name of the model.')
    parser.add_argument('Algorithm', type = str, help = 'Implemented Policy gradient base algorithm. [PO, PPO, PPO2]')
    parser.add_argument('device', type = int, help = 'device choosing for training. [-1 is cpu]')

    parser.add_argument('--optimizer', type = str, default = 'Adam', help = 'The optimizer you can choose.')
    parser.add_argument('--iterations', type = int, default = 10000, help = 'How many episode to train your policy net.')
    parser.add_argument('--episode_size', type = int, default = 4, help = 'How many games to play in an episode.')
    parser.add_argument('--checkpoint', type = int, default = 1000, help = 'The interval of saving a model checkpoint.')
    parser.add_argument('--slice_scoreboard', type = str2bool, default = True,
            help = 'Method of image preprocess, if true, the scoreboard part of image would not feed into model.')
    parser.add_argument('--gray_scale', type = str2bool, default = True,
            help = 'Method of image preprocess, if true, the input image would from RGB -> Gray scale.')
    parser.add_argument('--minus_observation',  type = str2bool, default = True,
            help = 'Method of image preprocess, if true, input image would become the last state - now state.')
    parser.add_argument('--reward_normalize', type = str2bool, default = True,
            help = 'Method of reward process, if true, reward would be normalize by batch.')
    parser.add_argument('--decay_by_time', type = str2bool, default = True,
            help = 'Method of reward process, if true, reward would decay by time step.')

    opt = parser.parse_args()
    print(opt)

    return opt

if __name__ == '__main__':
    start_time = time.time()
    opt = init_parser(__name__)
    observation_dict = construct_observation_preprocess_dict([opt.slice_scoreboard, opt.gray_scale, opt.minus_observation])
    reward_dict = construct_reward_preprocess_dict([opt.decay_by_time, opt.reward_normalize])
    print(observation_dict, reward_dict)
    trainer = PGTrainer(opt.model_type, opt.model_name, observation_dict, reward_dict, opt.device,
            optimizer = opt.optimizer, policy = opt.Algorithm)
    trainer.play(opt.iterations, opt.episode_size, opt.checkpoint)
    print('All process done, cause %s seconds.' % (time.time() - start_time))


