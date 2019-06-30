import time
import argparse

from lib.utils import *
from lib.trainer import QTrainer

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
    return preprocess_dict

def init_parser(main):
    parser = argparse.ArgumentParser()

    parser.add_argument('model_type', type = str, help = 'One model can choose. [baseline]')
    parser.add_argument('model_name', type = str, help = 'Model name of the model.')
    parser.add_argument('Algorithm', type = str, help = 'Implemented Q-learning base algorithm. [Q_l1, Q_l2, Q_l1_target, Q_l2_target]')
    parser.add_argument('device', type = int, help = 'device choosing for training. [-1 is cpu]')

    parser.add_argument('--optimizer', type = str, default = 'Adam', help = 'The optimizer you can choose.')
    parser.add_argument('--iterations', type = int, default = 30000, help = 'How many episode to train your policy net.')
    parser.add_argument('--buffer_size', type = int, default = 10000, help = 'Teh maximum data size can put into replaybuffer.')
    parser.add_argument('--episode_size', type = int, default = 2, help = 'How many games to play in an episode.')
    parser.add_argument('--batch_size', type = int, default = 128, help = 'The mini-batch_size wants to used in one iteration.')
    parser.add_argument('--checkpoint', type = int, default = 3000, help = 'The interval of saving a model checkpoint.')
    parser.add_argument('--random_action', type = str2bool, default = True,
            help = 'Method of agent action space exploring, if true, the random probability would start from 1.0.')
    parser.add_argument('--slice_scoreboard', type = str2bool, default = True,
            help = 'Method of image preprocess, if true, the scoreboard part of image would not feed into model.')
    parser.add_argument('--gray_scale', type = str2bool, default = True,
            help = 'Method of image preprocess, if true, the input image would from RGB -> Gray scale.')
    parser.add_argument('--minus_observation',  type = str2bool, default = True,
            help = 'Method of image preprocess, if true, input image would become the last state - now state.')
    parser.add_argument('--decay_by_time', type = str2bool, default = True,
            help = 'Method of reward process, if true, reward would decay by time step.')

    opt = parser.parse_args()
    print(opt)

    return opt

if __name__ == '__main__':
    start_time = time.time()
    opt = init_parser(__name__)
    observation_dict = construct_observation_preprocess_dict([opt.slice_scoreboard, opt.gray_scale, opt.minus_observation])
    reward_dict = construct_reward_preprocess_dict([opt.decay_by_time])
    trainer = QTrainer(opt.model_type, opt.model_name, opt.buffer_size, opt.random_action, observation_dict, reward_dict, opt.device,
            optimizer = opt.optimizer, policy = opt.Algorithm)
    trainer.play(opt.iterations, opt.episode_size, opt.batch_size, opt.checkpoint)
    trainer.save_config(opt)
    print('All process done, cause %s seconds.' % (time.time() - start_time))


