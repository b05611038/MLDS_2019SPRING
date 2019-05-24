import time
import argparse

from lib.utils import *
from lib.trainer import PGTrainer

def init_parser(main):
    parser = argparse.ArgumentParser()
    parser.add_argument('model_type', type = str, help = 'One model can choose. [baseline]')
    parser.add_argument('model_name', type = str, help = 'Model name of the model.')
    parser.add_argument('device', type = int, help = 'device choosing for training. [-1 is cpu]')

    opt = parser.parse_args()
    print(opt)

    return opt

if __name__ == '__main__':
    start_time = time.time()
    opt = init_parser(__name__)

    print('All process done, cause %s seconds.' % (time.time() - start_time))
