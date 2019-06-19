import sys

from lib.utils import *
from lib.word2vec import Word2vec

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: python3 text_preprocessing.py [word2vec method] [least count] [max_length] [min_length]')
        exit(0)

    word2vec = Word2vec(path = './data/training_label.json', embedding = sys.argv[1], least_freq = int(sys.argv[2]), seq_max_length = int(sys.argv[3]), seq_min_length = int(sys.argv[4]))
    word2vec.label_dict('./data/training_label.json', save = 'training_label_dict.pkl')
    word2vec.label_dict('./data/testing_label.json', save = 'testing_label_dict.pkl')
    save_object('word2vec.pkl', word2vec)


