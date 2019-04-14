import sys

from lib.utils import *
from lib.word2vec import Word2vec

if __name__ == '__main__':
    if len(sys.argv) < 5:
        print('Usage: python3 text_preprocessing.py [word2vec method] [least count] [max_length] [min_length]')
        exit(0)

    word2vec = Word2vec(embedding = sys.argv[1], path = './data/clr_conversation.txt',
            least_freq = int(sys.argv[2]), 
            seq_length_min = int(sys.argv[4]), 
            seq_length_max = int(sys.argv[3]))

    word2vec.sentence_pair('./data/clr_conversation.txt', save = './data/sentence_pair.pkl')
    save_object('word2vec.pkl', word2vec)

