import sys

from lib.utils import *
from lib.word2vec import Word2vec

if __name__ == '__main__':
    if len(sys.argv) < 4:
        print('Usage: python3 text_preprocessing.py [word2vec method] [preprocess] [least count]')
        print('preprocess is the args for using the file of TA summary. [true or false]')
        exit(0)

    if sys.argv[2].lower() == 'true':
        preprocess = True
    elif  sys.argv[2].lower() == 'false':
        preprocess = False
    else:
        raise ValueError('Please input correct args [preprocess], true or false.')

    if not preprocess and len(sys.argv) < 6:
        print('Usage: python3 text_preprocessing.py [word2vec method] [preprocess] [least count] [max_length] [min_length]')
        exit(0)


    if preprocess:
        word2vec = Word2vec(embedding = sys.argv[1], path = './data/clr_conversation.txt', 
                least_freq = int(sys.argv[3]), 
                seq_length_min = 2, 
                seq_length_max = 15)

        word2vec.combine_qa('./data/question.txt', './data/answer.txt', save = './data/sentence_pair.pkl')
        
    else:
        word2vec = Word2vec(embedding = sys.argv[1], path = './data/clr_conversation.txt',
                least_freq = int(sys.argv[3]), 
                seq_length_min = int(sys.argv[5]), 
                seq_length_max = int(sys.argv[4]))

        word2vec.sentence_pair('./data/clr_conversation.txt', save = './data/sentence_pair.pkl')

    save_object('word2vec.pkl', word2vec)


