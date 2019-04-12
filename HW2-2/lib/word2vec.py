import re
import csv
import numpy as np
import torch

from lib.utils import *

class Word2vec():
    def __init__(self, embedding, path, least_freq, seq_length_min, seq_length_max):
        self.embedding = embedding
        if embedding not in ['one_hot']:
            raise ValueError(self.embedding, 'not in the word embedding method.')

        self.path = path
        self.least_freq = least_freq
        self.seq_length_min = seq_length_min
        self.seq_length_max = seq_length_max

        text = self._from_txt(path)
        word_dict = self._word_dict(text)
        self.__embedding_dict, self.__transberse_embedding, self.seq_max = self._embedding(embedding, word_dict)

    def w2v(self, word):
        word = word.lower()

        try:
            vec = self.__embedding_dict[word]
        except KeyError:
            vec = self.seq_max - 1

        return vec

    def v2w(self, vector):
        if type(vector) == torch.Tensor:
            vector = vector.numpy()
        elif type(vector) == np.ndarray:
            pass
        elif type(vector) == int:
            vector = np.array([vector])
        else:
            raise TypeError('Method: word2vec.v2w() only support specific interger, numpy array or torch tensor.')

        if len(vector.shape) != 1:
            raise RuntimeError('Please check input dimension of the vector.')

        if vector.shape[0] == 1:
            return self.__transberse_embedding[vector[0]]
        else:
            return self.__transberse_embedding[np.argmax(vector)]

    def _embedding(self, embedding, word_dict):
        embedding_dict = {}
        transberse_embedding = {}
        if embedding == 'one_hot':
            transberse_embedding[0] = '<padding>'
            embedding_dict['<padding>'] = 0
            transberse_embedding[1] = '<bos>'
            embedding_dict['<padding>'] = 1
            transberse_embedding[2] = '<eos>'
            embedding_dict['<padding>'] = 2

            index = 3
            for word in word_dict.keys():
                if word_dict[word][1] < self.least_freq:
                    continue
                else:
                    embedding_dict[word] = index
                    transberse_embedding[index] = word
                    index += 1

            transberse_embedding[index + 1] = '<unknown>'

        return embedding_dict, transberse_embedding, index + 2

    def _word_dict(self, sentence):
        word_dict = {}
        for index in range(len(sentence)):
            line = sentence[index]
            if line == '+++$+++\n':
                continue
            line = self._clean_string(line)
            for word in range(len(line)):
                if line[word] not in word_dict.keys():
                    word_dict[line[word]] = [len(word_dict), 1] # index, frequency
                else:
                    word_dict[line[word]][1] += 1

        return word_dict

    def _clean_string(self, sentnece_list):
        new = [w for w in sentnece_list if not re.match(r'[A-Z]+', w, re.I)]
        new = [re.sub('[0-9]', '', w) for w in new]

        return new

    def _from_txt(self, path):
        f = open(path, 'r')
        text = f.readlines()
        f.close()

        return text
