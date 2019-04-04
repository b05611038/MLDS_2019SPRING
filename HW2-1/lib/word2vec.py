import json
import torch
import numpy as np

from lib.utils import *

class Word2vec():
    def __init__(self, path, embedding, least_freq):
        self.path = path
        self.embedding = embedding
        self.least_freq = least_freq

        self.data = self._from_json(path)
        self.word_dict, self.sentences = self._word_dict(self.data)
        #the last item in the embedding dictionary is all unknown word
        self.__embedding_dict, self.__transberse_embedding, self.seq_max = self._embedding(embedding)

    #only support for MLDS hw2-1 .json label file
    def label_dict(self, json_path, save = None):
        data = self._from_json(json_path)
        data_dict = {}
        for video in range(len(data)):
            data_dict[data[video]['id']] = []
            for cap in range(len(data[video]['caption'])):
                strings = data[video]['caption'][cap].replace('.', '').lower().split(' ')
                seq = np.empty(len(strings) + 2, )
                seq[0] = self.w2v('<bos>')
                for word in range(1, len(strings) + 1):
                    seq[word] = self.w2v(strings[word - 1])

                seq[len(strings) + 1] = self.w2v('<eos>')

                data_dict[data[video]['id']].append(seq)

        if save is not None:
            save_object(save, data_dict)
            return data_dict
        else:
            return data_dict

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

    def _embedding(self, embedding):
        embedding_dict = {}
        transberse_embedding = {}
        if embedding == 'one_hot':
            transberse_embedding[0] = '<padding>'
            embedding_dict['<padding>'] = 0
            index = 1
            for word in self.word_dict.keys():
                if self.word_dict[word][1] < self.least_freq:
                    continue
                else:
                    embedding_dict[word] = index
                    transberse_embedding[index] = word
                    index += 1

            transberse_embedding[index] = '<bos>'
            embedding_dict['<bos>'] = index
            transberse_embedding[index + 1] = '<eos>'
            embedding_dict['<eos>'] = index + 1
            transberse_embedding[index + 2] = 'unknown'

            return embedding_dict, transberse_embedding, index + 3
        else:
            raise RuntimeError('No choosing embedding, please check the embedding setting.')

    def _word_dict(self, data):
        sentences = []
        word_dict = {}
        for video in range(len(data)):
            for cap in range(len(data[video]['caption'])):
                sentences.append(data[video]['caption'][cap].lower())
                strings = data[video]['caption'][cap].replace('.', '').lower().split(' ')
                for word in range(len(strings)):
                    if strings[word] not in word_dict.keys():
                        word_dict[strings[word]] = [len(word_dict), 1] #index, freqency
                    else:
                        word_dict[strings[word]][1] += 1

        return word_dict, sentences

    #only used in MLDS hw2-1 for reading specific format json label
    def _from_json(self, path):
        f = open(path, 'r')
        obj = f.read()
        f.close()

        obj = json.loads(obj)

        return obj


