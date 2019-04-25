import re
import csv
import numpy as np
import torch

from lib.utils import *

class Word2vec():
    def __init__(self, embedding, path_list, least_freq, seq_length_min, seq_length_max):
        self.embedding = embedding
        if embedding not in ['one_hot']:
            raise ValueError(self.embedding, 'not in the word embedding method.')

        self.path_list = path_list
        self.least_freq = least_freq * len(path_list)
        self.seq_length_min = seq_length_min
        self.seq_length_max = seq_length_max

        text_list = self._from_txt_list(path_list)
        word_dict = self._word_dict(text_list)
        self.__embedding_dict, self.__transberse_embedding, self.seq_max = self._embedding(embedding, word_dict)

    #only support for MLDS hw2-2 training txt file
    def combine_qa(self, question, answer, save = None):
        pair = []
        questions = self._from_txt(question)
        answers = self._from_txt(answer)
        if len(questions) != len(answers):
            raise IndexError('Question file and Answer file index mismatch.')

        for index in range(len(questions)):
            last = questions[index]
            then = answers[index]
            last = self._clean_string(last)
            then = self._clean_string(then)

            arr_last = np.empty(len(last), )
            arr_then = np.empty(len(then) + 2, ) #<bos>, sentence, <eos>

            for word in range(len(last)):
                arr_last[word] = self.w2v(last[word])

            arr_then[0] = self.w2v('<bos>')
            for word in range(1, len(then) + 1):
                arr_then[word] = self.w2v(then[word - 1])

            arr_then[len(then) + 1] = self.w2v('<eos>')

            pair.append([arr_last, arr_then])

        if save is not None:
            save_object(save, pair)
            return pair
        else:
            return pair

    #only support for MLDS hw2-2 training txt file
    def sentence_pair(self, txt_path, save = None):
        pair = []
        text = self._from_txt(txt_path)
        for index in range(len(text) - 1):
            last = text[index]
            then = text[index + 1]
            if last == '+++$+++\n' or then == '+++$+++\n':
                continue

            last = self._clean_string(last)
            then = self._clean_string(then)
            if len(last) <= self.seq_length_max and len(last) >= self.seq_length_min and len(then) <= self.seq_length_max and len(then) >= self.seq_length_min:
                arr_last = np.empty(len(last), )
                arr_then = np.empty(len(then) + 2, ) #<bos>, sentence, <eos>

                for word in range(len(last)):
                    arr_last[word] = self.w2v(last[word])

                arr_then[0] = self.w2v('<bos>')
                for word in range(1, len(then) + 1):
                    arr_then[word] = self.w2v(then[word - 1])

                arr_then[len(then) + 1] = self.w2v('<eos>')

                pair.append([arr_last, arr_then])

            else:
                pass

        if save is not None:
            save_object(save, pair)
            return pair
        else:
            return pair

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
            embedding_dict['<bos>'] = 1
            transberse_embedding[2] = '<eos>'
            embedding_dict['<eos>'] = 2

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

    def _word_dict(self, sentence_file):
        word_dict = {}
        for files in range(len(sentence_file)):
            sentence = sentence_file[files]
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

    def _clean_string(self, sentence):
        sentence_list = sentence.replace('\n', '').split(' ')
        new = [w for w in sentence_list if not re.match(r'[A-Z]+', w, re.I)]
        new = [re.sub('[0-9]', '', w) for w in new]
        new = [w for w in new if w]

        return new

    def _from_txt_list(self, path_list):
        text_list = []
        for paths in path_list:
            f = open(paths, 'r')
            text_temp = f.readlines()
            f.close()
            text_list.append(text_temp)

        return text_list

    def _from_txt(self, path):
        f = open(path, 'r')
        text = f.readlines()
        f.close()

        return text


