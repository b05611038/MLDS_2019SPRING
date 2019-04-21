import copy
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import *
from lib.model import Seq2seq

class BeamSearch(nn.Module):
    def __init__(self, whole_model, word2vec, env, k = 3, max_length = 20):
        super(BeamSearch, self).__init__()

        self.whole_model = whole_model
        self.env = env
        self.word2vec = word2vec
        self.bos = torch.tensor(self.word2vec.w2v('<bos>'))
        self.eos = torch.tensor(self.word2vec.w2v('<eos>'))
        self.k = k
        self.max_length = max_length
        self.seq = None

        self.encoder, decoder, self.bidirectional = self._load_trained_model(whole_model)
        self.direction = 2 if self.bidirectional else 1
        self.encoder.to(self.env)
        self.decoder_lstm = decoder.decoder_lstm
        self.decoder_embedding = decoder.decoder_embedding
        self.decoder_embedding_dropout = decoder.decoder_embedding_dropout
        self.cat = decoder.cat
        self.linear = decoder.linear
        self.attn = decoder.attn

    def forward(self, sentence):
        e_h = torch.zeros(self.direction, sentence.size(1), self.encoder.hidden_size).to(self.env)
        e_c = torch.zeros(self.direction, sentence.size(1), self.encoder.hidden_size).to(self.env)

        print(sentence.size(), e_h.size(), e_c.size())
        sentence_embedding, (e_h, e_c) = self.encoder(sentence, e_h, e_c)
        d_c = torch.zeros(1, sentence.size(1), self.encoder.hidden_size * self.direction).to(self.env)

        self._beam_search(self.bos, sentence_embedding, d_c)
        seq = copy.deepcopy(self.seq)
        seq = self._biggest(seq)
        self.seq = None
        return seq

    def _beam_search(self, bos, sentence_embedding, c):
        self.seq = []
        bos = [[torch.tensor([bos]), torch.tensor(1).float(), sentence_embedding[sentence_embedding.size(0) - 1:, :, :], c]]
        if self.k == 1:
            self._greedy_search(bos, sentence_embedding)
        else:
            self._k_search(bos, sentence_embedding)
        return None

    def _greedy_search(self, past_seq, sentence_embedding):
        last_word = past_seq[0][0][past_seq[0][0].size(0) - 1:]
        last_word = last_word.view(1, 1)
        new_out, hidden = self._decode(sentence_embedding, last_word, past_seq[0][2], past_seq[0][3])
        pro, index = torch.max(new_out, 1)
        pro = pro.detach().squeeze().to('cpu')
        index = index.detach().squeeze().to('cpu')
        new_pro = past_seq[0][1] * pro
        if (index == self.eos).item() == 1:
            seq = [torch.cat((past_seq[0][0], torch.tensor([index])), dim = 0), new_pro]
            self.seq.append(seq)
            return None

        elif past_seq[0][0].size(0) == self.max_length:
            seq = [torch.cat((past_seq[0][0], torch.tensor([index])), dim = 0), new_pro]
            self.seq.append(seq)
            return None

        else:
            seq = [[torch.cat((past_seq[0][0], torch.tensor([index])), dim = 0), new_pro, hidden[0], hidden[1]]]
            self._greedy_search(seq, sentence_embedding)

    def _k_search(self, past_seq, sentence_embedding):
        length = len(past_seq)
        seq_temp = []
        if self.k < length:
            length = self.k

        for beam in range(length):
            last_word = past_seq[beam][0][past_seq[beam][0].size(0) - 1:]
            last_word = last_word.view(1, 1)
            new_out, hidden = self._decode(sentence_embedding, last_word, past_seq[beam][2],  past_seq[beam][3])
            pro, index = torch.topk(new_out, self.k)
            index = index.detach().squeeze().to('cpu')
            pro = pro.detach().squeeze().to('cpu')
            for i in range(self.k):
                new_pro = past_seq[beam][1] * pro[i]
                if (index[i] == self.eos).item() == 1:
                    seq = [torch.cat((past_seq[beam][0], torch.tensor([index[i]])), dim = 0), new_pro]
                    self.seq.append(seq)
                    return None
                else:
                    seq = [torch.cat((past_seq[beam][0], torch.tensor([index[i]])), dim = 0), new_pro, hidden[0], hidden[1]]
                    seq_temp.append(seq)

        seq_temp.sort(key = lambda pro: pro[1])
        seq_temp[:self.k]
        self._k_search(seq_temp, sentence_embedding)

    def _decode(self, sentence, input_token, h, c):
        input_token = input_token.to(self.env)
        sentence = sentence.to(self.env)
        h = h.to(self.env)
        c = c.to(self.env)

        input_token = self.decoder_embedding(input_token)
        input_token = self.decoder_embedding_dropout(input_token)

        hiddens, (d_h, d_c) = self.decoder_lstm(input_token, (h, c))
        attn_weight = self.attn(hiddens, sentence)
        words_embedding = attn_weight.bmm(sentence.transpose(0, 1))
        words_embedding = torch.cat((hiddens, input_token), dim = 2)
        outs = words_embedding.view(-1, self.encoder.hidden_size * self.direction * 2)
        outs = torch.tanh(self.cat(outs))
        outs = self.linear(outs)
        outs = F.softmax(outs)

        return outs, [d_h, d_c]

    def _load_trained_model(self, path):
        seq2seq = torch.load(path, map_location = 'cpu')
        seq2seq = seq2seq.eval()
        seq2seq = seq2seq.to(self.env)
        bidirectional = seq2seq.bidirectional
        return seq2seq.encoder, seq2seq.decoder, bidirectional

    def _biggest(self, seq):
        pro = 0
        out = None
        for i in range(len(seq)):
            if (seq[i][1] > pro).item() == 1:
                pro = seq[i][1].detach()
                out = seq[i][0]

        return out


