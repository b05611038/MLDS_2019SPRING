import copy
import torch
import torch.cuda as cuda
import torch.nn as nn
import torch.nn.functional as F

from lib.utils import *
from lib.model import S2VT

class BeamSearch(nn.Module):
    def __init__(self, whole_model, word2vec, env, k = 2, max_length = 20):
        super(BeamSearch, self).__init__()

        self.whole_model = whole_model
        self.env = env
        self.word2vec = load_object(word2vec)
        self.bos = torch.tensor(self.word2vec.w2v('<bos>'))
        self.eos = torch.tensor(self.word2vec.w2v('<eos>'))
        self.k = k
        self.max_length = max_length
        self.seq = None

        self.encoder, decoder = self._load_trained_model(whole_model)
        self.encoder.to(self.env)
        self.lstm = decoder.lstm
        self.embedding = decoder.embedding
        self.embedding_dropout = decoder.embedding_dropout
        self.cat = decoder.cat
        self.linear = decoder.linear
        self.attn = decoder.attn

    def forward(self, video_seq):
        e_h = torch.zeros(2, video_seq.size(1), self.encoder.hidden_size).to(self.env)
        e_c = torch.zeros(2, video_seq.size(1), self.encoder.hidden_size).to(self.env)

        video_embedding, (e_h, e_c) = self.encoder(video_seq, e_h, e_c)
        d_c = torch.zeros(1, video_seq.size(1), self.encoder.hidden_size * 2).to(self.env)

        self._beam_search(self.bos, video_embedding, d_c)
        seq = copy.deepcopy(self.seq)
        seq = self._biggest(seq)
        self.seq = None
        return seq

    def _decode(self, video, input_token, h, c):
        input_token = input_token.to(self.env)
        video = video.to(self.env)
        h = h.to(self.env)
        c = c.to(self.env)

        input_token = self.embedding(input_token)
        input_token = self.embedding_dropout(input_token)

        hiddens, (d_h, d_c) = self.lstm(input_token, (h, c))
        attn_weight = self.attn(hiddens, video)
        words_embedding = attn_weight.bmm(video.transpose(0, 1))
        words_embedding = torch.cat((hiddens, input_token), dim = 2)
        outs = words_embedding.view(-1, self.encoder.hidden_size * 4)
        outs = torch.tanh(self.cat(outs))
        outs = self.linear(outs)
        outs = F.softmax(outs)

        return outs, [d_h, d_c]

    def _biggest(self, seq):
        pro = 0
        out = None
        for i in range(len(seq)):
            if (seq[i][1] > pro).item() == 1:
                pro = seq[i][1].detach()
                out = seq[i][0]

        return out

    def _beam_search(self, bos, video_embedding, c):
        self.seq = []
        bos = [torch.tensor([bos]), torch.tensor(1).float()]
        self._k_search(bos, video_embedding, video_embedding[video_embedding.size(0) - 1:, :, :], c)
        return None

    def _k_search(self, past_seq, video_embedding, h, c):
        if past_seq[0].size(0) >= self.max_length:
            return None
        last_word = past_seq[0][past_seq[0].size(0) - 1:]
        last_word = last_word.view(1, 1)
        new_out, hidden = self._decode(video_embedding, last_word, h, c)
        pro, index = torch.topk(new_out, self.k)
        pro = pro.detach().squeeze().to('cpu')
        index = index.detach().squeeze().to('cpu')
        for i in range(self.k):
            if (index[i] == self.eos).item() == 1:
                new_pro = past_seq[1] * pro[i]
                seq = [torch.cat((past_seq[0], torch.tensor([index[i]])), dim = 0), new_pro]
                self.seq.append(seq)
                return None
            else:
                new_pro = past_seq[1] * pro[i]
                seq = [torch.cat((past_seq[0], torch.tensor([index[i]])), dim = 0), new_pro]
                self._k_search(seq, video_embedding, hidden[0], hidden[1])

    def _load_trained_model(self, path):
        s2vt = torch.load(path, map_location = 'cpu')
        s2vt = s2vt.eval()
        s2vt.to(self.env)
        return s2vt.encoder, s2vt.decoder


