import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class S2VT(nn.Module):
    def __init__(self, out_size, env, video_seq_lenth = 80, input_size = 4096, hidden_size = 256,
            attension = 'general', mode = 'self', probability = 0.2, predict = False):
        super(S2VT, self).__init__()

        self.out_size = out_size
        self.video_seq_lenth = video_seq_lenth
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.attention = attension
        self.mode = mode
        self.probability = probability
        self.env = env

        self.encoder = S2VTencoder(video_seq_lenth, input_size, hidden_size)
        self.decoder = S2VTdecoder(out_size, hidden_size, attension, mode, probability)

    def forward(self, video_seq, guided_token, mask = None):
        #mask is the batch mask to select the output of different time step in seq gen.
        self._check_probability()
        self.decoder.probability = self.probability

        e_h = torch.zeros(2, video_seq.size(1), self.hidden_size).to(self.env)
        e_c = torch.zeros(2, video_seq.size(1), self.hidden_size).to(self.env)

        video_embedding, (e_h, e_c) = self.encoder(video_seq, e_h, e_c)
        d_c = torch.zeros(1, video_seq.size(1), self.hidden_size * 2).to(self.env)
        out, hidden = self.decoder(guided_token, video_embedding, d_c)

        if mask is None:
            return out
        else:
            mask = self._flat_mask(mask)
            out = torch.masked_select(out, mask)
            out = out.view(-1, self.out_size)
            return out, hidden

    def _check_probability(self):
        if self.probability > 1:
            self.probability = 1

    def _flat_mask(self, mask):
        new_mask = None
        for mini_batch in range(mask.size(1)):
            if new_mask is None:
                new_mask = mask[:, mini_batch]
            else:
                new_mask = torch.cat((new_mask, mask[:, mini_batch]), dim = 0)

        return new_mask.byte()

class Attension(nn.Module):
    def __init__(self, method, hidden_size):
        super(Attension, self).__init__()

        self.method = method
        if self.method not in ['dot', 'general']:
            raise ValueError(self.method, "is not an appropriate attention method.")
        self.hidden_size = hidden_size
        
        if self.method == 'general':
            #input is bidirectional LSTM
            self.attn = nn.Linear(hidden_size * 2, hidden_size * 2)

    def _dot_score(self, hidden, encoder_out):
        encoder_out = encoder_out.unsqueeze(1)
        return torch.sum(torch.mul(encoder_out, hidden), dim = 3).transpose(0, 2)
    
    def _general_score(self, hidden, encoder_out):
        energy = self.attn(encoder_out)
        energy = energy.unsqueeze(1)
        return torch.sum(torch.mul(hidden, energy), dim = 3).transpose(0, 2)
    
    def forward(self, hidden, encoder_out):
        if self.method == 'dot':
            attn_energies = self._dot_score(hidden, encoder_out)
        elif self.method == 'general':
            attn_energies = self._general_score(hidden, encoder_out)
        return F.softmax(attn_energies, dim = 2)

class S2VTdecoder(nn.Module):
    def __init__(self, out_size, hidden_size, attension, mode, probability, dropout = 0.1):
        super(S2VTdecoder, self).__init__()

        self.out_size = out_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.probability = probability

        self.embedding = nn.Embedding(out_size, hidden_size * 2)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size * 2, hidden_size * 2, num_layers = 1)
        self.cat = nn.Linear(4 * hidden_size, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)

        self.attn = Attension(attension, hidden_size)

    def forward(self, input_tokens, video_state, c):
        input_tokens = torch.squeeze(input_tokens)
        input_tokens = self.embedding(input_tokens)
        input_tokens = self.embedding_dropout(input_tokens)

        hiddens, (d_h, d_c) = self.lstm(input_tokens, (video_state[video_state.size(0) - 1:, :, :], c))

        attn_weight = self.attn(hiddens, video_state)
        words_embedding = attn_weight.bmm(video_state.transpose(0, 1))  
        if self.mode == 'guide':
            words_embedding = torch.cat((hiddens, input_tokens), dim = 2)
        elif self.mode == 'self':            
            select = True if random.random() < self.probability else False
            if select:
                words_embedding = torch.cat((hiddens, torch.cat((hiddens[1:, :, :], d_h), dim = 0)), dim = 2)
            else:
                words_embedding = torch.cat((hiddens, input_tokens), dim = 2)
        outs = self._time_flatten(words_embedding) #seq_length * mini_batch * word_vector
        outs = torch.tanh(self.cat(outs))
        outs = self.linear(outs)

        return outs, [d_h, d_c]

    def _time_flatten(self, word_embedding):
        outs = None
        for mini_batch in range(word_embedding.size(1)):
            seqs = word_embedding[:, mini_batch, :]
            if outs is None:
                outs = seqs
            else:
                outs = torch.cat((outs, seqs), dim = 0)

        return outs

class S2VTencoder(nn.Module):
    def __init__(self, video_seq_lenth, input_size, hidden_size):
        super(S2VTencoder, self).__init__()

        self.video_seq_lenth = video_seq_lenth
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = 1, bidirectional = True)

    def forward(self, video_seq, e_h, e_c):
        video_embedding, (e_h, e_c) = self.lstm(video_seq, (e_h, e_c))

        return video_embedding, (e_h, e_c)


