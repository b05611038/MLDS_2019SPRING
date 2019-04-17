import random
import torch
import torch.nn as nn
import torch.nn.functional as F

class Seq2seq(nn.Module):
    def __init__(self, out_size, env, max_seq_length, hidden_size, bidirectional, attention, mode, probability):
        super(Seq2seq, self).__init__()

        self.out_size = out_size
        self.env = env
        self.max_seq_length = max_seq_length
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.direction = 2 if self.bidirectional else 1
        self.attention = attention
        self.mode = mode
        self.probability = probability
        self.bos = 2 #padding: 0, <eos>: 1, <bos>: 2

        self.encoder = seq2seqEncoder(out_size, hidden_size, bidirectional)
        self.decoder = seq2seqDecoder(out_size, hidden_size, self.direction,
                attention, mode, self.probability, self.bos)
    def forward(self, input_tokens, guided_token, mask = None):
        #mask is the batch mask to select the output of different time step in seq gen.
        self._check_probability()
        self.decoder.probability = self.probability

        e_h = torch.zeros(self.direction, input_tokens.size(1), self.hidden_size).to(self.env)
        e_c = torch.zeros(self.direction, input_tokens.size(1), self.hidden_size).to(self.env)

        sentence_embedding, (e_h, e_c) = self.encoder(input_tokens, e_h, e_c)
        d_c = torch.zeros(1, input_tokens.size(1), self.hidden_size * self.direction).to(self.env)
        out, hidden = self.decoder(guided_token, sentence_embedding, d_c)

        if mask is None:
            return out, hidden
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

class Attention(nn.Module):
    def __init__(self, method, hidden_size, direction):
        super(Attention, self).__init__()

        self.method = method
        if self.method not in ['dot', 'general', 'concat']:
            raise ValueError(self.method, "is not an appropriate attention method.")

        self.hidden_size = hidden_size
        self.direction = direction

        if self.method == 'general':
            self.attn = nn.Linear(hidden_size * direction, hidden_size * direction)
        elif self.method == 'concat':
            self.attn = nn.Linear(2 * hidden_size * direction, hidden_size * direction)
            self.v = torch.nn.Parameter(torch.FloatTensor(hidden_size * direction))

    def _dot_score(self, hidden, encoder_out):
        encoder_out = encoder_out.unsqueeze(1)
        return torch.sum(torch.mul(encoder_out, hidden), dim = 3).transpose(0, 2)

    def _general_score(self, hidden, encoder_out):
        energy = self.attn(encoder_out)
        energy = energy.unsqueeze(1)
        return torch.sum(torch.mul(hidden, energy), dim = 3).transpose(0, 2)

    def _concat_score(self, hidden, encoder_out):
        decode_seq_length = hidden.size(0)
        encode_seq_length = encoder_out.size(0)
        encoder_out = encoder_out.expand(decode_seq_length, -1, -1, -1)
        decoder_out = hidden.expand(encode_seq_length, -1, -1, -1).transpose(0, 1)
        energy = self.attn(torch.cat((encoder_out, decoder_out), dim = 3)).transpose(0, 1).tanh()
        return torch.sum(torch.mul(self.v, energy), dim = 3).transpose(0, 2)

    def forward(self, hidden, encoder_out):
        # hidden (docoder input): seq_length * batch * hidden_size
        # encoder_out: input_seq_length * batch * hidden_size
        # return atte weight would be batch * seq_length * input_seq_length
        if self.method == 'dot':
            attn_energies = self._dot_score(hidden, encoder_out)
        elif self.method == 'general':
            attn_energies = self._general_score(hidden, encoder_out)
        elif self.method == 'concat':
            attn_energies = self._concat_score(hidden, encoder_out)

        return F.softmax(attn_energies, dim = 2)

class seq2seqDecoder(nn.Module):
    def __init__(self, out_size, hidden_size, direction, attention, mode, probability, bos, dropout = 0.1):
        super(seq2seqDecoder, self).__init__()

        self.out_size = out_size
        self.hidden_size = hidden_size
        self.direction = direction
        self.attention = attention
        self.mode = mode
        self.probability = probability
        self.bos = bos
        self.dropout = dropout

        self.decoder_embedding = nn.Embedding(out_size, hidden_size * direction)
        self.decoder_embedding_dropout = nn.Dropout(dropout)

        self.decoder_lstm = nn.LSTM(hidden_size * direction, hidden_size * direction, num_layers = 1)
        self.cat = nn.Linear(2 * hidden_size * direction, hidden_size)
        self.linear = nn.Linear(hidden_size, out_size)

        self.attn = Attention(attention, hidden_size, direction)

    def forward(self, input_tokens, encode_hidden, c):
        input_tokens = torch.squeeze(input_tokens)
        input_tokens = self.decoder_embedding(input_tokens)
        input_tokens = self.decoder_embedding_dropout(input_tokens)

        hiddens, (d_h, d_c) = self.decoder_lstm(input_tokens, (encode_hidden[encode_hidden.size(0) - 1:, :, :], c))

        attn_weight = self.attn(hiddens, encode_hidden)
        words_embedding = attn_weight.bmm(encode_hidden.transpose(0, 1))
        if self.mode == 'guide':
            words_embedding = torch.cat((hiddens, input_tokens), dim = 2)
        elif self.mode == 'self':
            select = True if random.random() < self.probability else False
            if select:
                begin = torch.tensor(self.bos).expand(encode_hidden.size(1), -1).transpose(0, 1)
                begin = self.decoder_embedding(begin)
                begin = self.decoder_embedding_dropout(begin)
                out_selves = torch.cat((begin, hiddens[1:, :, :]), dim = 0)
                words_embedding = torch.cat((hiddens, out_selves), dim = 2)
            else:
                words_embedding = torch.cat((hiddens, input_tokens), dim = 2)
        else:
            raise ValueError(self.mode, 'is not in mode setting (guide or self)')

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

class seq2seqEncoder(nn.Module):
    def __init__(self, max_index, hidden_size, bidirectional, dropout = 0.1):
        super(seq2seqEncoder, self).__init__()

        self.max_index = max_index
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional
        self.dropout = dropout
        self.direction = 2 if self.bidirectional else 1

        self.encoder_embedding = nn.Embedding(max_index, hidden_size)
        self.encoder_embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers = 1, bidirectional = bidirectional, dropout = 0.1)

    def forward(self, word_seq, e_h, e_c):
        word_seq = torch.squeeze(word_seq)
        word_seq = self.encoder_embedding(word_seq)
        word_seq = self.encoder_embedding_dropout(word_seq)
        word_embedding, (e_h, e_c) = self.lstm(word_seq, (e_h, e_c))

        return word_embedding, (e_h, e_c)


