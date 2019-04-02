import torch
import torch.nn as nn
import torch.nn.functional as F

class S2VT(nn.Module):
    def __init__(self, out_size, env, video_seq_lenth = 80, input_size = 4096, hidden_size = 256,
            mode = 'guide', probability= 0):
        super(S2VT, self).__init__()

        self.out_size = out_size
        self.video_seq_lenth = video_seq_lenth
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.probability = probability
        self.env = env

        self.encoder = S2VTencoder(video_seq_lenth, input_size, hidden_size)
        self.decoder = S2VTdecoder(out_size, hidden_size, mode, probability)

    def forward(self, video_seq, guided_token, mask_index = None):
        #mask is the batch mask to select the output of different time step in seq gen.
        e_h = torch.zeros(1, video_seq.size(1), self.hidden_size).to(self.env)
        e_c = torch.zeros(1, video_seq.size(1), self.hidden_size).to(self.env)

        video_embedding, (e_h, e_c) = self.encoder(video_seq, e_h, e_c)
        d_c = torch.zeros(1, video_seq.size(1), self.hidden_size).to(self.env)
        out = self.decoder(guided_token, video_embedding, d_c)

        if mask_index is None:
            return out
        else:
            mask_out = torch.empty(video_seq.size(1), self.out_size).to(self.env)
            for i in range(len(mask_index)):
                mask_out[i] = out[mask_index[i], i, :]

            return mask_out

class S2VTdecoder(nn.Module):
    def __init__(self, out_size, hidden_size, mode, probability):
        super(S2VTdecoder, self).__init__()

        self.out_size = out_size
        self.hidden_size = hidden_size
        self.mode = mode
        self.probability = probability

        self.embedding = nn.Embedding(out_size, hidden_size, padding_idx = 1)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers = 1)
        self.linear = nn.Linear(2 * hidden_size, out_size)

    def forward(self, input_tokens, video_state, c):
        input_tokens = self.embedding(input_tokens)
        hiddens, (d_h, d_c) = self.lstm(input_tokens, (video_state, c))
  
        if self.mode == 'guide':
            words_embedding = torch.cat((hiddens, input_tokens), dim = 2)
        elif self.mode == 'self':
            
            words_embedding = torch.cat((hiddens, torch.cat((hiddens[1:, :, :], d_h), dim = 0)), dim = 2)

        outs = self._time_flatten(words_embedding) #seq_length * mini_batch * word_vector

        return outs

    def _time_flatten(self, word_embedding):
        outs = None
        for mini_batch in range(word_embedding.size(1)):
            seqs = word_embedding[:, mini_batch, :]
            out = self.linear(seqs)
            if outs is None:
                outs = out.unsqueeze(dim = 1)
            else:
                outs = torch.cat((outs, out.unsqueeze(dim = 1)), dim = 1)

        return outs

class S2VTencoder(nn.Module):
    def __init__(self, video_seq_lenth, input_size, hidden_size):
        super(S2VTencoder, self).__init__()

        self.video_seq_lenth = video_seq_lenth
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers = 1)

    def forward(self, video_seq, e_h, e_c):
        video_embedding, (e_h, e_c) = self.lstm(video_seq, (e_h, e_c))
        video_embedding = video_embedding[self.video_seq_lenth - 1: , :, :] #seq_length, batch, out -> batch, last out

        return video_embedding, (e_h, e_c)


