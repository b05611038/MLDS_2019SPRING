import os
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data

from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

use_cuda = torch.cuda.is_available()

def read_data(file):
    with open(file,'r') as f:
        for line in (f.readlines()):
            line = line.strip().replace(' ', '')
            yield(line)

class TestDataset(data.Dataset):
    def __init__(self, input_file, output_file):
        self.max_len = 20
        self.vocab = 'vocab.txt'
        self.input_file = input_file
        self.output_file = output_file
        self.raw_input_data = list(read_data(self.input_file))
        self.raw_output_data = list(read_data(self.output_file))
        self.input_length = []
        self.output_length = []
        self.input_data = []
        self.output_data = []
        self._load_vocab()
        self._prepare_data(self.raw_input_data,self.input_data,self.input_length)
        self._prepare_data(self.raw_output_data,self.output_data,self.output_length)

    def _load_vocab(self):
        self.v2id, self.id2v = {}, {}
        with open(self.vocab, 'r') as f: 
            for line in f:
                i, w = line.strip().split()
                self.v2id[str(w)] = int(i)
                self.id2v[int(i)] = str(w)

    def _prepare_data(self,raw_data,data,length):
        for line in (raw_data):
            if line == '':
                continue
            line = [int(self.v2id.get(w, 3)) for w in line]
            if len(line)> self.max_len:
                line = line[:self.max_len]
                length.append([len(line)])
            else:
                length.append([len(line)])
                line = np.concatenate((line,np.zeros((self.max_len-len(line)),dtype = np.int32)),axis =0)
            data.append(np.asarray(line))

    def __getitem__(self,idx):
        return (torch.LongTensor(self.input_data[idx]),torch.LongTensor(self.output_data[idx]),
                torch.LongTensor(self.input_length[idx]),torch.LongTensor(self.output_length[idx]))

    def __len__(self):
        return (len(self.raw_input_data ))

class Encoder(nn.Module):
    def __init__(self):
        super(Encoder,self).__init__()
        self.embedding = nn.Embedding(3000, 300)
        self.gru1 = nn.GRU(300,256,batch_first = True)
        self.gru2 = nn.GRU(300,256,batch_first = True)
        self.fc1 = nn.Linear(512,1000)
        self.fc2 = nn.Linear(1000,1000)
        self.fc3 = nn.Linear(1000,1)
        self.relu = nn.ReLU()

    def forward(self,x1,x2,x1_len,x2_len):

        x1 = self.embedding(x1)
        x2 = self.embedding(x2) 

        x1_len_sorted,idx_1 = torch.sort(x1_len,0, descending=True)
        x2_len_sorted,idx_2 = torch.sort(x2_len,0, descending=True)

        x1_sorted = x1[idx_1.view(-1)]
        x2_sorted = x2[idx_2.view(-1)]
        x1_pack = pack_padded_sequence(x1_sorted, x1_len_sorted.view(-1).tolist(), batch_first=True)
        x2_pack = pack_padded_sequence(x2_sorted, x2_len_sorted.view(-1).tolist(), batch_first=True)
        
        output1, hidden1 = self.gru2(x1_pack,None)
        output2, hidden2 = self.gru2(x2_pack,None)

        (output1) = pad_packed_sequence(output1,batch_first=True)
        (output2) = pad_packed_sequence(output2,batch_first=True)

        _,idx_1 = idx_1.view(-1).sort(0)
        _,idx_2 = idx_2.view(-1).sort(0)

        output1 = hidden1[0][idx_1]
        output2 = hidden2[0][idx_2]

        output = torch.cat((output1,output2),1)
        output = self.relu(self.fc1(output))
        output = self.relu(self.fc2(output))

        return self.fc3(output)

class correlation_score():
    def __init__(self, input_file, output_file):
        self.batch_size = 32
        self.model_path = 'model/correlation.mdl'
        self.test_data = TestDataset(input_file, output_file)
        self.test_loader = data.DataLoader(dataset=self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=0)
        self.model = Encoder().cuda() if use_cuda else Encoder()
        self.model.load_state_dict(torch.load(self.model_path)) if use_cuda else self.model.load_state_dict(torch.load(self.model_path,map_location='cpu'))

    def predict(self):
        self.model.eval()
        score_sum = 0
        for (x1,x2,x1_len,x2_len) in self.test_loader:
            x1 = Variable(x1,volatile=True).cuda() if use_cuda else Variable(x1,volatile=True)
            x2 = Variable(x2,volatile=True).cuda() if use_cuda else Variable(x2,volatile=True)
            x1_len =(x1_len).cuda() if use_cuda else (x1_len)
            x2_len =(x2_len).cuda() if use_cuda else (x2_len)
            output = self.model(x1,x2,x1_len,x2_len)
            score = F.sigmoid(output)
            if not hasattr(score, 'sum'):
                score = score.cpu()
            score_sum += score.sum().data[0]
        score_sum /= len(self.test_data)
        return score_sum
        print ('correlation score : {0:.5f} (baseline: > 0.45)'.format(score_sum))

