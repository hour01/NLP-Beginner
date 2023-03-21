import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence

class LSTM(nn.Module):
    def __init__(self, opt):
        super(LSTM, self).__init__()
        self.opt = opt
        self.i2h = nn.Linear(opt.hidden_size, 4 * opt.hidden_size)
        self.h2h = nn.Linear(opt.hidden_size, 4*opt.hidden_size)
        if opt.dropoutrec > 0:
            self.dropout = nn.Dropout(opt.droputrec)

    def forward(self, x, prev_h, prev_c):
        gates = self.i2h(x) + self.h2h(prev_h)
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)
        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)
        if self.opt.dropoutrec > 0:
            cellgate = self.dropout(cellgate)
        cy = (forgetgate * prev_c) + (ingate * cellgate)
        hy = outgate * torch.tanh(cy)  # n_b x hidden_dim
        return hy, cy

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()
        self.opt = opt
        self.hidden_size = opt.hidden_size
        self.input_size = opt.word_manager.vocab_size
        self.embedding = nn.Embedding(self.input_size, self.hidden_size, padding_idx=opt.word_manager.get_symbol_idx('<PAD>'))
        self.lstm = LSTM(self.opt)
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)

    def forward(self, input_src, prev_h, prev_c):
        src_emb = self.embedding(input_src) # batch_size x src_length x emb_size
        if self.opt.dropout > 0:
            src_emb = self.dropout(src_emb)
        prev_hy, prev_cy = self.lstm(src_emb, prev_h, prev_c)
        return prev_hy, prev_cy

class Decoder(nn.Module):
    def __init__(self, opt):
        super(Decoder, self).__init__()
        self.opt = opt
        self.hidden_size = opt.hidden_size
        self.output_size = opt.form_manager.vocab_size
        self.embedding = nn.Embedding(self.output_size, self.hidden_size, padding_idx=opt.word_manager.get_symbol_idx('<PAD>'))
        self.lstm = LSTM(self.opt)
        self.linear = nn.Linear(self.hidden_size, self.output_size)
        if opt.dropout > 0:
            self.dropout = nn.Dropout(opt.dropout)

        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, input, prev_h, prev_c):
        output = self.embedding(input)
        if self.opt.dropout > 0:
            output = self.dropout(output)
        next_h, next_c = self.lstm(output, prev_h, prev_c)
        if self.opt.dropout > 0:
            next_h = self.dropout(next_h)
        h2y = self.linear(next_h)
        pred = self.softmax(h2y)
        return pred, next_h, next_c