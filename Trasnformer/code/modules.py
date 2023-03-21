import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab_size):
        super(Embeddings, self).__init__()
        self.emb = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model

    def forward(self, x):
        '''
        x is expected as (batch, len)
        output: (batch, len, hidden)
        '''
        return self.emb(x)

class PositionalEncoding(nn.Module):
    "Implement the PE function."

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model)
        # position.shape:(max_len, 1)
        position = torch.arange(0., max_len).unsqueeze(1)
        # torch.arange(0., d_model, 2) represent 2i in formula
        div_term = torch.exp(torch.arange(0., d_model, 2) *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        # pe.shape (1, max_len, d_model)
        pe = pe.unsqueeze(0)
        # register_buffer: store pe in state_dict(), but will not be trained like Parameter.
        self.register_buffer('pe', pe)

    def forward(self, x):
        '''
        x is expected as (batch, len, hidden)
        output: (batch, len, hidden)
        '''
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)


def subsequent_mask(batch, size):
    "Mask out subsequent positions."
    '''
    input:
        size : len
    output:
        give a lower triangle with True(include main diagonal), upper triangle is False
        which shape is (batch, size, size)
    '''
    attn_shape = (batch, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        "Take in model size and number of heads."
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        # We assume d_v always equals d_k
        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def attention(self, query, key, value, mask=None, dropout=None):
        "Compute 'Scaled Dot Product Attention'"
        '''
        input: 
            query, key, value are expected as (batch, head, len, d_k)
        output:
            result(batch, head, len, d_k), p_attn(batch, head, len, len)
        '''
        d_k = query.size(-1)
        # scores.shape (batch, head, len, len)
        # for i-th word in sequence, scores[i,j] is the j-th word's weight
        # in other words, each line in scores represent the weights of each word for the line_idx-th word
        scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
        # get a upper triangle(not include the main-diagonal) with value -1e9
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        # after softmax, the -1e9 will get 0 in the upper triangle
        # which means the i-th word can only "see" the previous j-th word(j<i)
        p_attn = F.softmax(scores, dim=-1)
        if dropout is not None:
            p_attn = dropout(p_attn)
        # p_attn.shape(batch, head, len, len)
        return torch.matmul(p_attn, value), p_attn

    def forward(self, query, key, value, mask=None):
        '''
        input: 
            query, key, value are expected as (batch, len, hidden)
        output:
            result(batch, len, hidden)
        '''
        # give a mask.shape(batch, 1, len, len)
        if mask is not None:
            # Same mask applied to all h heads.
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]
        # q,k,v shape(batch, head, len, d_k)
        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = self.attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        # x.shape (batch, head, len, d_k)
        # 3) "Concat" using a view and apply a final linear.
        # ps: view and transpose in torch will not change the way physical memory stroe the tensor, but only change way of indexing
        # tensor.contiguous() make the tensor's physical memory as the as the way of indexing, usually used in two view() or transpose
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        # x.shape (batch, len, hidden)
        return self.linears[-1](x)


class PositionwiseFeedForward(nn.Module):
    "Implements FFN equation."
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        '''
        input.shape == output.shape (batch, len, hidden)
        '''
        return self.w_2(self.dropout(F.relu(self.w_1(x))))


class LayerNorm(nn.Module):

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        '''
        x.shape(batch, len, hidden)
        '''
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    # pre-LN
    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))


class EncoderBlock(nn.Module):
    "Encoder is made up of self-attn and feed forward (defined below)"
    def __init__(self, d_model, d_ff, n_head, dropout):
        super(EncoderBlock, self).__init__()
        self.self_attn = MultiHeadedAttention(n_head, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(2)])
        self.d_model = d_model

    def forward(self, x, mask=None):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)

class Encoder(nn.Module):

    def __init__(self, d_model, d_ff, n_head, dropout, n_block):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([EncoderBlock(d_model, d_ff, n_head, dropout) for _ in range(n_block)])
        self.norm = LayerNorm(d_model)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)

class DecoderBlock(nn.Module):

    def __init__(self, d_model, d_ff, n_head, dropout):
        super(DecoderBlock, self).__init__()
        self.self_attn = MultiHeadedAttention(n_head, d_model)
        self.src_attn = MultiHeadedAttention(n_head, d_model)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.sublayer = nn.ModuleList([SublayerConnection(d_model, dropout) for _ in range(3)])

    def forward(self, x, m, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
        return self.sublayer[2](x, self.feed_forward)

class Decoder(nn.Module):

    def __init__(self, d_model, d_ff, n_head, dropout, n_block):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([DecoderBlock(d_model, d_ff, n_head, dropout) for _ in range(n_block)])
        self.norm = LayerNorm(d_model)

    def forward(self, x, m, mask):
        for layer in self.layers:
            x = layer(x, m, mask)
        return self.norm(x)

class Generator(nn.Module):
    "Define standard linear + softmax generation step."
    def __init__(self, d_model, vocab):
        super(Generator, self).__init__()
        self.proj = nn.Linear(d_model, vocab)

    def forward(self, x):
        return F.log_softmax(self.proj(x), dim=-1)