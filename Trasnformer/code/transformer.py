from linecache import checkcache
import random
import gensim
import torch
import time
from torch.nn.utils.rnn import pad_sequence
from modules import *
import torch.optim as optim
import os
from datasets import load_dataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)


class Transformer(nn.Module):

    def __init__(self, src_vocab, tgt_vocab, n_emb=512, n_hidden=512, dropout=0.1, d_ff=2048, n_head=8, n_block=6):
        super(Transformer, self).__init__()
        self.n_emb = n_emb
        self.n_hidden = n_hidden
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.dropout = dropout
        self.embedding = nn.Sequential(Embeddings(n_hidden, self.src_vocab), PositionalEncoding(n_hidden, dropout))
        self.text_encoder = Encoder(n_hidden, d_ff, n_head, dropout, n_block)
        self.decoder = Decoder(n_hidden, d_ff, n_head, dropout, n_block)
        self.output_layer = Generator(self.n_hidden, self.tgt_vocab)
        self.criterion = nn.CrossEntropyLoss(reduce=False, size_average=False, ignore_index=0)

        # Initialize parameters with Xavier Initialization.
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode_text(self, X):
        embs = self.embedding(X)
        out = self.text_encoder(embs)
        return out

    def decode(self, x, m, mask):
        embs = self.embedding(x)
        out = self.decoder(embs, m, mask)
        return out

    def forward(self, X, Y):
        out_text = self.encode_text(X)
        mask = subsequent_mask(Y.size(0), Y.size(1)).requires_grad_(False)
        outs = self.decode(Y, out_text, mask)

        Y = Y.t()
        outs = outs.transpose(0, 1)

        loss = self.criterion(outs.contiguous().view(-1, self.vocab_size),
                              Y[1:].contiguous().view(-1))

        return torch.mean(loss)

def rate(step, model_size, factor, warmup):
    """
    we have to default the step to 1 for LambdaLR function
    to avoid zero raising to negative power.
    """
    if step == 0:
        step = 1
    return factor * (model_size ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5)))

def inference_test():
    test_model = Transformer(11, 11, n_block=2)
    test_model.eval()
    src = torch.LongTensor([[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])

    memory = test_model.encode_text(src)
    ys = torch.zeros(1, 1).type_as(src)

    for i in range(9):
        out = test_model.decode(
            ys, memory, subsequent_mask(ys.size(0), ys.size(1)).type_as(src.data)
        )
        print(out.shape)
        prob = test_model.output_layer(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.data[0]
        ys = torch.cat(
            [ys, torch.empty(1, 1).type_as(src.data).fill_(next_word)], dim=1
        )

    print("Example Untrained Model Prediction:", ys)

if __name__ == '__main__':
    inference_test()