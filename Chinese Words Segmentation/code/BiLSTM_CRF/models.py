import torch.nn as nn
from torchcrf import CRF
import torch


class BiLSTM_CRF(nn.Module):

    def __init__(self, config):
        super(BiLSTM_CRF, self).__init__()
        # nn.Embedding: parameter size (num_words, embedding_dim)
        # for every word id, output a embedding for this word
        # input size: N x W, N is batch size, W is max sentence len
        # output size: (N, W, embedding_dim), embedding all the words
        self.embedding = nn.Embedding.from_pretrained(config.word_vec, freeze=False)
        self.bilstm = nn.LSTM(
            input_size=config.emb_size,
            hidden_size=config.hidden_size,
            batch_first=True,
            num_layers=config.layers,
            dropout=config.lstm_dropout if config.layers > 1 else 0,
            bidirectional=True
        )
        self.classifier = nn.Linear(config.hidden_size * 2, config.vocab.label_size())

        # https://pytorch-crf.readthedocs.io/en/stable/_modules/torchcrf.html
        self.crf = CRF(config.vocab.label_size(), batch_first=True)
        for p in self.crf.parameters():       
            _ = torch.nn.init.uniform_(p, -1, 1)

    def forward(self, unigrams, input_mask, input_tags):
        uni_embeddings = self.embedding(unigrams)   
        sequence_output, _ = self.bilstm(uni_embeddings)       
        tag_scores = self.classifier(sequence_output)  
        loss = self.crf(tag_scores, input_tags, input_mask) * (-1) 
        return tag_scores, loss
