import torch
import torch.nn as nn
import torch.nn.functional as F


class MY_CNN(nn.Module):
    def __init__(self, word_vec, n_filters, filter_sizes, freeze_emb, drop_out=0.5, cgtynum=2):
        super(MY_CNN, self).__init__()

        self.embedding_dim = word_vec.shape[1]

        self.embedding = torch.nn.Embedding.from_pretrained(word_vec, freeze=freeze_emb, padding_idx=0)
        
        self.dropout = nn.Dropout(drop_out) 
        self.convs = nn.ModuleList([nn.Sequential(nn.Conv2d(1, n_filters, (filter_size, self.embedding_dim)), nn.ReLU())
                                    for filter_size in filter_sizes])
        # 全连接层
        self.fc = nn.Linear(len(filter_sizes) * n_filters, cgtynum)

    def forward(self, x):
        #x = [batch size, sent len]
        
        embedded = self.embedding(x)
                
        #embedded = [batch size, sent len, emb dim]
        
        embedded = embedded.unsqueeze(1)
        
        #embedded = [batch size, 1, sent len, emb dim]
        
        conved = [conv(embedded).squeeze(3) for conv in self.convs]
            
        #conved_n = [batch size, n_filters, sent len - filter_sizes[n] + 1]
                
        pooled = [F.max_pool1d(conv, conv.shape[2]).squeeze(2) for conv in conved]
        
        #pooled_n = [batch size, n_filters]
        
        cat = self.dropout(torch.cat(pooled, dim = 1))

        #cat = [batch size, n_filters * len(filter_sizes)]
            
        return self.fc(cat)


class MY_LSTM_h(nn.Module):
    def __init__(self, word_vec, hidden_dim, freeze_emb, n_layers, bidirectional, drop_out=0.5, cgtynum=2):
        super(MY_LSTM_h, self).__init__()
        self.embedding_dim = word_vec.shape[1]
        self.n_layers = n_layers
        self.embedding = torch.nn.Embedding.from_pretrained(word_vec, freeze=freeze_emb, padding_idx=0)
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim//2, self.n_layers, bidirectional=bidirectional,
                            dropout=drop_out if self.n_layers != 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, cgtynum)
        self.dropout = nn.Dropout(drop_out)
        
    def forward(self, x):
        # x = [batch size, sent len]

        embedded = self.dropout(self.embedding(x))

        # embedded = [batch size, seq len, embedding dim]

        output, (hidden, cell) = self.lstm(embedded)

        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # output = [batch size, seq len, hidden dim * n directions]

        if self.n_layers > 1:

            if self.lstm.bidirectional:
                # the last layer
                hidden = self.dropout(torch.cat([hidden[-1], hidden[-2]], dim=-1))
                # hidden = [batch size, hidden dim * 2]
            else:
                # the last layer
                hidden = self.dropout(hidden[-1])
                # hidden = [batch size, hidden dim]
        # no dropout
        else:
            if self.lstm.bidirectional:
                # cat tow directions
                hidden = torch.cat([hidden[-1], hidden[-2]], dim=-1)
                # hidden = [batch size, hidden dim * 2]
            else:
                hidden = hidden[-1]
                # hidden = [batch size, hidden dim]
        

        prediction = self.fc(hidden)
        # prediction = [batch size, output dim]
        return prediction


class MY_LSTM_o(nn.Module):
    def __init__(self, word_vec, hidden_dim, freeze_emb, n_layers, bidirectional, drop_out=0.5, cgtynum=2):
        super(MY_LSTM_o, self).__init__()
        self.embedding_dim = word_vec.shape[1]
        self.n_layers = n_layers
        self.embedding = torch.nn.Embedding.from_pretrained(word_vec, freeze=freeze_emb, padding_idx=0)
        self.lstm = nn.LSTM(self.embedding_dim, hidden_dim//2 if bidirectional else hidden_dim, self.n_layers, bidirectional=bidirectional,
                            dropout=drop_out if self.n_layers != 1 else 0, batch_first=True)
        self.fc = nn.Linear(hidden_dim, cgtynum)
        self.dropout = nn.Dropout(drop_out)
        
    def forward(self, x):
        # x = [batch size, sent len]

        embedded = self.dropout(self.embedding(x))

        # embedded = [batch size, seq len, embedding dim]

        output, (hidden, cell) = self.lstm(embedded)

        # hidden = [n layers * n directions, batch size, hidden dim]
        # cell = [n layers * n directions, batch size, hidden dim]
        # output = [batch size, seq len, hidden dim * n directions]

        output = torch.transpose(output, 1, 2)
        output = torch.tanh(output)

        # output = [batch size, hidden dim * n directions, seq len]

        output = F.max_pool1d(output, output.size(2)).squeeze(2)
        
        # output = [batch size, hidden dim * n directions]
        
        prediction = self.fc(output)
        # prediction = [batch size, output dim]
        return prediction