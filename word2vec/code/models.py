import torch.nn as nn
import torch
import torch.nn.functional as F

class Skip_gram(nn.Module):
    def __init__(self, vocab_size, embed_size):
        super(Skip_gram, self).__init__()
        
        self.vocab_size = vocab_size
        self.embed_size = embed_size
        
        # as to two linear transformation with no bias
        self.in_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=True)
        self.out_embed = nn.Embedding(self.vocab_size, self.embed_size, sparse=True)
        
    def forward(self, input_labels, pos_labels, neg_labels):
        ''' input_labels: center words, [batch_size]
            pos_labels: positive words, [batch_size, (window_size * 2)]
            neg_labels: negative words, [batch_size, (window_size * 2 * K)]
            
            return: loss, [batch_size]
        '''
        input_embedding = self.in_embed(input_labels) # [batch_size, embed_size]
        pos_embedding = self.out_embed(pos_labels)# [batch_size, (window * 2), embed_size]
        neg_embedding = self.out_embed(neg_labels) # [batch_size, (window * 2 * K), embed_size]
        
        input_embedding = input_embedding.unsqueeze(2) # [batch_size, embed_size, 1]
        
        # bmm: batch matrix multiply , ignore the shape[0]
        pos_dot = torch.bmm(pos_embedding, input_embedding) # [batch_size, (window * 2), 1]
        pos_dot = pos_dot.squeeze(2) # [batch_size, (window * 2)]
        
        neg_dot = torch.bmm(neg_embedding, input_embedding) # [batch_size, (window * 2 * K), 1]
        neg_dot = neg_dot.squeeze(2) # batch_size, (window * 2 * K)]
        
        log_pos = F.logsigmoid(pos_dot).sum(1)
        log_neg = F.logsigmoid(-1 * neg_dot).sum(1)  # "-" before neg_dot, make it close to 0
        
        loss = sum(log_pos) + sum(log_neg)
        
        return -loss
    
    def input_embedding(self):
        return self.in_embed.weight.detach().numpy()