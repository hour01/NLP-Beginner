import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils import log_sum_exp
from transformers import BertTokenizer, BertModel


class BiLSTM(nn.Module):
    def __init__(self, config):
        super(BiLSTM, self).__init__()
        self.config = config

        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size
        self.tag2id = config.tag2id
        self.tagset_size = len(self.tag2id)

        # self.emb = nn.Embedding(self.vocab_size, self.emb_dim)
        self.emb = nn.Embedding.from_pretrained(config.word_vec, freeze=config.freeze_emb, padding_idx=0)
        self.lstm = nn.LSTM(self.emb_dim, self.hidden_dim, num_layers=config.layers, batch_first=True, bidirectional=True)
        # linear layer, predict the probability of each tag
        self.hidden2tag = nn.Linear(2 * self.hidden_dim, self.tagset_size)

        # Loss: compute the distance between our prediction and the gold tag
        self.loss = CrossEntropyLoss()

    def forward(self, sent, labels, lengths, mask):
        embedded = self.emb(sent)
        # [batch_size, max_len, emb_size]
        # The padded batch should be packed before LSTM
        embedded = pack_padded_sequence(embedded, lengths, batch_first=True)
        lstm_out, _ = self.lstm(embedded)
        # The packed batch should be padded after LSTM
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)   # lstm_out: [batch_size, max_len, hidden_dim]
        logits = self.hidden2tag(lstm_out)  # logits: [batch_size, max_len, tagset_size]

        # Predict the tags
        pred_tag = torch.argmax(logits, dim=-1)

        # Compute loss. Pad token must be masked before computing the loss.
        logits = logits.view(-1, self.tagset_size)[mask.view(-1) == 1.0]
        loss = self.loss(logits, labels.view(-1))

        return loss, pred_tag

class BiLSTM_CRF(nn.Module):
    def __init__(self, config):
        super(BiLSTM_CRF, self).__init__()
        self.embedding_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.vocab_size = config.vocab_size

        self.emb = nn.Embedding.from_pretrained(config.word_vec, freeze=config.freeze_emb, padding_idx=0)
        # self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, 
                        num_layers=config.layers, batch_first=True, bidirectional=True)
        self.crf = CRF(self.hidden_dim*2, len(config.tag2id))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, sent, labels, lengths, masks):
        # embedded = self.dropout(self.emb(sent))
        embedded = self.emb(sent)
        embedded = self.dropout(embedded)

        # The padded batch should be packed before LSTM
        embedded = pack_padded_sequence(embedded, lengths, batch_first=True)
        lstm_out, _ = self.lstm(embedded)
        # The packed batch should be padded after LSTM
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True) 

        lstm_out = self.dropout(lstm_out)

        loss = self.crf.loss(lstm_out, labels, masks)
        score, tag_seq = self.crf(lstm_out, masks)
        
        return loss, tag_seq

class Bert_CRF(nn.Module):
    def __init__(self, config):
        super(Bert_CRF, self).__init__()

        self.embedding_dim = 768

        self.model_bert = BertModel.from_pretrained(config.pretrain_path).to(config.device)
        if config.freeze_bert:
            for param in self.model_bert.parameters():
                param.requires_grad = False
        
        self.crf = CRF(self.hidden_dim, len(config.tag2id))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, sent, labels, masks):
        
        bert_out = self.dropout(self.model_bert(sent)[0])
        # remove the [CLS] [SEP] 
        bert_out = bert_out[:,1:-1,:]
        # print(bert_out.shape,labels.shape,masks.shape)
        loss = self.crf.loss(bert_out, labels, masks)
        score, tag_seq = self.crf(bert_out, masks)
        
        return loss, tag_seq

class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self,config) -> None:
        super(Bert_BiLSTM_CRF,self).__init__()
        self.embedding_dim = 768
        self.hidden_dim = config.hidden_dim

        self.model_bert = BertModel.from_pretrained(config.pretrain_path).to(config.device)
        if config.freeze_bert:
            for param in self.model_bert.parameters():
                param.requires_grad = False
        
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, 
                        num_layers=config.layers, batch_first=True, bidirectional=True)
        self.crf = CRF(self.hidden_dim*2, len(config.tag2id))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, sent, labels, lengths, masks):
        
        bert_out = self.dropout(self.model_bert(sent)[0])
        # remove the [CLS] [SEP], assume [SEP] as <PAD>.
        bert_out = bert_out[:,1:-1,:]
        # print(bert_out.shape,labels.shape,masks.shape)
        lengths = [m-2 for m in lengths]
        # The padded batch should be packed before LSTM
        embedded = pack_padded_sequence(bert_out, lengths, batch_first=True)
        lstm_out, _ = self.lstm(embedded)
        # The packed batch should be padded after LSTM
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True) 

        loss = self.crf.loss(lstm_out, labels, masks)
        score, tag_seq = self.crf(lstm_out, masks)
        
        return loss, tag_seq

IMPOSSIBLE = -1e4
class CRF(nn.Module):
    """General CRF module.
    The CRF module contain a inner Linear Layer which transform the input from features space to tag space.
    :param in_features: number of features for the input
    :param num_tag: number of tags. DO NOT include START, STOP tags, they are included internal.
    """

    def __init__(self, in_features, num_tags):
        super(CRF, self).__init__()

        # self.num_tags = num_tags + 2
        self.num_tags = num_tags
        self.start_idx = self.num_tags - 2
        self.stop_idx = self.num_tags - 1

        self.fc = nn.Linear(in_features, self.num_tags)

        # transition factor, Tij mean transition from j to i
        self.transitions = nn.Parameter(torch.randn(self.num_tags, self.num_tags), requires_grad=True)
        self.transitions.data[self.start_idx, :] = IMPOSSIBLE
        self.transitions.data[:, self.stop_idx] = IMPOSSIBLE

    def forward(self, features, masks):
        """decode tags
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
        """
        features = self.fc(features)
        return self.__viterbi_decode(features, masks[:, :features.size(1)].float())

    def loss(self, features, ys, masks):
        """negative log likelihood loss
        B: batch size, L: sequence length, D: dimension
        :param features: [B, L, D]
        :param ys: tags, [B, L]
        :param masks: masks for padding, [B, L]
        :return: loss
        """
        features = self.fc(features)

        L = features.size(1)
        masks_ = masks[:, :L].float()

        forward_score = self.__forward_algorithm(features, masks_)
        gold_score = self.__score_sentence(features, ys[:, :L].long(), masks_)
        loss = (forward_score - gold_score).mean()
        return loss

    def __score_sentence(self, features, tags, masks):
        """Gives the score of a provided tag sequence
        :param features: [B, L, C]
        :param tags: [B, L]
        :param masks: [B, L]
        :return: [B] score in the log space
        """
        B, L, C = features.shape

        # emission score
        emit_scores = features.gather(dim=2, index=tags.unsqueeze(-1)).squeeze(-1)

        # transition score
        start_tag = torch.full((B, 1), self.start_idx, dtype=torch.long, device=tags.device)
        tags = torch.cat([start_tag, tags], dim=1)  # [B, L+1]
        trans_scores = self.transitions[tags[:, 1:], tags[:, :-1]]

        # last transition score to STOP tag
        last_tag = tags.gather(dim=1, index=masks.sum(1).long().unsqueeze(1)).squeeze(1)  # [B]
        last_score = self.transitions[self.stop_idx, last_tag]

        score = ((trans_scores + emit_scores) * masks).sum(1) + last_score
        return score

    def __viterbi_decode(self, features, masks):
        """decode to tags using viterbi algorithm
        :param features: [B, L, C], batch of unary scores
        :param masks: [B, L] masks
        :return: (best_score, best_paths)
            best_score: [B]
            best_paths: [B, L]
            No Padding:for any b of B, L might be different.
        """
        B, L, C = features.shape

        bps = torch.zeros(B, L, C, dtype=torch.long, device=features.device)  # back pointers

        # Initialize the viterbi variables in log space
        max_score = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        max_score[:, self.start_idx] = 0

        for t in range(L):
            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            emit_score_t = features[:, t]  # [B, C]

            # [B, 1, C] + [C, C]
            acc_score_t = max_score.unsqueeze(1) + self.transitions  # [B, C, C]
            acc_score_t, bps[:, t, :] = acc_score_t.max(dim=-1)
            acc_score_t += emit_score_t
            max_score = acc_score_t * mask_t + max_score * (1 - mask_t)  # max_score or acc_score_t

        # Transition to STOP_TAG
        max_score += self.transitions[self.stop_idx]
        best_score, best_tag = max_score.max(dim=-1)

        # Follow the back pointers to decode the best path.
        best_paths = []
        bps = bps.cpu().numpy()
        for b in range(B):
            best_tag_b = best_tag[b].item()
            seq_len = int(masks[b, :].sum().item())

            best_path = [best_tag_b]
            for bps_t in reversed(bps[b, :seq_len]):
                best_tag_b = bps_t[best_tag_b]
                best_path.append(best_tag_b)
            # drop the last tag and reverse the left
            best_paths.append(best_path[-2::-1])

        return best_score, best_paths

    def __forward_algorithm(self, features, masks):
        """calculate the partition function with forward algorithm.
        TRICK: log_sum_exp([x1, x2, x3, x4, ...]) = log_sum_exp([log_sum_exp([x1, x2]), log_sum_exp([x3, x4]), ...])
        :param features: features. [B, L, C]
        :param masks: [B, L] masks
        :return:    [B], score in the log space
        """
        B, L, C = features.shape

        scores = torch.full((B, C), IMPOSSIBLE, device=features.device)  # [B, C]
        scores[:, self.start_idx] = 0.
        trans = self.transitions.unsqueeze(0)  # [1, C, C]

        # Iterate through the sentence
        for t in range(L):
            emit_score_t = features[:, t].unsqueeze(2)  # [B, C, 1]
            score_t = scores.unsqueeze(1) + trans + emit_score_t  # [B, 1, C] + [1, C, C] + [B, C, 1] => [B, C, C]
            score_t = log_sum_exp(score_t)  # [B, C]

            mask_t = masks[:, t].unsqueeze(1)  # [B, 1]
            scores = score_t * mask_t + scores * (1 - mask_t)
        scores = log_sum_exp(scores + self.transitions[self.stop_idx])
        return scores
