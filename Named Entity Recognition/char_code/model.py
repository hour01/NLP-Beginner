import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from utils import log_sum_exp

class BiLSTM_CRF(nn.Module):
    def __init__(self, config):
        super(BiLSTM_CRF, self).__init__()
        self.vocab_size = config.vocab_size
        self.embedding_dim = config.word_emb_dim + config.char_emb_dim
        self.char_embedding_dim = config.char_emb_dim
        
        self.hidden_dim = config.hidden_dim

        self.char_emb = nn.Embedding(len(config.char2id), self.char_embedding_dim)

        self.char_cnn = nn.Conv2d(in_channels=1, out_channels=self.char_embedding_dim, \
                        kernel_size=(config.kernel_size, self.char_embedding_dim), padding=(1,0))

        # # high way
        # self.hw_trans = nn.Linear(self.char_embedding_dim, self.char_embedding_dim)
        # self.hw_gate = nn.Linear(self.char_embedding_dim, self.char_embedding_dim)

        self.word_emb = nn.Embedding.from_pretrained(config.word_vec, freeze=config.freeze_emb, padding_idx=0)
        # self.emb = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, 
                        num_layers=config.layers, batch_first=True, bidirectional=True)

        self.crf = CRF(self.hidden_dim*2, len(config.tag2id))
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, sent, chars, labels, lengths, masks):
        # embedded = self.dropout(self.emb(sent))
        word_embedded = self.word_emb(sent)
        # chars = [batch, len, word_len]
        char_embedded = self.char_emb(chars)
        # char_embedded = [batch, len, word_len, char_emb_dim]
        char_embedded = char_embedded.view(-1,char_embedded.shape[2],char_embedded.shape[3]).unsqueeze(1)
        # char_embedded = [batch*len, 1, word_len, char_emb_dim]
        char_embedded = self.dropout(char_embedded)
        charcnn_output = self.char_cnn(char_embedded).squeeze(-1)
        # charcnn_output = [batch*len, output_cannels, word_len+2 - filter_sizes_1 +1]
        charcnn_output = F.max_pool1d(charcnn_output, charcnn_output.shape[2]).squeeze(2)
        # charcnn_output = [batch*len, output_cannels]
        charcnn_output = charcnn_output.view(chars.shape[0],chars.shape[1],-1)
        # charcnn_output = [batch, len, output_cannels]

        # t = self.hw_gate(charcnn_output) # high way
        # g = torch.sigmoid(t)
        # h = nn.functional.relu(self.hw_trans(charcnn_output))
        # chars_embeds = g * h + (1 - g) * charcnn_output

        embedded = torch.cat((word_embedded,charcnn_output),2)
        #embedded = [batch, len, wordemb+charemb]
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



IMPOSSIBLE = -1e4
class CRF(nn.Module):
    """
    :param in_features: number of features for the input
    :param num_tag: number of tags. The START, STOP tags are included internal.
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