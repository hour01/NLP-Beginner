import torch.nn as nn
from torchcrf import CRF
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import BertModel

class Bert_BiLSTM_CRF(nn.Module):
    def __init__(self,config) -> None:
        super(Bert_BiLSTM_CRF,self).__init__()
        self.embedding_dim = 768

        self.bert = BertModel.from_pretrained(config.bert_path)
        if config.fine_tune == 'not_full':
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.bilstm = nn.LSTM(self.embedding_dim, config.hidden_size, 
                        num_layers=config.layers, batch_first=True, dropout=config.lstm_dropout, bidirectional=True)
        self.classifier = nn.Linear(config.hidden_size*2, config.vocab.label_size())
        self.crf = CRF(config.vocab.label_size(), batch_first=True)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, input_data, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        input_ids, input_token_starts = input_data
        outputs = self.bert(input_ids,
                            attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            position_ids=position_ids,
                            head_mask=head_mask,
                            inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        # 去除[CLS]标签等位置，获得与label对齐的pre_label表示
        origin_sequence_output = [layer[starts.nonzero().squeeze(1)]
                                  for layer, starts in zip(sequence_output, input_token_starts)]
        # 将sequence_output的pred_label维度padding到最大长度
        padded_sequence_output = pad_sequence(origin_sequence_output, batch_first=True)
        # dropout pred_label的一部分feature
        padded_sequence_output = self.dropout(padded_sequence_output)
        lstm_output, _ = self.bilstm(padded_sequence_output)

        logits = self.classifier(lstm_output)

        loss_mask = labels.gt(-1)
        loss = self.crf(logits, labels, loss_mask) * (-1)

        return loss, logits
