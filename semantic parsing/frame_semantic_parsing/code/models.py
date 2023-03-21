import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch import optim
from utils.conll09 import VOCDICT, POSDICT, LEMDICT, LUDICT, LUPOSDICT, FRAMEDICT, FEDICT
from utils.discrete_argid_feats import SpanWidth, ArgPosition
from utils.housekeeping import Factor

class TIBiLSTM(nn.Module):
    def __init__(self, opt, word_vec):
        super(TIBiLSTM, self).__init__()
        self.opt = opt
        self.emb_token = nn.Embedding(VOCDICT.size(), opt.configuration['token_dim'])
        self.emb_pos = nn.Embedding(POSDICT.size(), opt.configuration['pos_dim'])
        self.emb_lem = nn.Embedding(LEMDICT.size(), opt.configuration['lemma_dim'])
        self.word_vec = torch.nn.Embedding.from_pretrained(word_vec, freeze=True)

        self.emb_dim = opt.configuration['token_dim']+opt.configuration['pos_dim']+opt.configuration['lemma_dim']+100
        self.linear = nn.Linear(self.emb_dim, opt.configuration["lstm_input_dim"])
        self.dropout = nn.Dropout(opt.configuration['dropout_rate'])

        self.bilstm = nn.LSTM(int(opt.configuration["lstm_input_dim"]), int(opt.configuration["lstm_dim"]),int(opt.configuration["lstm_depth"]), \
                              bidirectional=True, dropout=opt.configuration['dropout_rate'] if opt.configuration["lstm_depth"] != 1 else 0,\
                                proj_size = int(opt.configuration["lstm_dim"]/2),batch_first = True)
        self.fc = nn.Linear(2*int(opt.configuration["lstm_dim"]/2), 2)
        # self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, tokens, postags, lemmas):
        emb = torch.cat((self.emb_token(tokens), self.emb_pos(postags), self.emb_lem(lemmas), self.word_vec(tokens)),dim=1)
        x = self.dropout(F.relu(self.linear(emb)))
        # x[1, L, Hin]
        x = x.unsqueeze(0)
        lstm_out,_ = self.bilstm(x)
        # lstm_out[1,L, 2*Hout] [1,L,100]
        lstm_out = lstm_out.squeeze(0)
        scores = self.fc(self.dropout(lstm_out))
        # lstm_out[L,2]
        return scores

class FIBiLSTM(nn.Module):
    def __init__(self, opt, word_vec):
        super(FIBiLSTM, self).__init__()
        self.opt = opt
        self.emb_token = nn.Embedding(VOCDICT.size(), opt.configuration['token_dim'])
        self.emb_pos = nn.Embedding(POSDICT.size(), opt.configuration['pos_dim'])
        self.word_vec = torch.nn.Embedding.from_pretrained(word_vec, freeze=True)
        self.emb_lu = nn.Embedding(LUDICT.size(), opt.configuration['lu_dim'])
        self.emb_lupos = nn.Embedding(LUPOSDICT.size(), opt.configuration['lu_pos_dim'])

        self.emb_dim = opt.configuration['token_dim']+opt.configuration['pos_dim']+100
        self.linear = nn.Linear(self.emb_dim, opt.configuration["lstm_input_dim"])
        self.dropout = nn.Dropout(opt.configuration['dropout_rate'])

        self.bilstm = nn.LSTM(int(opt.configuration["lstm_input_dim"]), int(opt.configuration["lstm_dim"]),int(opt.configuration["lstm_depth"]), \
                              bidirectional=True, dropout=opt.configuration['dropout_rate'] if opt.configuration["lstm_depth"] != 1 else 0,\
                              batch_first = True)
        self.target_lstm = nn.LSTM(opt.configuration["lstm_dim"]*2, opt.configuration["lstm_dim"], opt.configuration["lstm_depth"],\
                               bidirectional=False, dropout=0, batch_first = True)
        self.fc1 = nn.Linear(opt.configuration["lstm_dim"]+opt.configuration['lu_dim']+opt.configuration['lu_pos_dim'],\
                              opt.configuration["hidden_dim"])
        self.fc2 = nn.Linear(opt.configuration["hidden_dim"], FRAMEDICT.size())
        # self.log_softmax = nn.LogSoftmax(dim=1)
    
    def forward(self, tokens, postags, lu_id, lu_pos, target_position):
        emb = torch.cat((self.emb_token(tokens), self.emb_pos(postags), self.word_vec(tokens)),dim=1)
        x = self.dropout(F.relu(self.linear(emb)))
        # x[1, L, Hin]
        x = x.unsqueeze(0)
        lstm_out,_ = self.bilstm(x)
        # lstm_out[1,L, 2*Hout] [1,L,200]
        target_lstm_out = torch.index_select(lstm_out, 1, target_position)
        # target_lstm_out[1,target_len,2*Hout]
        target_vec,_ = self.target_lstm(target_lstm_out)
        # target_vec[1,1,100]
        target_vec = target_vec[:,-1,:]

        fbemb = torch.cat((target_vec, self.emb_lu(lu_id).unsqueeze(0), self.emb_lupos(lu_pos).unsqueeze(0)),dim=1)
        
        scores = self.fc2(F.relu(self.fc1(fbemb)))
        scores = self.dropout(scores)
        # lstm_out[1, Frame_class]
        return scores

class SegRNN(nn.Module):
    def __init__(self, opt, word_vec):
        super(SegRNN, self).__init__()
        self.opt = opt
        self.emb_token = nn.Embedding(VOCDICT.size(), opt.configuration['token_dim'])
        self.emb_pos = nn.Embedding(POSDICT.size(), opt.configuration['pos_dim'])
        self.linear1 = nn.Linear(opt.configuration['token_dim']+opt.configuration['pos_dim']+1, opt.configuration["lstm_input_dim"])
        self.word_vec = torch.nn.Embedding.from_pretrained(word_vec, freeze=True)
        self.linear2 = nn.Linear(opt.configuration['pretrained_embedding_dim'], opt.configuration["lstm_input_dim"])
        self.dropout = nn.Dropout(opt.configuration['dropout_rate'])
        self.baselstm = nn.LSTM(int(opt.configuration["lstm_input_dim"]), int(opt.configuration["lstm_input_dim"]),int(opt.configuration["lstm_depth"]), \
                              bidirectional=True, dropout=opt.configuration['dropout_rate'] if opt.configuration["lstm_depth"] != 1 else 0,\
                              batch_first = True)
        self.linear3 = nn.Linear(opt.configuration['lstm_input_dim']*2, opt.configuration["lstm_input_dim"])
        
        self.target_lstm = nn.LSTM(opt.configuration["lstm_input_dim"], opt.configuration["lstm_dim"], opt.configuration["lstm_depth"],\
                               bidirectional=False, dropout=0, batch_first = True)
        self.ctxtar_lstm = nn.LSTM(opt.configuration["lstm_input_dim"], opt.configuration["lstm_dim"], opt.configuration["lstm_depth"],\
                               bidirectional=False, dropout=0, batch_first = True)
        self.emb_lu = nn.Embedding(LUDICT.size(), opt.configuration['lu_dim'])
        self.emb_lupos = nn.Embedding(LUPOSDICT.size(), opt.configuration['lu_pos_dim'])
        self.emb_frame = nn.Embedding(FRAMEDICT.size(), opt.configuration['frame_dim'])
        
        self.span_lstm = nn.LSTM(int(opt.configuration["lstm_input_dim"]), int(opt.configuration["lstm_dim"]),int(opt.configuration["lstm_depth"]), \
                              bidirectional=True, dropout=opt.configuration['dropout_rate'] if opt.configuration["lstm_depth"] != 1 else 0,\
                              batch_first = True)
        self.emb_span_len = nn.Embedding(SpanWidth.size(), SpanWidth.size())
        self.emb_span_relative_posi = nn.Embedding(ArgPosition.size(), ArgPosition.size())
        self.emb_frame_element = nn.Embedding(FEDICT.size(), opt.configuration["fe_dim"])
        self.fc1 = nn.Linear(opt.configuration["lstm_dim"]*2+opt.configuration['lu_dim']+opt.configuration['lu_pos_dim']\
                             +opt.configuration['frame_dim']+opt.configuration["lstm_input_dim"]+opt.configuration["lstm_dim"]\
                                +opt.configuration["fe_dim"]+ArgPosition.size()+SpanWidth.size()+2,
                                opt.configuration["hidden_dim"])
        self.fc2 = nn.Linear(opt.configuration["hidden_dim"], 1)
    
    def get_base_embedding(self, tokens, postags, relative_position):
        # base emb encode
        emb = torch.cat((self.emb_token(tokens), self.emb_pos(postags), relative_position.view(-1,1)),dim=1)
        # [len, 65]
        baseinput_x = self.linear1(emb)
        # [len, 64]
        baseinput_x = F.relu(baseinput_x + self.linear2(self.word_vec(tokens)))
        # [len, 64]
        baseinput_x = self.dropout(baseinput_x).unsqueeze(0)
        base_lstm_out,_ = self.baselstm(baseinput_x)
        base_lstm_out = F.relu(self.linear3(base_lstm_out))
        #[1, len, 64]
        return base_lstm_out
    
    def get_target_frame_embedding(self, tokens, lu_id, lu_pos, frame_id, base_lstm_out, target_position):
        # target frame encode
        target_lstm_out = torch.index_select(base_lstm_out, 1, target_position)
        #[1, target_len, 64]
        target_vec,_ = self.target_lstm(target_lstm_out)
        target_vec = target_vec[:,-1,:]
        # target_vec[1,64]

        # Adding context features of the target span
        ctxt = target_position
        if ctxt[0].item() > 0: ctxt = torch.cat((ctxt[0].reshape(-1),ctxt))
        if ctxt[-1].item() < len(tokens): ctxt = torch.cat((ctxt,ctxt[-1].reshape(-1)))
        target_context = torch.index_select(base_lstm_out, 1, ctxt)
        #[1, target_len, 64]
        target_ctxt_x,_ = self.ctxtar_lstm(target_context)
        target_ctxt_x = target_ctxt_x[:,-1,:]
        # target_ctxt_x[1,64]
        target_frame_emb = torch.cat((self.emb_lu(lu_id).unsqueeze(0), self.emb_lupos(lu_pos).unsqueeze(0),\
                                       self.emb_frame(frame_id).unsqueeze(0), target_vec, target_ctxt_x), dim=1)
        return target_frame_emb
    
    def get_span_embedding(self, base_lstm_emb):
        # [1,len,64]
        sentlen = base_lstm_emb.shape[1]
        # span_emb_fw[i][j]: forward LSTM hidden state from i to j
        # span_emb_bw[i][j]: backward LSTM hidden state from j to i
        span_emb_list = [[None for _ in range(sentlen)] for _ in range(sentlen)]

        for i in range(sentlen):
            for j in range(i,sentlen):
                if self.opt.configuration['use_span_clip'] and \
                j-i+1 > self.opt.configuration["allowed_max_span_length"]:
                    continue
                tmp,_ = self.span_lstm(base_lstm_emb[:,i:j+1,:])
                # [1, j-i+1, 128]
                assert tmp.shape[1] == j-i+1
                tmp = tmp.view(-1,2,64)
                # [j-i+1, 2, 64]
                # cat the forward lstm hidden state from i to j and backward hidden state from j to i
                span_emb_list[i][j] = torch.cat((tmp[-1,0,:],tmp[0,1,:])).unsqueeze(0)
        return span_emb_list

    def get_factor_expressions(self, span_emb_list, target_frame_emb, target_position, valid_fes):
        factexprs_dict = {}
        sentlen = len(span_emb_list)
        for i in range(sentlen):
            for j in range(i,sentlen):
                if self.opt.configuration['use_span_clip'] and \
                j-i+1 > self.opt.configuration["allowed_max_span_length"]:
                    continue
                spanlen = torch.tensor(j-i+1,device=self.opt.device)#.to(self.opt.device)
                logspanlen = torch.log(spanlen).to(self.opt.device)
                span_len_emb = self.emb_span_len(torch.tensor(SpanWidth.howlongisspan(i,j),device=self.opt.device))#.to(self.opt.device))
                span_posi_emb = self.emb_span_relative_posi(torch.tensor(\
                    ArgPosition.whereisarg((i, j), (target_position[0],target_position[-1])),device=self.opt.device))#.to(self.opt.device))
                # [1,xxx]
                span_feat = torch.cat((span_emb_list[i][j], target_frame_emb,\
                                        spanlen.view(1,1), logspanlen.view(1,1), \
                                            span_len_emb.unsqueeze(0), span_posi_emb.unsqueeze(0)),dim=1)
                # get the frame element embedding (vy in paper)
                # compute the scores(factexprs), which includes (ALLOWED_SPANLEN*sent_length*valid_fes) types
                for y in valid_fes:
                    fctr = Factor(i,j,y)
                    span_feat_ijy = torch.cat((span_feat, self.emb_frame_element(torch.tensor(y,device=self.opt.device)).unsqueeze(0)),dim=1)#.to(self.opt.device)).unsqueeze(0)),dim=1)
                    factexprs_dict[fctr] = self.fc2(F.relu(self.fc1(span_feat_ijy))).squeeze(0).squeeze(0)
        return factexprs_dict

    def forward(self, tokens, postags, relative_position, lu_id, lu_pos, frame_id, target_position, valid_frame_elements):
        #[1, len, 64]
        base_lstm_emb = self.get_base_embedding(tokens, postags, relative_position)

        #[1, 294]
        target_frame_emb = self.get_target_frame_embedding(tokens, lu_id, lu_pos, frame_id, base_lstm_emb, target_position)

        #[len,len]:[1,128]
        span_emb_list = self.get_span_embedding(base_lstm_emb)

        # {Factor:tensor}tesor.shape=[none]
        factexprs_dict = self.get_factor_expressions(span_emb_list, target_frame_emb, target_position, valid_frame_elements)

        return factexprs_dict

        
        

