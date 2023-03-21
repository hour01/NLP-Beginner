import os
import torch
from torch.utils.data import Dataset
import gensim
import numpy as np
from transformers import BertTokenizer,BertTokenizerFast

def build_vocab(config):
    '''
    get the vocab from train, val ,test
    return the word2id(python dict), id2word(python list), word_cnt
    '''
    word2id = {'PAD': 0}
    id2word = ['PAD']
    for f in [config.train_file]:
        with open(os.path.join(config.data_path, f), 'r', encoding='utf-8') as fr:
            raw = fr.read().splitlines()
        for line in raw:
            if line == '-DOCSTART- -X- -X- O' \
            or line== '-DOCSTART- -X- O O'\
            or line == '':
                continue
            # word = line.split(' ')[0].lower()
            word = line.split(' ')[0]
            if word not in word2id:
                word2id[word] = len(word2id)
                id2word.append(word)
    word2id['UNK'] = len(word2id)
    id2word.append('UNK')
    assert len(word2id) == len(id2word)
    return word2id, id2word, len(word2id)

def get_GoogleNews_wordvec(dict, config):
    emb_model = gensim.models.KeyedVectors.load_word2vec_format(config.googlewordvec_path,binary= True)
    word_vec = torch.randn([len(dict), 300])# 没在google字典中的随机初始化
    cnt_oov = 0
    for word,id in dict.items():
        if word in emb_model:
            vector = emb_model[word]
            word_vec[id,:] = torch.from_numpy(vector)
        else:
            cnt_oov += 1
    word_vec[0,:] = 0  # padding 
    torch.save(word_vec,os.path.join(config.data_path, config.wordvec_file))
    print('dict_size:{}, oov:{}'.format(len(dict),cnt_oov))
    return word_vec

def get_Glove(dict, config):
    with open(config.glove_path,'rb') as f:  # for glove embedding
        lines=f.readlines()
    # 用GloVe创建词典
    wordvec_dict={}
    n=len(lines)
    for i in range(n):
        line=lines[i].split()
        wordvec_dict[line[0].decode("utf-8").lower()]=[float(line[j]) for j in range(1,301)]

    word_vec = torch.randn([len(dict), 300])# 没在google字典中的随机初始化
    cnt_oov = 0
    for word,id in dict.items():
        if word in wordvec_dict:
            vector = wordvec_dict[word]
            word_vec[id,:] = torch.tensor(vector)
        else:
            cnt_oov += 1
    word_vec[0,:] = 0  # padding 
    torch.save(word_vec,os.path.join(config.data_path, config.wordvec_file))
    print('dict_size:{}, oov:{}'.format(len(dict),cnt_oov))
    return word_vec

def get_dict_wordvec(config):
    '''
    get the vocab and pretrain word vector
    return : word2id(python dict), id2word(python list), word_cnt, word_vec(torch tensor)
    '''
    word2id, id2word, word_cnt = build_vocab(config)

    if os.path.exists(os.path.join(config.data_path, config.wordvec_file)):
        word_vec = torch.load(os.path.join(config.data_path, config.wordvec_file))
    else:
        word_vec = get_GoogleNews_wordvec(word2id, config)
        # word_vec = get_Glove(word2id, config)
    return  word2id, id2word, word_cnt, word_vec


class Conll03Dataset(Dataset):
    def __init__(self, config, data_path):
        super(Conll03Dataset, self).__init__()
        self.data_path = data_path
        self.word2id = config.word2id
        self.tag2id = config.tag2id
        self.raw = None
        with open(self.data_path, 'r', encoding='utf-8') as fr:
            self.raw = fr.read().splitlines()

        self.x = []
        self.y = []
        self.lengths = []
        self.preprocess()

    def preprocess(self):
        sentence_words, sentence_tags = [], []
        for line in self.raw:
            if line == '-DOCSTART- -X- -X- O' \
            or line== '-DOCSTART- -X- O O':
                continue
            if line == '':
                if len(sentence_words) == 0:
                    continue
                self.x.append(sentence_words)
                self.y.append(sentence_tags)
                self.lengths.append(len(sentence_words))
                sentence_tags = []
                sentence_words = []
                continue
            items = line.split(' ')
            # only text word and ner-tag
            # word = items[0].lower()
            word = items[0]
            ner_tag = items[3]
            if word in self.word2id:
                sentence_words.append(self.word2id[word])
            else:
                sentence_words.append(self.word2id['UNK'])
            sentence_tags.append(self.tag2id[ner_tag])
        print('{}_size:{}'.format(self.data_path, len(self.x)))

    def __len__(self):
        assert len(self.x) == len(self.y)
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.lengths[index]
    
    def collate_fn(self, batch):
        '''
        need to return a descending sort of sequence for pack_padded_sequence..
        '''
        lengths = [f[2] for f in batch]
        max_len = max(lengths)
        # get the descending_idx
        ranks = list(np.argsort(lengths))
        ranks.reverse()
        # descending and padding
        input_ids = [batch[i][0] + [0] * (max_len - lengths[i]) for i in ranks]
        # 'END' for padding
        labels = [batch[i][1] + [10] * (max_len - lengths[i]) for i in ranks]
        # mark the useful part of the non-padding(mask the padding part)
        input_mask = [[1.0] * lengths[i] + [0.0] * (max_len - lengths[i]) for i in ranks]
        labels_flatten = []
        # concatenate the labels
        for i in ranks:
            labels_flatten.extend(batch[i][1])
    
        # descending 
        lengths.sort(reverse=True)
        return input_ids, labels, labels_flatten, lengths, input_mask

class Conll03Dataset_bert(Dataset):
    def __init__(self, config, data_path):
        super(Conll03Dataset_bert, self).__init__()
        self.data_path = data_path
        self.tokenizer = BertTokenizerFast.from_pretrained(config.pretrain_path)
        self.tag2id = config.tag2id
        self.raw = None
        with open(self.data_path, 'r', encoding='utf-8') as fr:
            self.raw = fr.read().splitlines()

        self.x = []
        self.y = []
        self.lengths = []
        self.aligned_tags = []
        self.offset_map = []
        self.preprocess()
    
    def aligned_offset(self,offset,labels,original_text,after_encode_text):
        '''
        return aligned_tags:not include the [CLS]and[SEP]
        '''
        original_offset = []
        aligned_tags = []
        offsetmap = []
        word_tokens = []
        length = 0
        for word in original_text:
            l = len(word)
            original_offset.append([length,length+l])
            length += l+1
       
        j = 0
        for i,token_offset in enumerate(offset):
            if token_offset[0]>original_offset[j][0] \
                and token_offset[1]<=original_offset[j][1]:
                word_tokens.append(i)
                aligned_tags.append(labels[j])
            else:
                if(len(word_tokens)!=0):
                    offsetmap.append(word_tokens)
                    j += 1
                    word_tokens = []
                if j==len(original_text):
                    break
                # [CLS] [SEP]
                elif token_offset[0] == token_offset[1]:
                    if i==0:
                        continue
                    else:
                        break
                # an non-split word
                elif token_offset[0]==original_offset[j][0] \
                    and token_offset[1]==original_offset[j][1]:
                    aligned_tags.append(labels[j])
                    j+=1
                # start token of a word
                elif token_offset[0]==original_offset[j][0] \
                    and token_offset[1]<original_offset[j][1]:
                    word_tokens.append(i)
                    aligned_tags.append(labels[j])
                else:
                    print('error!')
                if j==len(original_text):
                    break
        return aligned_tags,offsetmap

    def preprocess(self):
        sentence_words, sentence_tags = [], []
        for line in self.raw:
            if line == '-DOCSTART- -X- -X- O' \
            or line== '-DOCSTART- -X- O O':
                continue
            if line == '':
                if len(sentence_words) == 0:
                    continue
                
                # self.x.append(self.tokenizer.encode(' '.join(sentence_words)))
                text = ' '.join(sentence_words)
                tmp = self.tokenizer.encode_plus(text, padding=True, return_tensors="pt",return_offsets_mapping=True)
                ids = tmp['input_ids'][0]
                offset = tmp['offset_mapping'][0]
                after_encode = self.tokenizer.convert_ids_to_tokens(ids)
                aligned_tags,offsetmap = self.aligned_offset(offset,sentence_tags,sentence_words,after_encode)

                assert len(ids) == len(aligned_tags)+2
                self.offset_map.append(offsetmap)
                self.aligned_tags.append(aligned_tags)
                self.x.append(ids.tolist())
                self.y.append(sentence_tags)
                self.lengths.append(len(ids))
                sentence_tags = []
                sentence_words = []
                continue
            items = line.split(' ')
            # only text word and ner-tag
            # word = items[0].lower()
            word = items[0]
            ner_tag = items[3]
            sentence_words.append(word)
            sentence_tags.append(self.tag2id[ner_tag])
        print('{}_size:{}'.format(self.data_path, len(self.x)))

    def __len__(self):
        assert len(self.x) == len(self.y)
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.lengths[index], self.aligned_tags[index], self.offset_map[index]
    
    def collate_fn(self, batch):
        '''
        need to return a descending sort of sequence for pack_padded_sequence..
        return: 
            lengths: the sequence length after encode(a word might be splited to few tokens)
            labels: original labels with no padding
            aligned_labels: labels aligned to after_encode-ids
            offset_map: idx of each tokens in a word,i.e[[word1_1,word1_2],[word2_1,word2_2,word2_3]],
                        the idx is the input_ids's idx.(including [CLS],[SEP])
        '''
        lengths = [f[2] for f in batch]
        max_len = max(lengths)
        # get the descending_idx
        ranks = list(np.argsort(lengths))
        ranks.reverse()
        # descending and padding
        input_ids = [batch[i][0] + [0] * (max_len - lengths[i]) for i in ranks]
        # 'END' for padding
        # labels = [batch[i][1] + [10] * (max_len - lengths[i]) for i in ranks]
        labels = [batch[i][1] for i in ranks]
        aligned_labels = [batch[i][3] + [10]*(max_len - lengths[i]) for i in ranks]
        offset_map = [batch[i][4] for i in ranks]
        # mark the useful part of the non-padding(mask the padding part)
        # do not mask [SEP] [CLS], these will be removed in forward()
        input_mask = [[1.0] * (lengths[i]-2) + [0.0] * (max_len - lengths[i]) for i in ranks]
        
        # descending 
        lengths.sort(reverse=True)
        return input_ids, labels, lengths, input_mask, aligned_labels, offset_map
