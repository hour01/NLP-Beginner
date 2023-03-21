import os
import torch
from torch.utils.data import Dataset
import numpy as np

def build_vocab(config):
    '''
    get the vocab from train_set
    return the word2id(python dict), id2word(python list), word_cnt, char2id
    '''
    word2id = {'PAD': 0}  
    id2word = ['PAD']
    char2id = {}
    # for f in [config.train_file]:
    #     with open(os.path.join(config.data_path, f), 'r', encoding='utf-8') as fr:
    with open(os.path.join(config.data_path, config.train_file), 'r', encoding='utf-8') as fr:
        raw = fr.read().splitlines()
        for line in raw:
            if line == '-DOCSTART- -X- -X- O' \
            or line== '-DOCSTART- -X- O O'\
            or line == '':
                continue
            word = line.split(' ')[0]
            for c in word:
                if c not in char2id:
                    char2id[c] = len(char2id)
            if word.lower() not in word2id:
                word2id[word.lower()] = len(word2id)
                id2word.append(word.lower())
    word2id['UNK'] = len(word2id)
    id2word.append('UNK')
    char2id['UNK'] = len(char2id)
    assert len(word2id) == len(id2word)
    return word2id, id2word, len(word2id), char2id

def get_Glove(dict, config):
    with open(config.glove_path,'rb') as f:  # for glove embedding
        lines=f.readlines()
    # 用GloVe创建词典
    wordvec_dict={}
    n=len(lines)
    for i in range(n):
        line=lines[i].split()
        wordvec_dict[line[0].decode("utf-8").lower()]=[float(line[j]) for j in range(1,101)]

    word_vec = torch.randn([len(dict), 100])# 没在google字典中的随机初始化
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
    word2id, id2word, word_cnt, char2id = build_vocab(config)

    if os.path.exists(os.path.join(config.data_path, config.wordvec_file)):
        word_vec = torch.load(os.path.join(config.data_path, config.wordvec_file))
    else:
        word_vec = get_Glove(word2id, config)
    return  word2id, id2word, word_cnt, word_vec, char2id


class Conll03Dataset(Dataset):
    def __init__(self, config, data_path):
        super(Conll03Dataset, self).__init__()
        self.data_path = data_path
        self.word2id = config.word2id
        self.tag2id = config.tag2id
        self.char2id = config.char2id
        self.raw = None
        with open(self.data_path, 'r', encoding='utf-8') as fr:
            self.raw = fr.read().splitlines()

        # words ids
        self.x = []
        # ner tags
        self.y = []
        self.lengths = []
        # char for each word in sentence
        # shape like [batch, len, word_size]  [[[a],[b],[c]],...]
        # only capture the char in char2id
        self.chars = []
        # the capital form of each word in sentence
        # self.cap = []
        self.preprocess()

    def capital_feature(self,word):
        '''
        Capitalization feature:
        0 = lower case
        1 = all capital form
        2 = first letter captital
        3 = others
        '''
        if word.lower() == word:
            return 0
        elif word.upper() == word:
            return 1
        elif word[0].upper() == word[0]:
            return 2
        else:
            return 3

    def preprocess(self):
        sentence_words, sentence_tags = [], []
        sentence_words_chars = []
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
                self.chars.append(sentence_words_chars)
                # self.cap.append(sentence_words_cap)
                sentence_words, sentence_tags = [], []
                sentence_words_chars = []
                continue
            items = line.split(' ')
            # only text word and ner-tag
            # word = items[0].lower()
            word = items[0]
            ner_tag = items[3]
            sentence_words.append(self.word2id[word.lower()] \
                if word.lower() in self.word2id else self.word2id['UNK'])  
            sentence_tags.append(self.tag2id[ner_tag])
            sentence_words_chars.append([self.char2id[c] for c in word if c in self.char2id])
            # each char of the word is not in char2id
            if sentence_words_chars[-1] == []:
                sentence_words_chars.pop()
                sentence_words_chars.append([self.char2id['UNK']])
            # sentence_words_cap.append(self.capital_feature(word))
        print('{}_size:{}'.format(self.data_path, len(self.x)))

    def __len__(self):
        assert len(self.x) == len(self.y)
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.lengths[index], self.chars[index]#, self.cap[index] 
    
    def pad_chars(self, chars, max_len, max_len_word):
        '''
        pad the chars in char level of each word in a sentence
        chars: word's char of a single sentence ,[[word1],[word2],...]
        max_len: max_length of sentence in this batch
        length: length of this sentence
        mode: for 'LSTM' or 'CNN'
        return: padded chars, char_length
        '''
        # if mode == 'LSTM':
        #     # srot for char_len
        #     chars_sorted = sorted(chars, key=lambda p: len(p), reverse=True)
        #     d = {}
        #     # get the map : sorted to before sort
        #     # to get the right order when cat the emb to word_emb
        #     for i, ci in enumerate(chars):
        #         for j, cj in enumerate(chars_sorted):
        #             if ci == cj and not j in d and i not in d.values():
        #                 d[j] = i
        #     chars_length = [len(c) for c in chars_sorted]
        #     char_maxl = max(chars2_length)
        #     chars2_mask = np.zeros((len(chars_sorted), char_maxl), dtype='int')
        #     for i, c in enumerate(chars_sorted):
        #         chars2_mask[i, :chars2_length[i]] = c

        # if mode == 'CNN':
        # don't have to order by length
        # length of each word
        chars2_length = [len(c) for c in chars]
        char_maxl = max(chars2_length)
        # chars_mask = np.zeros((len(chars2_length), char_maxl), dtype='int')
        # max_len used to pad like the other sentence in the same batch
        chars_mask = np.zeros((max_len, max_len_word), dtype='int')
        for i, c in enumerate(chars):
            chars_mask[i, :chars2_length[i]] = c
        
        return chars_mask
        
    
    def collate_fn(self, batch):
        '''
        need to return a descending sort of sequence for pack_padded_sequence..
        '''
        lengths = [f[2] for f in batch]
        max_len = max(lengths)
        # max_len for word 
        max_len_word = 0
        for f in batch:
            for word in f[3]:
                max_len_word = max(max_len_word,len(word))

        # get the descending_idx
        ranks = list(np.argsort(lengths))
        ranks.reverse()
        # descending and padding
        input_ids = [batch[i][0] + [0] * (max_len - lengths[i]) for i in ranks]
        # 'END' for padding
        labels = [batch[i][1] + [10] * (max_len - lengths[i]) for i in ranks]
        # no padding at sentence length level
        chars = [self.pad_chars(batch[i][3], max_len, max_len_word) for i in ranks]
        # chars = self.pad_chars(chars)
        # mark the useful part of the non-padding(mask the padding part)
        input_mask = [[1.0] * lengths[i] + [0.0] * (max_len - lengths[i]) for i in ranks]
        labels_flatten = []
        # concatenate the labels
        for i in ranks:
            labels_flatten.extend(batch[i][1])
    
        # descending 
        lengths.sort(reverse=True)
        chars = np.array(chars)
        return input_ids, labels, labels_flatten, lengths, input_mask, chars
