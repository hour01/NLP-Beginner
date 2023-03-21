from turtle import pos
from typing import Text
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
import models
import os

from collections import Counter
import numpy as np
import random
import math

import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_similarity

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

from config import config

def get_data(path):
    if os.path.exists(config.processed_data+'/idx2word.pt'):
        return torch.load(config.processed_data+'/text.pt'), torch.load(config.processed_data+'/idx2word.pt'), \
               torch.load(config.processed_data+'/word2idx.pt'),\
               torch.load(config.processed_data+'/word_counts.pt'),torch.load(config.processed_data+'/word_freqs.pt')
    with open(path) as f:
        text = f.read()
    text = text.lower().split() 
    # get dict of word:count
    vocab_dict = dict(Counter(text).most_common(config.vocab - 1))
    # others words are marked <UNK>
    vocab_dict['<UNK>'] = len(text) - np.sum(list(vocab_dict.values())) 

    idx2word = [word for word in vocab_dict.keys()]
    word2idx = {word:i for i, word in enumerate(idx2word)}

    word_counts = np.array([count for count in vocab_dict.values()], dtype=np.float32)
    word_freqs = word_counts / np.sum(word_counts)
    word_freqs = word_freqs ** (3./4.)

    torch.save(word2idx,config.processed_data+'/word2idx.pt'),\
    torch.save(idx2word,config.processed_data+'/idx2word.pt'), torch.save(word_counts,config.processed_data+'/word_counts.pt'),\
    torch.save(word_freqs,config.processed_data+'/word_freqs.pt'), torch.save(text,config.processed_data+'/text.pt')

    return text, idx2word, word2idx, word_counts, word_freqs

class TextDataset(Dataset):
    def __init__(self, text, word2idx, idx2word, word_freqs, word_counts, window_size = config.window_size, neg_samples = config.neg_samples):
        ''' text: a list of words
            word2idx: the dictionary from word to index
            idx2word: index to word mapping
            word_freqs: the frequency of each word
            word_counts: the word counts
            C: the size of window
        '''
        super(TextDataset, self).__init__()
        self.text_encoded = torch.LongTensor([word2idx.get(word, word2idx['<UNK>']) for word in text])
        self.word2idx = word2idx
        self.idx2word = idx2word
        self.word_freqs = torch.Tensor(word_freqs)
        self.word_counts = torch.Tensor(word_counts)
        self.window_size = window_size
        self.neg_samples = neg_samples
        
        
    def __len__(self):
        return len(self.text_encoded)
    
    def __getitem__(self, idx):
        ''' return
            - 中心词
            - 这个单词附近的positive word
            - 随机采样的K个单词作为negative word
        '''
        # get the center word
        center_word = self.text_encoded[idx] 
        pos_indices = list(range(idx - self.window_size, idx)) + list(range(idx + 1, idx + self.window_size + 1)) # 先取得中心左右各C个词的索引
        pos_indices = [i % len(self.text_encoded) for i in pos_indices] # 为了避免索引越界，所以进行取余处理
        # pos_indices = list(range(max(idx - self.window_size, 0), idx)) \
        #             + list(range(idx + 1, min(idx + self.window_size + 1, len(self.text_encoded)))) 
        # to tensor
        pos_words = self.text_encoded[pos_indices] 
        neg_words = torch.multinomial(self.word_freqs, self.neg_samples * len(pos_indices), replacement=True)
        
        # torch.multinomial作用是对self.word_freqs做K * pos_words.shape[0]次取值，输出的是self.word_freqs对应的下标
        # 取样方式采用有放回的采样，并且self.word_freqs数值越大，取样概率越大
        # 每采样一个正确的单词(positive word)，就采样K个错误的单词(negative word)，pos_words.shape[0]是正确单词数量

        # 随机取样
        # neg_words = np.random.choice(len(pos_indices), (self.neg_samples * len(pos_indices),), True)

        return torch.LongTensor(center_word), torch.LongTensor(pos_words), torch.LongTensor(neg_words)
    
def find_nearest(word, word2idx, embedding_weights, idx_to_word, num):
    index = word2idx[word]
    embedding = embedding_weights[index]
    cos_dis = np.array([scipy.spatial.distance.cosine(e, embedding) for e in embedding_weights])
    return [idx_to_word[i] for i in cos_dis.argsort()[:num]]
    
if __name__ == '__main__':

    if config.mode == 'train':
        print('loading data........')
        text, idx2word, word2idx, word_counts, word_freqs = get_data(config.data_path)
        dataset = TextDataset(text, word2idx, idx2word, word_freqs, word_counts)
        dataloader_ = DataLoader(dataset, config.batch, shuffle=True)
        print('complete loading')

        model = models.Skip_gram(config.vocab, config.emb).to(device)
        optimizer = optim.SGD(model.parameters(), lr=config.lr)

        print('start training.....')
        model.train()
        for epoch in range(config.epoch):
            i = 0
            loss_s = 0
            for batch in dataloader_:
                i += 1
                center_word, pos_words, neg_words = batch
                center_word = center_word.to(device)
                pos_words = pos_words.to(device)
                neg_words = neg_words.to(device)
                
                optimizer.zero_grad()
                loss = model(center_word, pos_words, neg_words)

                loss.backward()
                optimizer.step()
                loss_s += loss
                if i%500 == 0:
                    print('loss_epoch{}:{}/{}:{}'.format(epoch, i, len(dataloader_), loss_s/500))
                    loss_s = 0
                if i%100000 == 0:
                    torch.save(model.state_dict(), config.save_path+'Skip_gram_ckpt_{}_{}_{}_{}.pt'.format(epoch, config.window_size, config.neg_samples,i))
                    
        torch.save(model.state_dict(), config.save_path+'Skip_gram_{}_{}_{}.pt'.format(config.epoch, config.window_size, config.neg_samples))
    
    else:
        if config.load_path == '':
            print('need argument -load_path')
        model = models.Skip_gram(config.vocab, config.emb).to(device)
        model_dict = torch.load(config.load_path, map_location=device)
        model.load_state_dict(model_dict)

        _, idx2word, word2idx, _, _ = get_data(config.data_path)
        while True:
            word = input('target word(type q to exit):')
            if word == 'q':
                break
            num = int(input('number of most similar words:'))
            print(find_nearest(word, word2idx, model.input_embedding(), idx2word, num))



