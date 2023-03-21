import numpy as np
import os
from my_funcs import *
from my_layers import *
from my_lossfuncs import *
import random

def get_dict_Glove():

    if os.path.exists('./data/word_dict.npy'):
        return np.load('./data/word_dict.npy',allow_pickle=True).item(), np.load('./data/word_vec.npy')

    dict = {}
    lines = open('./data/words_dict.txt',encoding='utf-8').readlines()
    for line in lines:
        dict[(line.split('|'))[0]] = len(dict)

    with open('./data/glove.6B.50d.txt','rb') as f:  # for glove embedding
        lines=f.readlines()
    # 用GloVe创建词典
    trained_dict={}
    n=len(lines)
    for i in range(n):
        line=lines[i].split()
        trained_dict[line[0].decode("utf-8").lower()]=[float(line[j]) for j in range(1,51)]

    word_vec = np.random.rand(len(dict), 50)# 没在GloVe字典中的随机初始化
    for word,id in dict.items():
        if word in trained_dict:
            word_vec[id,:] = np.array(trained_dict[word])

    np.save('./data/word_vec.npy',word_vec)
    np.save('./data/word_dict.npy',dict)
    return dict,word_vec

# embeding
def get_data_dict(dict,word_vec):

    if os.path.exists('./data/dict_train.npy'):
        return np.load('./data/dict_train.npy',allow_pickle=True).item(), np.load('./data/dict_dev.npy',allow_pickle=True).item(), np.load('./data/dict_test.npy',allow_pickle=True).item()

    dict_train = {}
    for text in open('./data/train.txt').read().splitlines(): 
        tmp = text.split('|')
        if tmp[0]=='sentiment':
                continue
        ctgy = np.zeros(2)
        ctgy[int(tmp[0])] = 1    # get the sentiment
        ctgy.astype(np.int16)
        # words in sentence
        sentence = np.array([word_vec[dict[word]] for word in (tmp[1].lower()).split(' ')])
        dict_train[len(dict_train)] = (sentence,ctgy)

    dict_dev = {}
    for text in open('./data/dev.txt').read().splitlines(): 
        tmp = text.split('|')
        if tmp[0]=='sentiment':
                continue
        ctgy = np.zeros(2)
        ctgy[int(tmp[0])] = 1    # get the sentiment
        ctgy.astype(np.int16)
        # words in sentence
        sentence = np.array([word_vec[dict[word]] for word in (tmp[1].lower()).split(' ')])
        dict_dev[len(dict_dev)] = (sentence,ctgy)
    
    dict_test = {}
    for text in open('./data/test.txt').read().splitlines(): 
        tmp = text.split('|')
        if tmp[0]=='sentiment':
                continue
        ctgy = np.zeros(2)
        ctgy[int(tmp[0])] = 1    # get the sentiment
        ctgy.astype(np.int16)
        # words in sentence
        sentence = np.array([word_vec[dict[word]] for word in (tmp[1].lower()).split(' ')])
        dict_test[len(dict_test)] = (sentence,ctgy)
    np.save('./data/dict_train.npy',dict_train)
    np.save('./data/dict_dev.npy',dict_dev)
    np.save('./data/dict_test.npy',dict_test)
    return dict_train, dict_dev, dict_test

def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic

class NN:
    def __init__(self):
        self.layers = list()

    def forward(self,x:np.ndarray)->np.ndarray:
        a = x
        for layer in self.layers:
            a = layer(a)
        return a

    def backward(self,dc_da_last:np.ndarray)->np.ndarray:
        d = dc_da_last
        for layer in self.layers[::-1]:
            #print(d)
            d = layer.backward(d)
        return d

    def train(self,input_vec,label,loss_func:LossFunc,lr):
        y = self.forward(input_vec)
        loss = loss_func.derivate(label,y)
        self.backward(loss * -lr)
        return loss_func(label,y)

    def set_layers(self,layers):
        self.layers = layers 


if __name__ == '__main__':
    dict, word_vec = get_dict_Glove()
    dict_train, dict_dev, dict_test = get_data_dict(dict,word_vec)

    my_nn = NN()
    my_nn.set_layers([
        TextCNNLayer(50, (3,50)),
        FuncLayer(relu),
        MaxPoolingLayer(),
        Dropout(p=0.5),
        FullConnectedLayer(50, 2),
        FuncLayer(softmax)
    ])

    

    for i in range(10):
        cnt = 0
        loss = 0
        for sample in random_dic(dict_train).values():
            cnt += 1
            loss += my_nn.train(sample[0], sample[1], cross_entropy, 0.001)
            #exit()
            if cnt%500 == 0:
                print('{}/{}_loss:{}'.format(cnt,len(dict_train),loss/500))
                loss = 0
        
        acc = 0
        loss = 0
        for sample in random_dic(dict_dev).values():
            predict = my_nn.forward(sample[0])
            acc += 1 if np.argmax(predict)==np.argmax(sample[1]) else 0
            loss += cross_entropy.f(sample[1], predict)
        print('dev_acc:{},loss:{}'.format(acc/len(dict_dev),loss/len(dict_dev)))
