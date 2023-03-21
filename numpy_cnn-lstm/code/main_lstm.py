import numpy as np
import os
from my_lstm import LSTM, crossEntropy, crossEntropy_derivate
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
def get_data_dict(dict, word_vec, batch_size):

    if os.path.exists('./data/dict_train_batch_{}.npy'.format(batch_size)):
        return np.load('./data/dict_train_batch_{}.npy'.format(batch_size),allow_pickle=True).item(), np.load('./data/dict_dev_batch_{}.npy'.format(batch_size),allow_pickle=True).item(), np.load('./data/dict_test_batch_{}.npy'.format(batch_size),allow_pickle=True).item()

    dict_train = {}
    cnt = 0
    l = 0
    sentences = []
    class_ = np.zeros((2,batch_size))
    class_.astype(np.int16)
    for text in open('./data/train.txt').read().splitlines(): 
        
        tmp = text.split('|')
        if tmp[0]=='sentiment':
                continue
        class_[int(tmp[0]),cnt] = 1   # get the sentiment
        # words in sentence
        sentence = np.array([word_vec[dict[word]] for word in (tmp[1].lower()).split(' ')])
        l = max(l,len(sentence))
        sentences.append(sentence)
        cnt += 1
        if cnt == batch_size:
            batch = np.zeros((50,batch_size,l))
            for i in range(0,batch_size):
                batch[:,i,:] = np.pad(sentences[i],((0,l-sentences[i].shape[0]),(0,0)),'constant').T
            dict_train[len(dict_train)] = (batch,class_)
            l = 0
            cnt = 0
            sentences.clear()
            class_ = np.zeros((2,batch_size))

    dict_dev = {}
    cnt = 0
    l = 0
    sentences = []
    class_ = np.zeros((2,batch_size))
    class_.astype(np.int16)
    for text in open('./data/dev.txt').read().splitlines(): 
        
        tmp = text.split('|')
        if tmp[0]=='sentiment':
                continue
        class_[int(tmp[0]),cnt] = 1   # get the sentiment
        # words in sentence
        sentence = np.array([word_vec[dict[word]] for word in (tmp[1].lower()).split(' ')])
        l = max(l,len(sentence))
        sentences.append(sentence)
        cnt += 1
        if cnt == batch_size:
            batch = np.zeros((50,batch_size,l))
            for i in range(0,batch_size):
                batch[:,i,:] = np.pad(sentences[i],((0,l-sentences[i].shape[0]),(0,0)),'constant').T
            dict_dev[len(dict_dev)] = (batch,class_)
            l = 0
            cnt = 0
            sentences.clear()
            class_ = np.zeros((2,batch_size))
    
    dict_test = {}
    cnt = 0
    l = 0
    sentences = []
    class_ = np.zeros((2,batch_size))
    class_.astype(np.int16)
    for text in open('./data/test.txt').read().splitlines(): 
        
        tmp = text.split('|')
        if tmp[0]=='sentiment':
                continue
        class_[int(tmp[0]),cnt] = 1   # get the sentiment
        # words in sentence
        sentence = np.array([word_vec[dict[word]] for word in (tmp[1].lower()).split(' ')])
        l = max(l,len(sentence))
        sentences.append(sentence)
        cnt += 1
        if cnt == batch_size:
            batch = np.zeros((50,batch_size,l))
            for i in range(0,batch_size):
                batch[:,i,:] = np.pad(sentences[i],((0,l-sentences[i].shape[0]),(0,0)),'constant').T
            dict_test[len(dict_test)] = (batch,class_)
            l = 0
            cnt = 0
            sentences.clear()
            class_ = np.zeros((2,batch_size))


    np.save('./data/dict_train_batch_{}.npy'.format(batch_size),dict_train)
    np.save('./data/dict_dev_batch_{}.npy'.format(batch_size),dict_dev)
    np.save('./data/dict_test_batch_{}.npy'.format(batch_size),dict_test)
    return dict_train, dict_dev, dict_test

def random_dic(dicts):
    dict_key_ls = list(dicts.keys())
    random.shuffle(dict_key_ls)
    new_dic = {}
    for key in dict_key_ls:
        new_dic[key] = dicts.get(key)
    return new_dic




if __name__ == '__main__':
    batch_size = 16
    dict, word_vec = get_dict_Glove()
    dict_train, dict_dev, dict_test = get_data_dict(dict,word_vec,batch_size)
    # print(dict_dev[0][0][:,0,:].T,dict_dev[50][1][:,:])
    # print(word_vec[dict['it']],word_vec[dict['a']])
    # exit()
    lstm = LSTM(50,50,batch_size,2)


    for i in range(10):
        cnt = 0
        loss = 0
        for sample in random_dic(dict_train).values():
            cnt += 1
            _, _, caches, y_pred = lstm.lstm_forward(sample[0],np.zeros((50,1)))
            loss += crossEntropy(sample[1], y_pred)
            gradients = lstm.lstm_backward(crossEntropy_derivate(sample[1], y_pred),caches)
            lstm.update_parameters(gradients)
            #exit()
            if cnt%100 == 0:
                print('{}/{}_loss:{}'.format(cnt,len(dict_train),loss/(100)))
                loss = 0
        
        acc = 0
        loss = 0
        for id,sample in random_dic(dict_dev).items():
            _, _, caches, y_pred = lstm.lstm_forward(sample[0],np.zeros((50,1)))
            loss += crossEntropy(sample[1], y_pred)
            acc += np.sum(np.argmax(y_pred,axis=0)==np.argmax(sample[1],axis=0))
            # print('sample')
            # print(y_pred,sample[1])
            # print(id,y_pred,acc)
        print('dev_acc:{},loss:{}'.format(acc/(len(dict_dev)*batch_size),loss/len(dict_dev)))
