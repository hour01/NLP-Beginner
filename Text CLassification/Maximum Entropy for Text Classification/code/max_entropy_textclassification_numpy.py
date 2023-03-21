import math
import numpy as np
import plot
# hyper-parameters#########
epoch = 20
n_gram = 1
############################
ctgyNum = 2

def feature_extraction():
    index = 0
    for text in train_set: 
        tmp = text.split('|')
        if tmp[0]=='sentiment':
            continue
        ctgy = int(tmp[0])    # get the sentiment
        category[index] = ctgy #每个文本对应的类别序号

        word_count_text = {}    #每句话的表示 word-count in sentence: word - count
        sum = 0    # 这句话总词数(ngram)

        # words in sentence
        words = (tmp[1].lower()).split(' ')
        #print(words,tmp[1])
        for i in range(0,len(words)):
            n_word = ''

            for k in range(0,n_gram):
                if i+k>=len(words):
                    break

                if k==0:
                    n_word = words[i]
                else:
                    n_word = n_word +' '+words[i+k]
                
                #print(n_word)

                if(n_word in word_count_text):
                    word_count_text[n_word] += 1
                else:
                    word_count_text[n_word] = 1
                sum += 1
        for word,count in word_count_text.items():

            EP_prior[words_dict[word]][ctgy] += count/sum
            feature[index][words_dict[word]] = count/sum

        index+=1


def updateWeight():
    global feature_weight
    EP_post = np.zeros((wordNum,ctgyNum))


    # prob是 文本数*类别 的矩阵，记录每个文本属于每个类别的概率
    cond_prob_textNum_ctgyNum = np.zeros((train_size,ctgyNum))#[[0.0 for x in range(ctgyNum)] for y in range(train_size)]
    
    #计算p(类别|文本)
    for i in range(train_size):#对每一个文本
        zw = 0.0  #归一化因子
        for j in range(ctgyNum):#对每一个类别
            tmp = 0.0

            # 本该是两个二维矩阵点乘，但除了选中的类别，其他都是0
            tmp = math.exp(np.vdot(feature[i],feature_weight[j])) 
            
            zw+=tmp #zw关于类别求和
            cond_prob_textNum_ctgyNum[i][j]=tmp   # 写入分子                     
        for j in range(ctgyNum):
            cond_prob_textNum_ctgyNum[i][j]/=zw
        
    #上面的部分根据当前的feature_weight矩阵计算得到prob矩阵（文本数*类别的矩阵，每个元素表示文本属于某类别的概率），
    #下面的部分根据prob矩阵更新feature_weight矩阵。
    
    for x in range(train_size):
        EP_post += feature[x].reshape((wordNum,1)).repeat(ctgyNum,axis = 1) * np.array([cond_prob_textNum_ctgyNum[x][0],cond_prob_textNum_ctgyNum[x][1]]).reshape((1,2)).repeat(wordNum,axis=0)
    
    tmp = np.log((EP_prior/EP_post).transpose())   
    tmp[tmp == np.inf] = 0
    tmp[tmp == -np.inf] = 0
    tmp[np.isnan(tmp)] = 0
    feature_weight += tmp  
      
def modelTest():

    errorCnt = 0
    corrCnt = 0
    totalCnt = 0
    for line in test_set: #对每个句子
        feature_test = np.zeros((1,wordNum))
        tmp = line.split('|')
        if tmp[0]=='sentiment':
            continue
        ctgy = int(tmp[0])    # get the sentiment
        sum = 0    # 这句话总词数(ngram)
        # words in sentence
        words = (tmp[1].lower()).split(' ')

        

        for i in range(0,len(words)):
            n_word = ''

            for k in range(0,n_gram):
                if i+k>=len(words):
                    break

                if k==0:
                    n_word = words[i]
                else:
                    n_word = n_word +' '+words[i+k]

                feature_test[0,words_dict[n_word]] += 1
                sum += 1

        feature_test /= sum

        pro = (feature_test.repeat(ctgyNum,axis = 0) * feature_weight).sum(axis=1)
        ctgyEst = pro.argmax()
        totalCnt+=1
        if ctgyEst!=ctgy: 
            errorCnt+=1
        else:
            corrCnt += 1
    print ("测试总文本个数:"+str(totalCnt)+"  总错误个数:"+str(errorCnt)+"  总正确个数:"+str(corrCnt)+"  总正确率:"+str(corrCnt/float(totalCnt)))
    return corrCnt/float(totalCnt)

def train():
    accuracy_hitstory = []
    for i in range(epoch):
        feature_extraction()
        print ("迭代{}次后的模型效果：".format(i+1))
        updateWeight()
        accuracy_hitstory.append(modelTest())
    plot.plot_accuracy(accuracy_hitstory,name = 'test_accuracy_{}-gram'.format(n_gram))


print('initializing')
# get the dictionary
words_dict = {}
lines = open('processed_data_{}-gram/words_dict.txt'.format(n_gram)).readlines()
for line in lines:
    words_dict[(line.split('|'))[0]] = len(words_dict)   

# get the train set
train_set = open('processed_data_{}-gram/train.txt'.format(n_gram)).read().splitlines()
test_set = open('processed_data_{}-gram/test.txt'.format(n_gram)).read().splitlines()

# parrameters
wordNum = len(words_dict)
train_size = len(train_set)
#feature_weight: matrix of dict_words*categories
feature_weight = np.zeros((ctgyNum,wordNum))
EP_prior = np.zeros((wordNum,ctgyNum))

# the whole featrue matrix
feature = np.zeros((train_size,wordNum))
texts_list_dict = [{}]*train_size # dict of word frequency for each sentence in train_set
category = [0]*train_size        # catogory of each sentence in train set   
train()
