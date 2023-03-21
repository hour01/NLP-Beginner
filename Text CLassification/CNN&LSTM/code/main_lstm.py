from linecache import checkcache
import random
import argparse
import gensim
import torch
import time
from torch.nn.utils.rnn import pad_sequence
import models
import torch.optim as optim
import os
from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument('-epoch', type=int, default=100, help="epoch")
parser.add_argument('-lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('-batch_size', type=int, default=64, help="Batch size")
parser.add_argument('-hidden', type=int, default=300, help="hidden size of lstm")
parser.add_argument('-bidirec', type=bool, default=True, help="bidirection of lstm")
parser.add_argument('-layers', type=int, default=1, help="layers of lstm")
parser.add_argument('-mode', type=str, default='train', help="Train or test")
parser.add_argument('-dropout', type=float, default=0.5, help="Dropout rate")
parser.add_argument('-n_filters', type=int, default=100, help="n_filters")
parser.add_argument('-restore', type=str, default='', help="Restoring model path,ie. CKPT_DIR/checkpoint_latest.pt")
parser.add_argument('-dir', type=str, default='./CKPT_DIR', help="Checkpoint directory")
parser.add_argument('-freeze_emb', type=bool, default=True, help="word_emb requires_grad")
parser.add_argument('-filter_sizes',nargs='+', type=int, default=[3,4,5], help="filter_sizes")
parser.add_argument('-emb_random', type=bool, default=False, help="random_emb or not")
opt = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)


data_path = './data/'
train_path = data_path + 'train.txt'
test_path = data_path + 'test.txt'
dev_path = data_path + 'dev.txt'
wordvec_path = data_path + 'word_vec.pt'

class DataSet(torch.utils.data.Dataset):

    def __init__(self, data_path, word2idx):
        
        self.datas = open(data_path,encoding='utf-8').read().splitlines()[1:]
    
        # 导入字典
        self.word2idx = word2idx
        self.vocab_size = len(self.word2idx)
        

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        data = self.datas[index].split('|')
        category, text = data[0], data[1]

        Y = [0,0]
        Y[int(category)] = 1
        
        X = [self.word2idx[word.lower()] for word in text.split(' ')]

        return X, Y

class hf_DataSet(torch.utils.data.Dataset):
    def __init__(self, dataset, word2idx):
        self.datas = dataset
        # 导入字典
        self.word2idx = word2idx
        self.vocab_size = len(self.word2idx)
        

    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        data = self.datas[index]
        category, text = data['label'], data['sentence']

        Y = [0,0]
        Y[category] = 1

        tmp = text.split(' ')
        # try:
        #     X = [self.word2idx[word] for word in tmp[0:len(tmp)-1]]
        # except:
        #     print(tmp)
        #     exit()
        X = [self.word2idx[word] for word in tmp[0:len(tmp)-1]]

        return X, Y
    
def get_dict_wordvec():
    dict = {}
    lines = open(data_path+'dict_glue_sst2.txt',encoding='utf-8').readlines()
    for line in lines:
        dict[(line.split('|'))[0]] = len(dict)
    revdict = {y:x for x,y in dict.items()}

    if opt.emb_random:
        word_vec = torch.randn([len(dict), 300])
        word_vec[0,:] = 0  # padding 
        return dict,revdict,word_vec

    if os.path.exists(wordvec_path):
        word_vec = torch.load(wordvec_path)
    else:
        word_vec = get_GoogleNews_wordvec(dict)
    return dict,revdict,word_vec

def get_GoogleNews_wordvec(dict):
    emb_model = gensim.models.KeyedVectors.load_word2vec_format(data_path+'GoogleNews-vectors-negative300.bin',binary= True)
    word_vec = torch.randn([len(dict), 300])# 没在google字典中的随机初始化
    for word,id in dict.items():
        if word in emb_model:
            vector = emb_model[word]
            word_vec[id,:] = torch.from_numpy(vector)
        # else:
        #     print(word+' not in emb')
    word_vec[0,:] = 0  # padding 
    torch.save(word_vec,wordvec_path)
    return word_vec

def get_Glove(dict):
    with open(data_path+'glove.6B.100d.txt','rb') as f:  # for glove embedding
        lines=f.readlines()
    # 用GloVe创建词典
    trained_dict={}
    n=len(lines)
    for i in range(n):
        line=lines[i].split()
        trained_dict[line[0].decode("utf-8").lower()]=[float(line[j]) for j in range(1,101)]

    word_vec = torch.randn([len(dict), 100])# 没在google字典中的随机初始化
    for word,id in dict.items():
        if word in trained_dict:
            vector = trained_dict[word]
            word_vec[id,:] = torch.tensor(vector)
        # else:
        #     print(word+' not in emb')
    word_vec[0,:] = 0  # padding 
    torch.save(word_vec,wordvec_path)
    return word_vec

def collate_fn(batch_data):
    sentence, emotion = zip(*batch_data)
    sentences = [torch.tensor(sent) for sent in sentence]  # 把句子变成Longtensor类型
    padded_sents = pad_sequence(sentences, batch_first=True, padding_value=0)  # 自动padding
    ctgy = [i for i in emotion]
    return torch.as_tensor(padded_sents, dtype=torch.int64), torch.as_tensor(ctgy, dtype=torch.float32)

def get_data_loader(dataset, batch_size, is_train = True):
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=is_train, collate_fn=collate_fn)

def binary_accuracy(preds, y):
    """
    Returns accuracy per batch
    """
    predict_label = torch.argmax(preds,dim=1)
    correct = (predict_label == torch.argmax(y,dim=1)).float() #convert into float for division 
    acc = correct.sum() / len(correct)
    return acc

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def train(model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    
    for batch in iterator:
        
        optimizer.zero_grad()
        X, Y = batch
        X = X.to(device)
        Y = Y.to(device)
        predictions = model(X).squeeze(1)
        
        #print(Y.shape,predictions.shape)
        #torch.Size([64]) torch.Size([64, 2])
        loss = criterion(predictions, Y)
        
        acc = binary_accuracy(predictions, Y)
        
        loss.backward()
        
        optimizer.step()
        
        epoch_loss += loss.item()
        epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)


def evaluate(model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
            X, Y = batch
            X = X.to(device)
            Y = Y.to(device)
            predictions = model(X).squeeze(1)
            
            loss = criterion(predictions, Y)
            
            acc = binary_accuracy(predictions, Y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def test_glue(model, iterator):
    
    model.eval()
    
    with torch.no_grad():
    
        f_label = open('./SST-2.tsv','a',encoding='utf-8')
        f_label.write('index\tprediction\n')
        cnt = 0
        for batch in iterator:
            X, Y = batch
            X = X.to(device)
            predictions = model(X).squeeze(1)
            predict_label = torch.argmax(predictions,dim=1)
            for label in predict_label.tolist():
                f_label.write(str(cnt)+'\t'+str(label)+'\n')
                cnt += 1
        
    print('total test:{}\n'.format(cnt))


if __name__ == '__main__':
    if opt.mode == 'test':
        print("starting load...")
        start_time = time.time()

        word2idx,_,word_vec = get_dict_wordvec()
        dataset = load_dataset("sst2", cache_dir='./.cache/huggingface/datasets')
        test_set = hf_DataSet(dataset['test'],word2idx)
        #test_set = DataSet(test_path,word2idx)
        test_batch = get_data_loader(test_set,opt.batch_size,is_train=False)

        print("loading time:", time.time() - start_time)
        
        model = models.MY_LSTM_o(word_vec,opt.hidden,opt.freeze_emb,opt.layers,opt.bidirec,drop_out=opt.dropout).to(device)
        if opt.restore == '':
            print('please add -restore!')
            exit()
        else:
            model_dict = torch.load(opt.restore, map_location=device)
            model.load_state_dict(model_dict)
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        start_time = time.time()
            
        #test_loss, test_acc = evaluate(model, test_batch, criterion)
        test_glue(model, test_batch)
        # epoch_mins, epoch_secs = epoch_time(start_time, time.time())
        
        # print(f'Test Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')

    else:
        print("starting load...")
        start_time = time.time()

        word2idx,_,word_vec = get_dict_wordvec()
        # train_set = DataSet(train_path,word2idx)
        # dev_set = DataSet(dev_path,word2idx)
        dataset = load_dataset("sst2", cache_dir='./.cache/huggingface/datasets')
        train_set = hf_DataSet(dataset['train'],word2idx)
        dev_set = hf_DataSet(dataset['validation'],word2idx)
        
        train_batch = get_data_loader(train_set,opt.batch_size)
        dev_batch = get_data_loader(dev_set,opt.batch_size,is_train=False)

        print("loading time:", time.time() - start_time)

        model = models.MY_LSTM_o(word_vec,opt.hidden,opt.freeze_emb,opt.layers,opt.bidirec,drop_out=opt.dropout).to(device)
        if opt.restore != '':
            model_dict = torch.load(opt.restore)
            model.load_state_dict(model_dict)
        optimizer = optim.Adam(model.parameters(),lr=opt.lr)
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        
        best_valid_loss = float('inf')
        best_valid_acc = float(0)

        for epoch in range(opt.epoch):

            start_time = time.time()
            
            train_loss, train_acc = train(model, train_batch, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model, dev_batch, criterion)

            epoch_mins, epoch_secs = epoch_time(start_time, time.time())
            
            if valid_loss < best_valid_loss:
                best_valid_loss = valid_loss
                torch.save(model.state_dict(), opt.dir+'/ckpt_best_loss.pt')
            if valid_acc > best_valid_acc:
                best_valid_acc = valid_acc
                torch.save(model.state_dict(), opt.dir+'/ckpt_best_acc.pt')
            torch.save(model.state_dict(), opt.dir+'/ckpt_latest_batch.pt')
            
            print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'\tTrain Loss: {train_loss:.3f} | Train Acc: {train_acc*100:.2f}%')
            print(f'\t Val. Loss: {valid_loss:.3f} |  Val. Acc: {valid_acc*100:.2f}%')