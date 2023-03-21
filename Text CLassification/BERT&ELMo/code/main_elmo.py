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
from allennlp.modules.elmo import Elmo, batch_to_ids

parser = argparse.ArgumentParser()
parser.add_argument('-epoch', type=int, default=64, help="epoch")
parser.add_argument('-lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('-batch_size', type=int, default=16, help="Batch size")
parser.add_argument('-mode', type=str, default='train', help="Train or test")
parser.add_argument('-dropout', type=float, default=0.5, help="Dropout rate of elmo output")
parser.add_argument('-n_filters', type=int, default=100, help="n_filters")
parser.add_argument('-restore', type=str, default='', help="Restoring model path,ie. CKPT_DIR/checkpoint_latest.pt")
parser.add_argument('-dir', type=str, default='./CKPT_DIR', help="Checkpoint directory")
parser.add_argument('-freeze_emb', type=bool, default=False, help="word_emb requires_grad")
parser.add_argument('-filter_sizes',nargs='+', type=int, default=[3,4,5], help="filter_sizes")
parser.add_argument('-emb_random', type=bool, default=False, help="random_emb or not")
parser.add_argument('-elmo_out', type=int, default=1024, help="hidden of elmo output")
parser.add_argument('-model', type=str, default='lstm', help="classifier")
parser.add_argument('-hidden', type=int, default=300, help="hidden size of lstm")
parser.add_argument('-bidirec', type=bool, default=True, help="bidirection of lstm")
parser.add_argument('-layers', type=int, default=1, help="layers of lstm")
opt = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

# elmo
options_file = "./elmo_small/options.json" # 配置文件
weight_file = "./elmo_small/weights.hdf5" # 权重文件

data_path = './data/'
train_path = data_path + 'train.txt'
test_path = data_path + 'test.txt'
dev_path = data_path + 'dev.txt'
wordvec_path = data_path + 'word_vec.pt'


class hf_DataSet(torch.utils.data.Dataset):
    def __init__(self, dataset):
        self.datas = dataset
        
    def __len__(self):
        return len(self.datas)
    
    def __getitem__(self, index):
        data = self.datas[index]
        label, text = data['label'], data['sentence']

        Y = [0,0]
        Y[label] = 1

        X = text.split(' ')

        return X, Y
    
def collate_fn(batch_data):
    # give a tuple
    X, Y = zip(*batch_data)
    sentences = list(X)  # tuple to list
    padded_sequence = batch_to_ids(sentences) # padding, token to idx
    ctgy = [i for i in Y]
    return padded_sequence, torch.as_tensor(ctgy, dtype=torch.float32)

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

def train(model_elmo, model, iterator, optimizer, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.train()
    model_elmo.eval()
    
    for batch in iterator:
        
        X, Y = batch
        X = X.to(device)
        Y = Y.to(device)

        with torch.no_grad():
            X = model_elmo(X)['elmo_representations'][0]
            # X [batch, len, hidden]

        optimizer.zero_grad()
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


def evaluate(model_elmo, model, iterator, criterion):
    
    epoch_loss = 0
    epoch_acc = 0
    
    model.eval()
    model_elmo.eval()
    
    with torch.no_grad():
    
        for batch in iterator:
            X, Y = batch
            X = X.to(device)
            Y = Y.to(device)
            X = model_elmo(X)['elmo_representations'][0]
            predictions = model(X).squeeze(1)
            
            loss = criterion(predictions, Y)
            
            acc = binary_accuracy(predictions, Y)

            epoch_loss += loss.item()
            epoch_acc += acc.item()
        
    return epoch_loss / len(iterator), epoch_acc / len(iterator)

def test_glue(model_elmo, model, iterator):
    
    model.eval()
    model_elmo.eval()
    
    with torch.no_grad():
    
        f_label = open('./SST-2.tsv','a',encoding='utf-8')
        f_label.write('index\tprediction\n')
        cnt = 0
        for batch in iterator:
            X, Y = batch
            X = X.to(device)
            X = model_elmo(X)['elmo_representations'][0]
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

        dataset = load_dataset("sst2", cache_dir='./.cache/huggingface/datasets')
        test_set = hf_DataSet(dataset['test'])
        #test_set = DataSet(test_path,word2idx)
        test_batch = get_data_loader(test_set,opt.batch_size,is_train=False)

        print("loading time:", time.time() - start_time)

        model_elmo = Elmo(options_file, weight_file, 1, dropout=0).to(device)
        if opt.model == 'cnn':
            model = models.MY_CNN(opt.elmo_out,opt.n_filters,opt.filter_sizes,drop_out=opt.dropout).to(device)
        else:
            model = models.MY_LSTM(opt.elmo_out,opt.hidden,opt.layers,opt.bidirec,drop_out=opt.dropout).to(device)
        if opt.restore == '':
            print('please add -restore!')
            exit()
        else:
            model_dict = torch.load(opt.restore, map_location=device)
            model.load_state_dict(model_dict)
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        start_time = time.time()
            
        #test_loss, test_acc = evaluate(model, test_batch, criterion)
        test_glue(model_elmo, model, test_batch)
        # epoch_mins, epoch_secs = epoch_time(start_time, time.time())
        
        # print(f'Test Time: {epoch_mins}m {epoch_secs}s')
        # print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}%')

    else:
        print("starting load...")
        start_time = time.time()

        dataset = load_dataset("sst2", cache_dir='./.cache/huggingface/datasets')
        train_set = hf_DataSet(dataset['train'])
        dev_set = hf_DataSet(dataset['validation'])

        train_batch = get_data_loader(train_set,opt.batch_size)
        dev_batch = get_data_loader(dev_set,opt.batch_size,is_train=False)

        print("loading time:", time.time() - start_time)

        model_elmo = Elmo(options_file, weight_file, 1, dropout=0).to(device)
        if opt.model == 'cnn':
            model = models.MY_CNN(opt.elmo_out,opt.n_filters,opt.filter_sizes,drop_out=opt.dropout).to(device)
        else:
            model = models.MY_LSTM(opt.elmo_out,opt.hidden,opt.layers,opt.bidirec,drop_out=opt.dropout).to(device)

        if opt.restore != '':
            model_dict = torch.load(opt.restore)
            model.load_state_dict(model_dict)

        optimizer = optim.Adam(model.parameters(),lr=opt.lr)
        criterion = torch.nn.BCEWithLogitsLoss().to(device)
        
        best_valid_loss = float('inf')
        best_valid_acc = float(0)

        for epoch in range(opt.epoch):

            start_time = time.time()
            
            train_loss, train_acc = train(model_elmo, model, train_batch, optimizer, criterion)
            valid_loss, valid_acc = evaluate(model_elmo, model, dev_batch, criterion)

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