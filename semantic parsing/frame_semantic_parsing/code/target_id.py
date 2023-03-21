from utils.util import *
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
import models
import torch.optim as optim
import torch.nn.init as init
import os
from config_TI import config

from utils.conll09 import lock_dicts, post_train_lock_dicts, VOCDICT, POSDICT, LEMDICT, LUDICT, LUPOSDICT
from utils.dataio import create_target_lu_map, get_wvec_map, read_conll
from utils.evaluation import calc_f, evaluate_example_targetid
from utils.frame_semantic_graph import LexicalUnit
from utils.housekeeping import unk_replace_tokens
from utils.raw_data import make_data_instance


config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(3407)

def inference(config, model, dataloader:DataLoader, criterion):
    eval_loss = 0
    acc = 0
    for step, batch in enumerate(dataloader):
        model.eval()
        #[batch, len]
        input_tokens, postags, lemmas, labels = batch

        input_tokens = torch.tensor(input_tokens).to(config.device)
        postags = torch.tensor(postags).to(config.device)
        lemmas = torch.tensor(lemmas).to(config.device)
        labels = torch.tensor(labels).to(config.device)

        scores = model(input_tokens, postags, lemmas)

        loss = criterion(scores, labels)

        acc += (torch.sum(torch.max(scores,dim=1)[1]==labels)/len(labels)).item()
        eval_loss += loss.item()

    eval_loss /= len(dataloader)
    acc /= len(dataloader)

    return eval_loss, acc*100



def train(config, model, train_set, val_set):

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # lr_d = config.lr
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    assert config.batch_size == 1
    train_dataloader = DataLoader(train_set, batch_size=config.batch_size,collate_fn=train_set.collate_fn, shuffle=True, drop_last=True, )
    val_dataloader = DataLoader(val_set, batch_size=1, collate_fn=val_set.collate_fn)

    best_valid_loss = float('inf')
    
    for epoch in range(config.epoch):

        start_time = time.time()
        epoch_loss = 0

        for step, batch in enumerate(train_dataloader):
            optimizer.zero_grad()
            model.train()   

            #[batch, len]
            input_tokens, postags, lemmas, labels = batch

            input_tokens = torch.tensor(input_tokens).to(config.device)
            postags = torch.tensor(postags).to(config.device)
            lemmas = torch.tensor(lemmas).to(config.device)
            labels = torch.tensor(labels).to(config.device)

            scores = model(input_tokens, postags, lemmas)

            loss = criterion(scores, labels)

            # Update the parameters of the model
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        validloss, acc = inference(config, model, val_dataloader, criterion)
        if best_valid_loss > validloss:
            best_valid_loss = validloss
            torch.save(model, config.model_path+'/TI_ckpt_best_loss.pt')
        
        epoch_mins,epoch_secs = epoch_time(start_time, time.time())
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print('TrainLoss: {:8.4f} | ValidLoss:{:8.4f} | Acc: {:8.4f}'.format(epoch_loss/len(train_dataloader), validloss, acc))


def test(config, model, test_set):
    criterion = nn.CrossEntropyLoss(reduction='sum')
    test_dataloader = DataLoader(test_set, batch_size=1, collate_fn=test_set.collate_fn)
    loss, acc = inference(config, model, test_dataloader, criterion)
    print('TestLoss: {:8.4f} | Acc: {:8.4f}'.format(loss, acc))

def main():
    print("starting load...")
    start_time = time.time()

    train_set = FNTargetIDDataset(config, '{}/{}'.format(config.data_dir, config.train_file))
    post_train_lock_dicts()
    lock_dicts()
    val_set = FNTargetIDDataset(config, '{}/{}'.format(config.data_dir, config.val_file),train=False)

    word_vec = get_wvec('{}/{}'.format(config.data_path, config.word_vec_file))
    
    # model
    model = models.TIBiLSTM(config, word_vec).to(config.device)

    print("loading time:", time.time() - start_time)

    if config.mode == 'train':
        if config.restore != '':
            model = torch.load(config.restore)
        train(config, model, train_set, val_set)
    else:
        if config.restore != '':
            model = torch.load(config.restore)
        else:
            print('need --restore')
            exit()
        test_set = FNTargetIDDataset(config, '{}/{}'.format(config.data_dir, config.test_file),train=False)
        test(config, model, test_set)



if __name__ == '__main__':
    main()