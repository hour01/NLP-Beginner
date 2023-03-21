from utils.util import *
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
import models
import torch.optim as optim
import torch.nn.init as init
import os
from config_AI import config
import torch.nn.functional as F

from utils.conll09 import lock_dicts, post_train_lock_dicts, VOCDICT, POSDICT, LEMDICT, LUDICT, LUPOSDICT
from utils.dataio import create_target_lu_map, get_wvec_map, read_conll, read_related_lus
from utils.evaluation import calc_f, evaluate_example_targetid
from utils.frame_semantic_graph import LexicalUnit
from utils.housekeeping import unk_replace_tokens
from utils.raw_data import make_data_instance


config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(3407)

def inference(config, model, dataloader:DataLoader, lufrmmap):
    eval_loss = 0
    correct = 0
    total = 0
    for step, batch in enumerate(dataloader):
        model.eval()
        #[batch, len]
        input_tokens, postags, lu_id, lu_pos, target_position, frame_id = batch

        input_tokens = torch.tensor(input_tokens).to(config.device)
        postags = torch.tensor(postags).to(config.device)
        lu_id = torch.tensor(lu_id).to(config.device)
        lu_pos = torch.tensor(lu_pos).to(config.device)
        valid_frames = torch.tensor(list(lufrmmap[lu_id.item()])).to(config.device)
        target_position = torch.tensor(target_position).to(config.device)

        if len(valid_frames) > 1:

            scores = model(input_tokens, postags, lu_id, lu_pos, target_position)
            scores = torch.index_select(scores, 1, valid_frames)
            loss = -F.log_softmax(scores,dim=1).squeeze(0)[torch.nonzero(valid_frames==frame_id).item()]

            eval_loss += loss.item()
            pred = torch.max(scores,dim=1)[1].item()
            if pred == torch.nonzero(valid_frames==frame_id).item():
                correct += 1
            total += 1

    eval_loss /= total
    acc = correct/total

    return eval_loss, acc*100



def train(config, model, train_set, val_set, lufrmmap):

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    # lr_d = config.lr
    
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
            input_tokens, postags, lu_id, lu_pos, target_position, frame_id = batch

            input_tokens = torch.tensor(input_tokens).to(config.device)
            postags = torch.tensor(postags).to(config.device)
            lu_id = torch.tensor(lu_id).to(config.device)
            lu_pos = torch.tensor(lu_pos).to(config.device)
            valid_frames = torch.tensor(list(lufrmmap[lu_id.item()])).to(config.device)
            target_position = torch.tensor(target_position).to(config.device)
            
            if len(valid_frames) > 1:

                scores = model(input_tokens, postags, lu_id, lu_pos, target_position)
                scores = torch.index_select(scores, 1, valid_frames)
                loss = -F.log_softmax(scores,dim=1).squeeze(0)[torch.nonzero(valid_frames==frame_id).item()]

                # Update the parameters of the model
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

        validloss, acc = inference(config, model, val_dataloader, lufrmmap)
        if best_valid_loss > validloss:
            best_valid_loss = validloss
            torch.save(model, config.model_path+'/FI_ckpt_best_loss.pt')
        
        epoch_mins,epoch_secs = epoch_time(start_time, time.time())
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print('TrainLoss: {:8.4f} | ValidLoss:{:8.4f} | Acc: {:8.4f}'.format(epoch_loss/len(train_dataloader), validloss, acc))

def test(config, model, test_set, lufrmmap):
    test_dataloader = DataLoader(test_set, batch_size=1, collate_fn=test_set.collate_fn)
    loss, acc = inference(config, model, test_dataloader, lufrmmap)
    print('TestLoss: {:8.4f} | Acc: {:8.4f}'.format(loss, acc))

def main():
    print("starting load...")
    start_time = time.time()

    train_set = FNFrameIDDataset(config, '{}/{}'.format(config.data_dir, config.train_file))
    post_train_lock_dicts()

    lufrmmap, _ = read_related_lus()
    lock_dicts()

    val_set = FNFrameIDDataset(config, '{}/{}'.format(config.data_dir, config.val_file),train=False)

    word_vec = get_wvec('{}/{}'.format(config.data_path, config.word_vec_file))
    
    # model
    model = models.FIBiLSTM(config, word_vec).to(config.device)

    print("loading time:", time.time() - start_time)

    if config.mode == 'train':
        if config.restore != '':
            model = torch.load(config.restore)
        train(config, model, train_set, val_set, lufrmmap)
    else:
        if config.restore != '':
            model = torch.load(config.restore)
        else:
            print('need --restore')
            exit()
        test_set = FNFrameIDDataset(config, '{}/{}'.format(config.data_dir, config.test_file),train=False)
        test(config, model, test_set, lufrmmap)



if __name__ == '__main__':
    main()