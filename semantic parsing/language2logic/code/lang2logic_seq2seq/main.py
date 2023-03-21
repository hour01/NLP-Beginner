from utils import *
import argparse
import torch
import torch.nn as nn
import time
from torch.utils.data import DataLoader
import model
import torch.optim as optim
import torch.nn.init as init
import os
from config import config
import pickle as pkl
import utils

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(3407)


def generate(config, encoder, decoder, dataloader):
    '''
    the batch of dataloader should be 1.
    '''
    encoder.eval()    # Switch to the evaluate mode
    decoder.eval()

    reference_list = []
    candidate_list = []

    ref_str_list = []
    cand_str_list = []

    for batch in dataloader:

        input_ids, symblos, lengths = batch
        input_ids = torch.tensor(input_ids).to(config.device)
        symblos = torch.tensor(symblos).to(config.device)

        assert input_ids.shape[0] == 1

        enc_len = input_ids.size(1)

        # initialize the rnn state to all zeros
        prev_c  = torch.zeros((1, encoder.hidden_size), requires_grad=False).to(config.device)
        prev_h  = torch.zeros((1, encoder.hidden_size), requires_grad=False).to(config.device)

        for i in range(enc_len):
            prev_h, prev_c = encoder(input_ids[:,i], prev_h, prev_c)
        
        prev_word = torch.tensor([config.form_manager.get_symbol_idx('<S>')], dtype=torch.long).to(config.device)
        
        # <S>
        text_gen = [0]
        while True:
            pred, prev_h, prev_c = decoder(prev_word,prev_h, prev_c)
            # log probabilities from the previous timestamp
            form_id = pred.argmax().item()
            prev_word = torch.tensor([form_id],dtype=torch.long).to(config.device)
            text_gen.append(form_id)
            if (form_id == config.form_manager.get_symbol_idx('<E>')) or (len(text_gen) >= config.dec_seq_length):
                break
        
        # to list
        symblos_list = [int(c) for c in symblos[0]]
        ref_str = convert_to_string(symblos_list, config.form_manager)
        cand_str = convert_to_string(text_gen, config.form_manager)

        reference_list.append(symblos_list)
        candidate_list.append(text_gen)
        # print to console
        if config.display > 0:
            print("results: ")
            print(ref_str)
            print(cand_str)
            print(' ')

    if config.output != '':
        with open(config.output) as f:
            for i in range(len(ref_str_list)):
                f.write('{}\n'.format(ref_str_list[i]))
                f.write('{}\n'.format(cand_str_list[i]))
                f.write('\n')

    val_acc = compute_accuracy(candidate_list, reference_list)
    return val_acc*100

def train(config, encoder, decoder, train_set, val_set):

    # optimizer = optim.Adam(model.parameters(), lr=config.lr)
    lr_d = config.lr
    encoder_optimizer = optim.RMSprop(encoder.parameters(),  lr=lr_d, alpha=config.decay_rate)
    decoder_optimizer = optim.RMSprop(decoder.parameters(),  lr=lr_d, alpha=config.decay_rate)
    criterion = nn.NLLLoss(reduction='sum', ignore_index=0)

    train_dataloader = DataLoader(train_set, batch_size=config.batch_size,collate_fn=train_set.collate_fn, shuffle=True, drop_last=True, )
    val_dataloader = DataLoader(val_set, batch_size=1, collate_fn=val_set.collate_fn)

    best_valid_acc = float('inf')
    
    for epoch in range(config.epoch):

        start_time = time.time()
        epoch_loss = 0

        for step, batch in enumerate(train_dataloader):
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            encoder.train()   
            decoder.train()   

            #[batch, len]
            input_ids, symblos, lengths = batch

            input_ids = torch.tensor(input_ids).to(config.device)
            symblos = torch.tensor(symblos).to(config.device)

            enc_max_len = input_ids.size(1)
            # do not predict after <E>
            dec_max_len = symblos.size(1) -1

            c = torch.zeros((config.batch_size, config.hidden_size), dtype=torch.float, requires_grad=True).to(config.device)
            h = torch.zeros((config.batch_size, config.hidden_size), dtype=torch.float, requires_grad=True).to(config.device)

            for i in range(enc_max_len):
                h, c = encoder(input_ids[:, i], h, c)
            
            loss = 0
            for i in range(dec_max_len):
                pred, h, c = decoder(symblos[:, i], h, c)
                loss += criterion(pred, symblos[:, i+1])

            # Update the parameters of the model
            loss = loss / config.batch_size
            loss.backward()
            torch.nn.utils.clip_grad_value_(encoder.parameters(),config.grad_clip)
            torch.nn.utils.clip_grad_value_(decoder.parameters(),config.grad_clip)
            encoder_optimizer.step()
            decoder_optimizer.step()

            epoch_loss += loss.item()

        res = generate(config, encoder, decoder, val_dataloader)
        
        if best_valid_acc > res:
            best_valid_acc = res
            checkpoint = {}
            checkpoint["encoder"] = encoder
            checkpoint["decoder"] = decoder
            torch.save(checkpoint, config.model_path+'/ckpt_best_acc.pt')
        

        epoch_mins,epoch_secs = epoch_time(start_time, time.time())
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print('TrainLoss: {:8.4f} | Acc: {:8.4f}'.format(epoch_loss/len(train_dataloader), res))

        #exponential learning rate decay
        if config.learning_rate_decay < 1:
            if epoch >= config.learning_rate_decay_after:
                decay_factor = config.learning_rate_decay
                lr_d = lr_d * decay_factor #decay it
                for param_group in encoder_optimizer.param_groups:
                    param_group['lr'] = lr_d
                for param_group in decoder_optimizer.param_groups:
                    param_group['lr'] = lr_d

def test(config, encoder, decoder, test_set):

    test_dataloader = DataLoader(test_set, batch_size=1, collate_fn=test_set.collate_fn)
    acc = generate(config, encoder,decoder, test_dataloader)
    print('Test Acc:{}'.format(acc))

def main():
    print("starting load...")
    start_time = time.time()

    managers = pkl.load(open("{}/map.pkl".format(config.data_dir), "rb" ))
    config.word_manager, config.form_manager = managers

    train_set = utils.AtisDataset(config, '{}/{}'.format(config.data_dir, config.train_file))
    val_set = utils.AtisDataset(config, '{}/{}'.format(config.data_dir, config.val_file))
    test_set = utils.AtisDataset(config, '{}/{}'.format(config.data_dir, config.test_file))
    
    # model
    encoder = model.Encoder(config).to(config.device)
    decoder = model.Decoder(config).to(config.device)
    # init parameters
    for name, param in encoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -config.init_weight, config.init_weight)
    for name, param in decoder.named_parameters():
        if param.requires_grad:
            init.uniform_(param, -config.init_weight, config.init_weight)

    print("loading time:", time.time() - start_time)

    if config.mode != 'test':
        if config.restore != '':
            checkpoint = torch.load(config.restore)
            encoder = checkpoint["encoder"]
            decoder = checkpoint["decoder"]
        train(config, encoder, decoder, train_set, val_set)
    else:
        if config.restore != '':
            checkpoint = torch.load(config.restore)
            encoder = checkpoint["encoder"]
            decoder = checkpoint["decoder"]
        else:
            print('need --restore')
            exit()
        test(config, encoder, decoder, test_set)



if __name__ == '__main__':
    main()