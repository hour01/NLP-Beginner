import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
import time
import os
from utils import *
from config import config
import logging
import numpy as np
import models 
from torch.utils.data import DataLoader
from metric import *
from tqdm import tqdm

config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(3407)

def inference(data_loader, model, config):
    with torch.no_grad():
        model.eval()
        true_tags = []
        pred_tags = []
        sent_data = []
        dev_losses = 0.0
        for idx, batch in enumerate(tqdm(data_loader)):  
            words, labels, masks, lens = batch
            sent_data.extend([[config.vocab.id2word.get(idx.item()) for i, idx in enumerate(indices) if mask[i] > 0]
                            for (mask, indices) in zip(masks, words)])
            words = words.to(config.device)
            labels = labels.to(config.device)
            masks = masks.to(config.device)
            y_pred,dev_loss = model(words, masks, labels)  
            labels_pred = model.crf.decode(y_pred, mask=masks)
            targets = [itag[:ilen] for itag, ilen in zip(labels.cpu().numpy(), lens)]
            true_tags.extend([[config.vocab.id2label.get(idx) for idx in indices] for indices in targets])      
            pred_tags.extend([[config.vocab.id2label.get(idx) for idx in indices] for indices in labels_pred])  
            dev_losses += dev_loss.item()
        assert len(pred_tags) == len(true_tags)
        assert len(sent_data) == len(true_tags)

        # logging loss, f1 and report
        metrics = {}      
        f1, p, r = f1_score(true_tags, pred_tags)  
        metrics['f1'] = f1      
        metrics['p'] = p        
        metrics['r'] = r        
        metrics['loss'] = float(dev_losses) / len(data_loader)
        if config.mode == 'test':
            bad_case(sent_data, pred_tags, true_tags)
            output_write(sent_data, pred_tags)
        return metrics

def train(config, model, train_set, val_set):
    optimizer = optim.Adam(model.parameters(), lr=config.lr, betas=config.betas) 
    scheduler = StepLR(optimizer, step_size=config.lr_steps, gamma=config.lr_gamma) 

    train_loader = DataLoader(train_set, batch_size=config.batch_size,     
                              shuffle=True, collate_fn=train_set.collate_fn)
    dev_loader = DataLoader(val_set, batch_size=config.batch_size,         
                            shuffle=False, collate_fn=val_set.collate_fn)
    
    best_valid_f1 = 0.0

    for epoch in range(config.epoch):
        start_time = time.time()
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            model.train()   
            model.zero_grad()

            x, y, mask, lens = batch
            x = x.to(config.device)
            y = y.to(config.device)
            mask = mask.to(config.device)

            tag_scores, loss = model(x, mask, y)

            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()
        scheduler.step()
        epoch_mins,epoch_secs = epoch_time(start_time, time.time())
        logging.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logging.info('TrainLoss: {:8.4f}'.format(epoch_loss/len(train_loader)))

        metric = inference(dev_loader, model, config)
        val_f1 = metric['f1']       # 当前模型参数的效率值
        dev_loss = metric['loss']
        logging.info('DevLoss: {:8.4f} | F1 score:{:3.3f}'.format(dev_loss, val_f1))
        if(val_f1 > best_valid_f1):
            best_valid_f1 = val_f1
            torch.save(model,config.model_path+'/best_f1.pt')

def test(config, model, test_set):
    test_loader = DataLoader(test_set, batch_size=config.batch_size,     
                              shuffle=False, collate_fn=test_set.collate_fn)
    metric = inference(test_loader,model, config)
    f1 = metric['f1']
    p = metric['p']
    r = metric['r']
    test_loss = metric['loss']
    logging.info("Test loss: {}, f1 score: {}, precision:{}, recall: {}"
                     .format( test_loss, f1, p, r))

def main():
    logging.info("starting load...")
    start_time = time.time()

    # set the logger
    set_logger(config.log_path)
    logging.info("device: {}".format(config.device))

    config.vocab = Vocabulary(config)
    config.vocab.get_vocab()   
    config.word_vec = get_embedding(config)

    model = models.BiLSTM_CRF(config)
    model.to(config.device)

    logging.info("loading time:{}".format(time.time() - start_time))

    if config.mode != 'test':
        if config.restore != '':
            model = torch.load(config.restore)
        word_train, word_dev, label_train, label_dev = dev_split(config.data_dir+config.train_file)
        train_set = SegDataset(word_train, label_train, config.vocab, config.label2id)
        val_set = SegDataset(word_dev, label_dev, config.vocab, config.label2id)
        train(config, model, train_set, val_set)
    else:
        if config.restore != '':
            model = torch.load(config.restore)
        else:
            logging.info('need --restore')
            exit()
        test_data = np.load(config.data_dir+config.test_file, allow_pickle=True)
        test_dataset = SegDataset(test_data['words'], test_data['labels'], config.vocab, config.label2id)
        test(config, model, test_dataset)



if __name__ == '__main__':
    main()
   