import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from transformers.optimization import get_cosine_schedule_with_warmup, AdamW
import time
import os
from utils import *
from config import config
import logging
import numpy as np
import models 
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from metric import *
from tqdm import tqdm
from torch.nn.parallel import DistributedDataParallel

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
            batch_data, batch_token_starts, batch_tags, ori_data = batch
            # shift tensors to GPU if available
            batch_data = batch_data.to(config.device)
            batch_token_starts = batch_token_starts.to(config.device)
            batch_tags = batch_tags.to(config.device)
            sent_data.extend(ori_data)
            batch_masks = batch_data.gt(0)  # get padding mask
            label_masks = batch_tags.gt(-1)

            loss, pred = model((batch_data, batch_token_starts),
                         token_type_ids=None, attention_mask=batch_masks, labels=batch_tags)
            dev_losses += loss.item()   
            labels_pred = model.crf.decode(pred, mask=label_masks)

            batch_tags = batch_tags.to('cpu').numpy()
            pred_tags.extend([[config.vocab.id2label.get(idx) for idx in indices] for indices in labels_pred])
            # (batch_size, max_len - padding_label_len)
            true_tags.extend([[config.vocab.id2label.get(idx) for idx in indices if idx > -1] for indices in batch_tags])
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
    optimizer = AdamW(config.optimizer_grouped_parameters, lr=config.lr, correct_bias=False)
    train_steps_per_epoch = len(train_set) // config.batch_size
    scheduler = get_cosine_schedule_with_warmup(optimizer,
                                                num_warmup_steps=2 * train_steps_per_epoch,
                                                num_training_steps=config.epoch * train_steps_per_epoch)

    train_loader = DataLoader(train_set, batch_size=config.batch_size,     
                              shuffle=True, collate_fn=train_set.collate_fn,
                              )
    dev_loader = DataLoader(val_set, batch_size=config.batch_size,         
                            shuffle=False, collate_fn=val_set.collate_fn,
                            )
    
    best_valid_f1 = 0.0

    for epoch in range(config.epoch):
        start_time = time.time()
        epoch_loss = 0.0
        for step, batch in enumerate(tqdm(train_loader)):
            optimizer.zero_grad()
            model.train()   
            model.zero_grad()

            batch_data, batch_token_starts, batch_labels, _ = batch
            batch_data = batch_data.to(config.device)
            batch_token_starts = batch_token_starts.to(config.device)
            batch_labels = batch_labels.to(config.device)
            batch_masks = batch_data.gt(0)  # get padding mask
            # compute model output and loss
            loss,_ = model((batch_data, batch_token_starts),
                        token_type_ids=None, attention_mask=batch_masks, labels=batch_labels)

            loss.backward()
            # gradient clipping
            torch.nn.utils.clip_grad_norm_(parameters=model.parameters(), max_norm=config.clip_grad)
            # performs updates using calculated gradients
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
        scheduler.step()
        epoch_mins,epoch_secs = epoch_time(start_time, time.time())
        logging.info(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        logging.info('TrainLoss: {:8.4f}'.format(epoch_loss/len(train_loader)))

        metric = inference(dev_loader, model, config)
        val_f1 = metric['f1']       # 当前模型参数的效率值
        dev_loss = metric['loss']
        logging.info('DevLoss: {:8.4f} | F1 score:{:8.4f}'.format(dev_loss, val_f1))
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
    set_logger(config.log_dir)
    logging.info("device: {}".format(config.device))

    config.vocab = Vocabulary(config)
    config.vocab.get_vocab() 

    model = models.Bert_BiLSTM_CRF(config)
    model.to(config.device)

     # Prepare optimizer
    if config.fine_tune == 'full':
        # model.named_parameters(): [bert, classifier, crf]
        bert_optimizer = list(model.bert.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        config.optimizer_grouped_parameters = [
            {'params': [p for n, p in bert_optimizer if not any(nd in n for nd in no_decay)],
             'weight_decay': config.weight_decay},
            {'params': [p for n, p in bert_optimizer if any(nd in n for nd in no_decay)],
             'weight_decay': 0.0},
            {'params': model.classifier.parameters(), 'lr': config.lr * 100},
            {'params': model.bilstm.parameters(), 'lr': config.lr * 100},
            {'params': model.crf.parameters(), 'lr': config.lr * 100}
        ]
    # only fine-tune the head classifier
    else:
        config.optimizer_grouped_parameters = [
            {'params': model.classifier.parameters(), 'lr': config.lr * 100},
            {'params': model.bilstm.parameters(), 'lr': config.lr * 100},
            {'params': model.crf.parameters(), 'lr': config.lr * 100}
        ]

    if config.mode != 'test':
        if config.restore != '':
            model = torch.load(config.restore)
        word_train, word_dev, label_train, label_dev = dev_split(config.data_dir+config.train_file)
        train_set = SegDataset(word_train, label_train, config)
        val_set = SegDataset(word_dev, label_dev, config)
        logging.info("loading time:{}".format(time.time() - start_time))
        train(config, model, train_set, val_set)
    else:
        if config.restore != '':
            model = torch.load(config.restore)
        else:
            logging.info('need --restore')
            exit()
        test_data = np.load(config.data_dir+config.test_file, allow_pickle=True)
        test_dataset = SegDataset(test_data['words'], test_data['labels'], config)
        logging.info("loading time:{}".format(time.time() - start_time))
        test(config, model, test_dataset)



if __name__ == '__main__':
    main()