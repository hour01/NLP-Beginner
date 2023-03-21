from utils import *
import argparse
import torch
import time
from torch.utils.data import DataLoader
import models
import torch.optim as optim
import os
import preprocess

parser = argparse.ArgumentParser()

# Training configs
parser.add_argument('--epoch', type=int, default=20, help="epoch")
parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
parser.add_argument('--batch_size', type=int, default=16, help="Batch size")
parser.add_argument('--eval_steps', type=int, default=400, help='Total number of training epochs to perform.')
parser.add_argument('--mode', type=str, default='train', help="Train or test")
parser.add_argument('--restore', type=str, default='', help="Restoring model path,ie. CKPT_DIR/checkpoint_latest.pt")

# Model configs
parser.add_argument('--emb_dim', type=int, default=100, help='Tokens will be embedded to a vector.')
parser.add_argument('--hidden_dim', type=int, default=100, help="hidden size of lstm")
parser.add_argument('--layers', type=int, default=1, help="layers of lstm")
parser.add_argument('--dropout', type=float, default=0.5, help="Dropout rate")
parser.add_argument('--freeze_emb', type=bool, default=False, help="freeze the embedding")
parser.add_argument('--bert', type=bool, default=False, help="to use bert-crf")
parser.add_argument('--freeze_bert', type=bool, default=False, help="freeze bert")

# path and data configs
parser.add_argument('--data_path', default='./data/conll03', help='The dataset path.', type=str)
parser.add_argument('--model_path', default='./CKPT', help='The model will be saved to this path.', type=str)
parser.add_argument('--train_file', default='eng.train', type=str)
parser.add_argument('--val_file', default='eng.testa', type=str)
parser.add_argument('--test_file', default='eng.testb', type=str)
parser.add_argument('--wordvec_file', default='word_vec.pt', type=str)
parser.add_argument('--googlewordvec_path', default='./data/GoogleNews-vectors-negative300.bin', type=str)
parser.add_argument('--glove_path', default='./data/glove.6B.300d.txt', type=str)

opt = parser.parse_args()
opt.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
opt.tag2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8, 'START':9, 'END':10}
opt.id2tag = {y:x for x,y in opt.tag2id.items()}
opt.pretrain_path='./bert-base-uncased'
set_seed(3407)


def train(config, model, train_set, val_set):

    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    train_dataloader = DataLoader(train_set, batch_size=config.batch_size,
                                  collate_fn=train_set.collate_fn, shuffle=True, drop_last=True )
    val_dataloader = DataLoader(val_set, batch_size=config.batch_size, collate_fn=val_set.collate_fn,drop_last=True)

    best_valid_loss = float('inf')
    best_valid_F1 = float(0)
    

    for epoch in range(config.epoch):

        start_time = time.time()
        epoch_loss = 0
        metric = Metric(config.id2word, config.id2tag)

        for step, batch in enumerate(train_dataloader):
            model.zero_grad()
            model.train()   

            if opt.bert:
                input_ids, labels, lengths, input_mask, aligned_labels, offset_map = batch

                input_ids = torch.tensor(input_ids).to(config.device)
                # labels = torch.tensor(labels).to(config.device)
                aligned_labels = torch.tensor(aligned_labels).to(config.device)
                input_mask = torch.tensor(input_mask).to(config.device)
                
                loss, tag_seq = model(input_ids, aligned_labels, lengths, input_mask)
                idx_list, tags_list = integrate(tag_seq,offset_map)
                assert len(tags_list[0]) == len(labels[0])
                for i in range(len(labels)):
                    metric.add(idx_list[i],
                            tags_list[i],
                            labels[i])
            else:
                input_ids, labels, _, lengths, input_mask = batch

                input_ids = torch.tensor(input_ids).to(config.device)
                labels = torch.tensor(labels).to(config.device)
                input_mask = torch.tensor(input_mask).to(config.device)
                
                loss, tag_seq = model(input_ids, labels, lengths, input_mask)
            
                for i in range(len(labels)):
                    metric.add(input_ids[i, :lengths[i]].cpu().numpy().tolist(),
                            tag_seq[i],
                            labels[i].cpu().numpy().tolist())

            # Update the parameters of the model
            loss.backward()
            optimizer.step()
            model.zero_grad()

            epoch_loss += loss.item()

            if step % config.eval_steps == 0 and step != 0:
                res = evaluate(config, model, val_dataloader)
                print('Epoch: {:3} | Step: {:6} | ValLoss: {:8.4f} | Val_P: {:5.2f} | Val_R: {:5.2f} | Val_F1: {:5.2f}'.format(
                    epoch + 1, step, res['loss'], res['p'], res['r'], res['f1']))
                
                if best_valid_loss > res['loss']:
                    best_valid_loss = res['loss']
                    torch.save(model.state_dict(), opt.model_path+'/ckpt_best_loss.pt')
                if best_valid_F1 < res['f1']*100:
                    torch.save(model.state_dict(), opt.model_path+'/ckpt_best_F1.pt')

        epoch_mins,epoch_secs = epoch_time(start_time, time.time())
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        p, r, f1 = metric.get()
        print('TrainLoss: {:8.4f} | Train_P: {:5.2f} | Train_R: {:5.2f} | Train_F1: {:5.2f}'.format(
                    epoch_loss/len(train_dataloader), p * 100, r * 100, f1 * 100))
        torch.save(model.state_dict(), opt.model_path+'/ckpt_latest.pt')


def evaluate(config, model, val_dataloader):
    metric = Metric(config.id2word, config.id2tag)
    model.eval()    # Switch to the evaluate mode
    loss = 0
    for batch in val_dataloader:

        if opt.bert:
            input_ids, labels, lengths, input_mask, aligned_labels, offset_map = batch

            input_ids = torch.tensor(input_ids).to(config.device)
            # labels = torch.tensor(labels).to(config.device)
            aligned_labels = torch.tensor(aligned_labels).to(config.device)
            input_mask = torch.tensor(input_mask).to(config.device)
            # labels_flatten = torch.tensor(labels_flatten).to(config.device)
            
            # loss, pred = model(input_ids, labels_flatten, lengths, input_mask)
            l, tag_seq = model(input_ids, aligned_labels, lengths, input_mask)
            idx_list, tags_list = integrate(tag_seq,offset_map)
            assert len(tags_list[0]) == len(labels[0])
            loss += l
            for i in range(len(labels)):
                metric.add(idx_list[i],
                        tags_list[i],
                        labels[i])
        else:
            input_ids, labels, _, lengths, input_mask = batch

            input_ids = torch.tensor(input_ids).to(config.device)
            labels = torch.tensor(labels).to(config.device)
            input_mask = torch.tensor(input_mask).to(config.device)
            # labels_flatten = torch.tensor(labels_flatten).to(config.device)
            
            # loss, pred = model(input_ids, labels_flatten, lengths, input_mask)
            l, pred = model(input_ids, labels, lengths, input_mask)
            loss += l
            for i in range(len(labels)):
                metric.add(input_ids[i, :lengths[i]].cpu().numpy().tolist(),
                        pred[i],
                        labels[i].cpu().numpy().tolist())

    p, r, f1 = metric.get()
    return {
        'p': p * 100,
        'r': r * 100,
        'f1': f1 * 100,
        'loss':loss/len(val_dataloader)
    }

def test(config, model, test_features):
    test_dataloader = DataLoader(test_features, batch_size=config.batch_size, shuffle=False, collate_fn=test_features.collate_fn,drop_last=False)
    metric = Metric(config.id2word, config.id2tag)
    model.eval()
    loss = 0
    f = open('./output.txt','w',encoding='utf-8')
    for batch in test_dataloader:

        if opt.bert:
            input_ids, labels, lengths, input_mask, aligned_labels, offset_map = batch

            input_ids = torch.tensor(input_ids).to(config.device)
            # labels = torch.tensor(labels).to(config.device)
            aligned_labels = torch.tensor(aligned_labels).to(config.device)
            input_mask = torch.tensor(input_mask).to(config.device)
            # labels_flatten = torch.tensor(labels_flatten).to(config.device)
            
            # loss, pred = model(input_ids, labels_flatten, lengths, input_mask)
            l, tag_seq = model(input_ids, aligned_labels, lengths, input_mask)
            idx_list, tags_list = integrate(tag_seq,offset_map)
            assert len(tags_list[0]) == len(labels[0])
            loss += l
            write_to_file(f,config,idx_list,tags_list,labels)
            for i in range(len(labels)):
                metric.add(idx_list[i],
                        tags_list[i],
                        labels[i])
        else:
            input_ids, labels, _, lengths, input_mask = batch

            input_ids = torch.tensor(input_ids).to(config.device)
            labels = torch.tensor(labels).to(config.device)
            input_mask = torch.tensor(input_mask).to(config.device)
            # labels_flatten = torch.tensor(labels_flatten).to(config.device)
            
            # loss, pred = model(input_ids, labels_flatten, lengths, input_mask)
            l, pred = model(input_ids, labels, lengths, input_mask)
            loss += l
            
            for i in range(len(labels)):
                metric.add(input_ids[i, :lengths[i]].cpu().numpy().tolist(),
                        pred[i],
                        labels[i].cpu().numpy().tolist())
                write_to_file(f,config,[input_ids[i, :lengths[i]].cpu().numpy().tolist()],
                                    [pred[i]],
                                    [labels[i].cpu().numpy().tolist()])

    p, r, f1 = metric.get()
    print('Test :loss:{:8.4f} | P: {:5.2f} | R: {:5.2f} | F1: {:5.2f}'.format(loss/len(test_dataloader),p * 100, r * 100, f1 * 100))


def main():
    print("starting load...")
    start_time = time.time()

    word2id, id2word, word_cnt, word_vec = preprocess.get_dict_wordvec(opt)

    opt.vocab_size = word_cnt
    opt.word_vec = word_vec
    opt.word2id = word2id
    opt.id2word = id2word
    if opt.bert:
        train_set = preprocess.Conll03Dataset_bert(opt, os.path.join(opt.data_path, opt.train_file))
        val_set = preprocess.Conll03Dataset_bert(opt, os.path.join(opt.data_path, opt.val_file))
        test_set = preprocess.Conll03Dataset_bert(opt, os.path.join(opt.data_path, opt.test_file))
        # model = models.Bert_CRF(opt).to(opt.device)
        model = models.Bert_BiLSTM_CRF(opt).to(opt.device)
        print('Bert-CRF')
    else:
        train_set = preprocess.Conll03Dataset(opt, os.path.join(opt.data_path, opt.train_file))
        val_set = preprocess.Conll03Dataset(opt, os.path.join(opt.data_path, opt.val_file))
        test_set = preprocess.Conll03Dataset(opt, os.path.join(opt.data_path, opt.test_file))
        model = models.BiLSTM_CRF(opt).to(opt.device)
        print('BiLSTM-CRF')



    print("loading time:", time.time() - start_time)

    if opt.mode != 'test':
        if opt.restore != '':
            model.load_state_dict(torch.load(opt.restore, map_location=opt.device))
        train(opt, model, train_set, val_set)
    else:
        if opt.restore != '':
            model.load_state_dict(torch.load(opt.restore, map_location=opt.device))
            print(model.crf.transitions)
        else:
            print('need --restore')
            exit()
        test(opt, model, test_set)



if __name__ == '__main__':
    main()
    # train_set = preprocess.Conll03Dataset_bert(opt, os.path.join(opt.data_path, 'minidataset.txt'))
    # # for i in range(10):
    # #     print(train_set.__getitem__(i))
    # train_dataloader = DataLoader(train_set, batch_size=2,
    #                               collate_fn=train_set.collate_fn, shuffle=False)
    # for batch in train_dataloader:
    #     print(batch)
    #     print(len(batch[3]),len(batch[0]))
    #     print(batch[0])
    #     print(batch[4])
    #     print(batch[3])
    #     exit()