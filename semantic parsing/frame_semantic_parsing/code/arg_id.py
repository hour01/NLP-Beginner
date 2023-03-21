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
from utils.dataio import create_target_lu_map, get_wvec_map, read_conll, read_related_lus, read_frame_maps
from utils.evaluation import calc_f, evaluate_example_targetid, evaluate_example_argid
from utils.frame_semantic_graph import LexicalUnit
from utils.housekeeping import unk_replace_tokens, Factor
from utils.raw_data import make_data_instance
import math


config.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(3407)

def cost(factor, goldfactors):
    '''
    compute the cost(s, s*) in paper
    '''
    alpha = 2
    beta = 1

    if factor in goldfactors:
        return 0
    i = factor.begin
    j = factor.end
    alphabetacost = 0.0
    # false positive
    if factor.label != FEDICT.getid(EMPTY_FE):
        alphabetacost += beta
    # false negtive
    # find number of good gold factors it kicks out
    for gf in goldfactors:
        if i <= gf.begin <= j and gf.label != FEDICT.getid(EMPTY_FE):
            alphabetacost += alpha

    return torch.tensor(alphabetacost,requires_grad=True,device=config.device)#.to(config.device)

def get_loss(factexprs_dict, gold_fes, valid_fes, sentlen):

    goldfactors = [Factor(span[0], span[1], feid) for feid in gold_fes for span in gold_fes[feid]]
    numeratorexprs = torch.tensor([factexprs_dict[gf] for gf in goldfactors],requires_grad=True)
    numerator = torch.sum(numeratorexprs)

    # compute logZ in the paper
    logalpha = [None for _ in range(sentlen)]
    for j in range(sentlen):
        # full length spans
        spanscores = []
        if not config.configuration['use_span_clip'] or j < config.configuration["allowed_max_span_length"]:
            spanscores = [factexprs_dict[Factor(0, j, y)]
                          + cost(Factor(0, j, y), goldfactors) for y in valid_fes]

        # recursive case
        istart = 0
        if config.configuration['use_span_clip'] and j > config.configuration["allowed_max_span_length"]:
            istart = max(0, j - config.configuration["allowed_max_span_length"])
        # from istart to j
        for i in range(istart, j):
            # print(i+1,j)
            facscores = [logalpha[i]
                         + factexprs_dict[Factor(i + 1, j, y)]
                         + cost(Factor(i + 1, j, y), goldfactors) for y in valid_fes]
            spanscores.extend(facscores)
        logalpha[j] = torch.logsumexp(torch.tensor(spanscores,requires_grad=True),dim=0)
    
    loss = logalpha[sentlen - 1] - numerator

    if loss.item() < 0:
        raise Exception("negative probability! probably overcounting spans?")

    return loss

def decode(factexprscalars, sentlen, valid_fes):
    alpha = [None for _ in range(sentlen)]
    backpointers = [None for _ in range(sentlen)]

    # [a,b], where a==0 and b from 0 to j-1
    for j in range(sentlen):
        if config.configuration['use_span_clip'] and j >= config.configuration["allowed_max_span_length"]: continue
        bestscore = float("-inf")
        bestlabel = None
        for y in valid_fes:
            fac = Factor(0, j, y)
            facscore = math.exp(factexprscalars[fac])
            if facscore > bestscore:
                bestscore = facscore
                bestlabel = y
        alpha[j] = bestscore
        backpointers[j] = (0, bestlabel)

    for j in range(sentlen):
        bestscore = float("-inf")
        bestbeg = bestlabel = None
        if alpha[j] is not None:
            bestscore = alpha[j]
            bestbeg, bestlabel = backpointers[j]

        istart = 0
        if config.configuration['use_span_clip'] and j > config.configuration["allowed_max_span_length"]:
            istart = max(0, j - config.configuration["allowed_max_span_length"])
        for i in range(istart, j):
            for y in valid_fes:
                fac = Factor(i + 1, j, y)
                facscore = math.exp(factexprscalars[fac])
                if facscore * alpha[i] > bestscore:
                    bestscore = facscore * alpha[i]
                    bestlabel = y
                    bestbeg = i + 1
        alpha[j] = bestscore
        backpointers[j] = (bestbeg, bestlabel)

    # get possible spans of each frame element
    # argmax[frame_element_id] = [spans:(begin, end)]
    j = sentlen - 1
    i = backpointers[j][0]
    argmax = {}
    while i >= 0:
        fe = backpointers[j][1]
        if fe in argmax:
            argmax[fe].append((i, j))
        else:
            argmax[fe] = [(i, j)]
        if i == 0:
            break
        j = i - 1
        i = backpointers[j][0]

    # merging neighboring spans in prediction (to combat spurious ambiguity)
    mergedargmax = {}
    for fe in argmax:
        mergedargmax[fe] = []
        if fe == FEDICT.getid(EMPTY_FE):
            mergedargmax[fe].extend(argmax[fe])
            continue

        argmax[fe].sort()
        mergedspans = [argmax[fe][0]]
        for span in argmax[fe][1:]:
            prevsp = mergedspans[-1]
            if span[0] == prevsp[1] + 1:
                prevsp = mergedspans.pop()
                mergedspans.append((prevsp[0], span[1]))
            else:
                mergedspans.append(span)
        mergedargmax[fe] = mergedspans
    return mergedargmax

def inference(config, model, dataloader:DataLoader, frmfemap, corefrmfemap):
    eval_loss = 0
    ures = labldres = tokenwise = [0.0, 0.0, 0.0]
    for step, batch in enumerate(dataloader):
        if(step%100==0):
            print('step:{}/{}'.format(step,len(dataloader)))
        model.eval()
        #[batch, len]
        input_tokens, postags, lu_id, lu_pos, target_position, relative_position, frame_id, gold_fes = batch

        input_tokens = torch.tensor(input_tokens).to(config.device)
        postags = torch.tensor(postags).to(config.device)
        lu_id = torch.tensor(lu_id).to(config.device)
        lu_pos = torch.tensor(lu_pos).to(config.device)
        target_position = torch.tensor(target_position).to(config.device)
        relative_position = torch.tensor(relative_position).to(config.device)
        valid_fes = frmfemap[frame_id] + [FEDICT.getid(EMPTY_FE)]
        frame_id = torch.tensor(frame_id).to(config.device)

        factexprs_dict = model(input_tokens, postags, relative_position, lu_id, lu_pos, frame_id, target_position, valid_fes)
        
        loss = get_loss(factexprs_dict, gold_fes, valid_fes, input_tokens.shape[0])
        eval_loss += loss.item()

        facexprscalars = {fact: factexprs_dict[fact].item() for fact in factexprs_dict}
        argmax = decode(facexprscalars, len(input_tokens), valid_fes)

        if frame_id in corefrmfemap:
            corefes = corefrmfemap[frame_id]
        else:
            corefes = {}
        u, l, t = evaluate_example_argid(gold_fes, argmax, corefes, len(input_tokens), FEDICT.getid(EMPTY_FE))
        ures = np.add(ures, u)
        labldres = np.add(labldres, l)
        tokenwise = np.add(tokenwise, t)

    lp, lr, lf = calc_f(labldres)
    eval_loss /= len(dataloader)

    return eval_loss, lf

def train(config, model, train_set, val_set, frmfemap, corefrmfemap):

    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    
    assert config.batch_size == 1
    train_dataloader = DataLoader(train_set, batch_size=config.batch_size,collate_fn=train_set.collate_fn, shuffle=True, drop_last=True, )
    val_dataloader = DataLoader(val_set, batch_size=1, collate_fn=val_set.collate_fn)

    best_valid_loss = float('inf')
    
    for epoch in range(config.epoch):

        start_time = time.time()
        epoch_loss = 0
        for step, batch in enumerate(train_dataloader):
            if(step%100==0):
                print('step:{}/{}'.format(step,len(train_set)))
            optimizer.zero_grad()
            model.train()   

            input_tokens, postags, lu_id, lu_pos, target_position, relative_position, frame_id, gold_fes = batch

            input_tokens = torch.tensor(input_tokens).to(config.device)
            postags = torch.tensor(postags).to(config.device)
            lu_id = torch.tensor(lu_id).to(config.device)
            lu_pos = torch.tensor(lu_pos).to(config.device)
            target_position = torch.tensor(target_position).to(config.device)
            relative_position = torch.tensor(relative_position).to(config.device)
            valid_fes = frmfemap[frame_id] + [FEDICT.getid(EMPTY_FE)]
            frame_id = torch.tensor(frame_id).to(config.device)

            factexprs_dict = model(input_tokens, postags, relative_position, lu_id, lu_pos, frame_id, target_position, valid_fes)
            
            loss = get_loss(factexprs_dict, gold_fes, valid_fes, input_tokens.shape[0])

            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

        validloss, f1 = inference(config, model, val_dataloader, frmfemap, corefrmfemap)
        if best_valid_loss > validloss:
            best_valid_loss = validloss
            torch.save(model, config.model_path+'/AI_ckpt_best_loss.pt')
        
        epoch_mins,epoch_secs = epoch_time(start_time, time.time())
        print(f'Epoch: {epoch+1:02} | Epoch Time: {epoch_mins}m {epoch_secs}s')
        print('TrainLoss: {:8.4f} | ValidLoss:{:8.4f} | Acc: {:8.4f}'.format(epoch_loss/len(train_dataloader), validloss, f1))

def test(config, model, test_set, frmfemap, corefrmfemap):
    test_dataloader = DataLoader(test_set, batch_size=1, collate_fn=test_set.collate_fn)
    loss, f1 = inference(config, model, test_dataloader, frmfemap, corefrmfemap)
    print('TestLoss: {:8.4f} | F1: {:8.4f}'.format(loss, f1))

def main():
    print("starting load...")
    start_time = time.time()

    train_set = FNArgumentIDDataset(config, '{}/{}'.format(config.data_dir, config.train_file))
    post_train_lock_dicts()
    frmfemap, corefrmfemap, _ = read_frame_maps()
    lock_dicts()

    val_set = FNArgumentIDDataset(config, '{}/{}'.format(config.data_dir, config.val_file),train=False)

    word_vec = get_wvec('{}/{}'.format(config.data_path, config.word_vec_file))

    # model
    model = models.SegRNN(config, word_vec).to(config.device)

    print("loading time:", time.time() - start_time)

    if config.mode == 'train':
        if config.restore != '':
            model = torch.load(config.restore)
        train(config, model, train_set, val_set, frmfemap, corefrmfemap)
    else:
        if config.restore != '':
            model = torch.load(config.restore)
        else:
            print('need --restore')
            exit()
        test_set = FNArgumentIDDataset(config, '{}/{}'.format(config.data_dir, config.test_file),train=False)
        test(config, model, test_set, frmfemap, corefrmfemap)



if __name__ == '__main__':
    main()