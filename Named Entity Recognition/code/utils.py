import torch
import random
import numpy as np

# Set the random seed to make the training result consistent
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_sentandtag(input_ids, tags, labels, config):
    with open('./predict_sample.txt','a',encoding='utf-8') as f:
        f.write('{}\n{}\n{}\n\n'.format([config.id2word[id] for id in input_ids],
                    [config.id2tag[id] for id in tags],[config.id2tag[id] for id in labels]))

# Compute log sum exp in a numerically stable way for the forward algorithm
def log_sum_exp(x):
    """calculate log(sum(exp(x))) = max(x) + log(sum(exp(x - max(x))))
    """
    max_score = x.max(-1)[0]
    return max_score + (x - max_score.unsqueeze(-1)).exp().sum(-1).log()

def integrate(tags,offset_map):
    '''
    integrate tokens to a whole word
    tags:(B,L) with no padding, L is actual length of sentence.
    offset_map: the idx is still the idx of ids([CLS],...,[SEP],padding).
    '''
    idx_list = []
    tags_list = []
    # batch
    for i in range(len(tags)):
        ofst = offset_map[i]
        sentence_tags = []
        j = 0 # tags idx
        # ofst contains the tokens of a whole word,[[1,2,3],[5,6],[8,9,10]]
        for m in ofst:
            while j<m[0]-1:
                sentence_tags.append(tags[i][j])
                j+=1
            labels = [tags[i][idx-1] for idx in m]
            j+=len(m)
            # select the most frequent in tokens ot represent the word
            sentence_tags.append(max(labels,key=labels.count))
        while j < len(tags[i]):
            sentence_tags.append(tags[i][j])
            j+=1
        tags_list.append(sentence_tags)
        idx_list.append(list(range(len(sentence_tags))))
    return idx_list, tags_list

def write_to_file(file, config, word, pred, label):
    '''
    write pred to file
    test code conlleval
    expected input:
        file:python writable file
        word:list of [batch,sentence_words], ids of word
        pred:list of [batch,tags of sentence]
        label: ths same as pred
    '''
    for i in range(len(word)):
        for j in range(len(word[i])):
            file.write(str(word[i][j]) +' '+config.id2tag[label[i][j]]+' '+config.id2tag[pred[i][j]]+'\n')
        # split each sentence.
        file.write('\n')


# Compute the precision, recall and F1 score
class Metric:
    def __init__(self, id2word, id2tag):
        self.gold_num = 0
        self.pred_num = 0
        self.correct = 0
        self.id2word = id2word
        self.id2tag = id2tag

    def add(self, sent, pred, gold):
        '''
        input is a sample(with no batch)
        '''
        pred_entity = self.get_entity(sent, pred)
        gold_entity = self.get_entity(sent, gold)
        self.gold_num += len(gold_entity)
        self.pred_num += len(pred_entity)
        self.correct += len([item for item in pred_entity if item in gold_entity])

    def get_entity(self, sent, tag):
        '''
        get the entity-list from the sent
        entity: continuous sequence like (B-type_1,.., I-type_1), (I-type,.., I-type)
        the previous one must immediately followed by a entity
        '''
        res = []
        # cache a phrase of continuous words
        entity = []
        for j in range(len(sent)):
            # <PAD> or tag 'O' or the last word in this sentence
            if sent[j] == 0 or tag[j] == 0 or tag[j] == 10 or tag[j] == 9:
                continue
            # (B-type,..,I-type) case:find a 'B-type' tag
            # previous one must not be 'O' and the 'tyep' in previous (I-type) has to equal to 'type' in (B-type)
            if self.id2tag[tag[j]][0] == 'B' and j >0 \
                and self.id2tag[tag[j-1]][1:]==self.id2tag[tag[j]][1:] and len(entity) == 0:
                entity = []
                entity = [str(sent[j]) + '|' + self.id2tag[tag[j]]]
                # case (B-type), this is the last word or the next tag is not I-type
                if j == len(sent) - 1 or self.id2tag[tag[j + 1]][0] != 'I' :
                    res.append(entity)
                    entity = []
            # (B-type,..,I-type) case:'I-type' tag follwed by the 'B-type' tag  
            elif self.id2tag[tag[j]][0] == 'I' and len(entity) != 0 \
                    and entity[0].split('|')[1][0] == 'B'\
                    and entity[0].split('|')[1][1:] == self.id2tag[tag[j]][1:]:
                # add a new word to the cache of (B-type,..,I-type) case
                entity.append(str(sent[j]) + '|' + self.id2tag[tag[j]])
                # this is the last word in sentence, or the next one donesn't equal to this type
                if j == len(sent) - 1 or tag[j + 1] != tag[j]:
                    res.append(entity)
                    entity = []
            # (I-type,.., I-type) case, the begin of (I-type,.., I-type) case
            # the previous one must be 'O', (issue: (I-type1), (I-type2) is allowed)
            elif self.id2tag[tag[j]][0] == 'I' and len(entity)==0\
                and (j==0 or tag[j-1] != tag[j]):   # issue:and (j==0 or tag[j-1] != 0):
                entity.append(str(sent[j]) + '|' + self.id2tag[tag[j]])
                if j == len(sent) - 1 or tag[j + 1] != tag[j]:
                    res.append(entity)
                    entity = []
            # (I-type,.., I-type) case
            elif self.id2tag[tag[j]][0] == 'I' and len(entity) != 0 \
                and entity[0].split('|')[1] == self.id2tag[tag[j]]:
                # add a new word to the cache of (I-type,..,I-type) case
                entity.append(str(sent[j]) + '|' + self.id2tag[tag[j]])
                # this is the last word in sentence, or the next one donesn't equal to this type
                if j == len(sent) - 1 or tag[j + 1] != tag[j]:
                    res.append(entity)
                    entity = []
            else:
                entity = []
        return res

    def get(self):
        if self.pred_num == 0 or self.gold_num == 0:
            return 0, 0, 0
        p = self.correct / self.pred_num
        r = self.correct / self.gold_num
        if p + r == 0:
            return 0, 0, 0
        f1 = 2*p*r / (p + r)
        return p, r, f1


if __name__ == '__main__':
    id2word = ['pad','example']
    tag2id = {'O': 0, 'B-PER': 1, 'I-PER': 2, 'B-ORG': 3, 'I-ORG': 4, 'B-LOC': 5, 'I-LOC': 6, 'B-MISC': 7, 'I-MISC': 8}
    id2tag = {y:x for x,y in tag2id.items()}
    m = Metric(id2word,id2tag)
    sent = [1,1,1,1,1,1,1]
    tag =  [0,2,0,4,1,2,2]
    print(m.get_entity(sent,tag))

    sent = [1,1,1,1,1,1,1]
    tag =  [1,2,0,4,6,6,1]
    print(m.get_entity(sent,tag))

    sent = [1,1,1,1,1,1,1]
    tag =  [3,0,6,6,6,6,5]
    print(m.get_entity(sent,tag))

    sent = [1,1,1,1,1,1,1]
    tag =  [8,8,8,6,3,4,1]
    print(m.get_entity(sent,tag))

    sent = [1,1,1,1,1,1,1]
    tag =  [8,8,8,6,3,4,1]
    print(m.get_entity(sent,tag))

    tensor = torch.randn(3,3)
    print(tensor)
    print(torch.max(tensor,axis=1,keepdim=True)[0])

    # tags = [[0,0,1,1,3,5,6,5,0],[1,2,3,5,6,6,7,8]]
    # offset_map = [[[3,4],[8,9]],[[4,5,6]]]
    # print(integrate(tags,offset_map))