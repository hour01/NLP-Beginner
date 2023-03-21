import torch
import random
import numpy as np
from torch.utils.data import Dataset
from .dataio import read_conll, UNK, EMPTY_FE
from .conll09 import lock_dicts, post_train_lock_dicts, VOCDICT, POSDICT, LEMDICT, LUDICT, LUPOSDICT, FEDICT
from .housekeeping import filter_long_ex

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

def combine_examples(corpus_ex):
    """
    Target ID needs to be trained for all targets in the sentence jointly, as opposed to
    frame and arg ID. Returns all target annotations for a given sentence.
    """
    combined_ex = [corpus_ex[0]]
    for ex in corpus_ex[1:]:
        if ex.sent_num == combined_ex[-1].sent_num:
            current_sent = combined_ex.pop()
            # conbine the targetframedict
            target_frame_dict = current_sent.targetframedict.copy()
            target_frame_dict.update(ex.targetframedict)
            current_sent.targetframedict = target_frame_dict
            combined_ex.append(current_sent)
            continue
        combined_ex.append(ex)
    # sys.stderr.write("Combined {} instances in data into {} instances.\n".format(
    #     len(corpus_ex), len(combined_ex)))
    return combined_ex

def unk_replace_tokens(tokens, replaced, vocdict, unkprob, unktoken):
    """
    replaces singleton tokens in the train set with UNK with a probability UNK_PROB
    :param tokens: original token IDs
    :param replaced: replaced token IDs
    :return:
    """
    for t in tokens:
        if vocdict.is_singleton(t) and random.random() < unkprob:
            replaced.append(unktoken)
        else:
            replaced.append(t)

def get_wvec(path):
    with open(path,'rb') as f:  # for glove embedding
        lines=f.readlines()
    # 用GloVe创建词典
    wd_vecs={}
    n=len(lines)
    for i in range(n):
        line=lines[i].split()
        wd_vecs[line[0].decode("utf-8").lower()]=[float(line[j]) for j in range(1,101)]

    # wvf = open(path, 'r')
    # wvf.readline()
    # wd_vecs = {line.split(' ')[0].decode("utf-8").lower() :
    #             [float(f) for f in line.strip().split(' ')[1:]] for line in wvf}
    
    word_vec = torch.randn([VOCDICT.size(), 100])
    for word,id in VOCDICT._strtoint.items():
        if word.lower() in wd_vecs:
            word_vec[id,:] = torch.tensor(wd_vecs[word.lower()])
    return word_vec

class FNTargetIDDataset(Dataset):
    def __init__(self, config, data_path, train=True):
        super(FNTargetIDDataset, self).__init__()
        self.data_path = data_path

        examples, _, _ = read_conll(data_path)
        self.combined_examples = combine_examples(examples)
        self.config = config
        self.train = train

    def __len__(self):
        return len(self.combined_examples)

    def __getitem__(self, index):
        return self.combined_examples[index]

    def collate_fn(self, batch):
        '''
        batch size must be 1.
        '''
        assert len(batch) == 1
        example = batch[0]
        inputtokens = []
        unk_replace_tokens(example.tokens, inputtokens, VOCDICT, self.config.configuration['unk_prob'], VOCDICT.getid(UNK))
        labels = [0]*len(inputtokens)
        for idx in example.targetframedict.keys():
            labels[idx] = 1

        return inputtokens if self.train else example.tokens, example.postags,  example.lemmas, labels

class FNFrameIDDataset(Dataset):
    def __init__(self, config, data_path, train=True):
        super(FNFrameIDDataset, self).__init__()
        self.data_path = data_path

        examples, _, _ = read_conll(data_path)
        self.examples = examples
        self.config = config
        self.train = train

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def collate_fn(self, batch):
        '''
        batch size must be 1.
        '''
        assert len(batch) == 1
        example = batch[0]

        inputtokens = []
        unk_replace_tokens(example.tokens, inputtokens, VOCDICT, self.config.configuration['unk_prob'], VOCDICT.getid(UNK))

        # labels = [0]*len(inputtokens)
        # for idx in example.targetframedict.keys():
        #     labels[idx] = 1

        return inputtokens if self.train else example.tokens, example.postags,\
               example.lu.id, example.lu.posid, list(example.targetframedict.keys()), example.frame.id


class FNArgumentIDDataset(Dataset):
    def __init__(self, config, data_path, train=True):
        super(FNArgumentIDDataset, self).__init__()
        self.data_path = data_path

        examples, _, _ = read_conll(data_path)
        self.examples = filter_long_ex(examples, config.configuration['use_span_clip'],\
                                        config.configuration["allowed_max_span_length"], FEDICT.getid(EMPTY_FE))
        self.config = config
        self.train = train

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        return self.examples[index]

    def collate_fn(self, batch):
        '''
        batch size must be 1.
        '''
        assert len(batch) == 1
        example = batch[0]

        inputtokens = []
        unk_replace_tokens(example.tokens, inputtokens, VOCDICT, self.config.configuration['unk_prob'], VOCDICT.getid(UNK))

        target_position = sorted(list(example.targetframedict.keys()))
        relative_position = [i - target_position[0] for i in range(len(inputtokens))]


        return inputtokens if self.train else example.tokens, example.postags,\
               example.lu.id, example.lu.posid, target_position, relative_position, example.frame.id, example.invertedfes