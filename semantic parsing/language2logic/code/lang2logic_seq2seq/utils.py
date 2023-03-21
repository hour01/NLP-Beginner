import torch
import random
import numpy as np
from torch.utils.data import Dataset
import pickle as pkl
import tree
from operator import itemgetter

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

def convert_to_string(idx_list, form_manager):
    w_list = []
    for i in range(len(idx_list)):
        w_list.append(form_manager.get_idx_symbol(int(idx_list[i])))
    return " ".join(w_list)

class SymbolsManager():
    def __init__(self, whether_add_special_tags):
        self.symbol2idx = {}
        self.idx2symbol = {}
        self.vocab_size = 0
        self.whether_add_special_tags = whether_add_special_tags
        if whether_add_special_tags:
            # start symbol = 0
            self.add_symbol('<S>')
            # end symbol = 1
            self.add_symbol('<E>')
            # UNK symbol = 2
            self.add_symbol('<U>')
            # pad symbol = 3
            self.add_symbol('<PAD>')

    def add_symbol(self,s):
        if s not in self.symbol2idx:
            self.symbol2idx[s] = self.vocab_size
            self.idx2symbol[self.vocab_size] = s
            self.vocab_size = self.vocab_size + 1
        return self.symbol2idx[s]

    def get_symbol_idx(self, s):
        if s not in self.symbol2idx:
            if self.whether_add_special_tags:
                return self.symbol2idx['<U>']
            else:
                print("this should never be reached (always add <U>")
                return 0
        return self.symbol2idx[s]

    def get_idx_symbol(self, idx):
        if idx not in self.idx2symbol:
            return '<U>'
        return self.idx2symbol[idx]

    def init_from_file(self, fn, min_freq, max_vocab_size):
        print("loading vocabulary file: {}\n".format(fn))
        with open(fn, "r") as f:
            for line in f:
                l_list = line.strip().split('\t')
                c = int(l_list[1])
                if c >= min_freq:
                    self.add_symbol(l_list[0])
                if self.vocab_size >= max_vocab_size:
                    break

    def get_symbol_idx_for_list(self,l):
        r = []
        for i in range(len(l)):
            r.append(self.get_symbol_idx(l[i]))
        return r

class AtisDataset(Dataset):
    def __init__(self, config, data_path):
        super(AtisDataset, self).__init__()
        self.data_path = data_path
        self.data = pkl.load(open(data_path, "rb"))
        self.word_manager = config.word_manager
        self.form_manager = config.form_manager

        self.x = []
        self.y = []
        self.lengths_x = []
        self.lengths_y = []
        self.read_data()

    def read_data(self):
        for sample in self.data:
            x = sample[0]
            x.insert(0,self.word_manager.get_symbol_idx('<S>'))
            x.append(self.word_manager.get_symbol_idx('<E>'))
            y = sample[1]
            y.insert(0,self.form_manager.get_symbol_idx('<S>'))
            y.append(self.form_manager.get_symbol_idx('<E>'))
            self.x.append(x)
            self.y.append(y)
            self.lengths_x.append(len(x))
            self.lengths_y.append(len(y))
        print('{}_size:{}'.format(self.data_path, len(self.x)))

    def __len__(self):
        return len(self.x)

    def __getitem__(self, index):
        return self.x[index], self.y[index], self.lengths_x[index], self.lengths_y[index]

    def collate_fn(self, batch):
        lengths_x = [f[2] for f in batch]
        max_len_x = max(lengths_x)
        lengths_y = [f[3] for f in batch]
        max_len_y = max(lengths_y)
        # padding
        input_ids = [batch[i][0] + [self.word_manager.get_symbol_idx('<PAD>')] * (max_len_x - lengths_x[i]) for i in range(len(batch))]
        symbols = [batch[i][1] + [self.word_manager.get_symbol_idx('<PAD>')] * (max_len_y - lengths_y[i]) for i in range(len(batch))]
        return input_ids, symbols, lengths_x

def is_all_same(c1, c2):
    if len(c1) == len(c2):
        all_same = True
        for j in range(len(c1)):
            if c1[j] != c2[j]:
                all_same = False
                break
        return all_same
    else:
        return False

def compute_accuracy(candidate_list, reference_list):

    c = 0
    for i in range(len(candidate_list)):
        if is_all_same(candidate_list[i], reference_list[i]):
            c = c+1
    return c/float(len(candidate_list))

if __name__ == '__main__':
    pass