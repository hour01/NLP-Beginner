import time
import pickle as pkl
import torch
from utils import SymbolsManager
from sys import path
import argparse
import random
import numpy as np

def process_train_data(opt):
    # retrieve the vacab
    time_start = time.time()
    word_manager = SymbolsManager(True)
    word_manager.init_from_file("{}/vocab.q.txt".format(opt.data_dir), opt.min_freq, opt.max_vocab_size)
    print(word_manager.vocab_size)
    form_manager = SymbolsManager(True)
    form_manager.init_from_file("{}/vocab.f.txt".format(opt.data_dir), 0, opt.max_vocab_size)
    print(form_manager.vocab_size)

    data = []
    with open("{}/{}.txt".format(opt.data_dir, opt.train), "r") as f:
        for line in f:
            l_list = line.split("\t")
            w_list = word_manager.get_symbol_idx_for_list(l_list[0].strip().split(' '))
            r_list = form_manager.get_symbol_idx_for_list(l_list[1].strip().split(' '))
            data.append((w_list, r_list))
    out_mapfile = "{}/map.pkl".format(opt.processed_data_dir)
    out_datafile = "{}/train.pkl".format(opt.processed_data_dir)
    with open(out_mapfile, "wb") as out_map:
        pkl.dump([word_manager, form_manager], out_map)
    with open(out_datafile, "wb") as out_data:
        pkl.dump(data, out_data)

def process_dev_test_data(opt):
    data = []
    managers = pkl.load( open("{}/map.pkl".format(opt.processed_data_dir), "rb" ) )
    word_manager, form_manager = managers
    for fn in [opt.test, opt.dev]:
        with open("{}/{}.txt".format(opt.data_dir, fn), "r") as f:
            for line in f:
                l_list = line.split("\t")
                w_list = word_manager.get_symbol_idx_for_list(l_list[0].strip().split(' '))
                r_list = form_manager.get_symbol_idx_for_list(l_list[1].strip().split(' '))
                data.append((w_list, r_list))
        out_datafile = "{}/{}.pkl".format(opt.processed_data_dir, fn)
        with open(out_datafile, "wb") as out_data:
            pkl.dump(data, out_data)



parser = argparse.ArgumentParser()
parser.add_argument("-data_dir", type=str, default="./atis",help="data dir")
parser.add_argument("-processed_data_dir", type=str, default="./processed_data",help="processed data dir")
parser.add_argument("-train", type=str, default="train",help="train dir")
parser.add_argument("-dev", type=str, default="dev",help="dev dir")
parser.add_argument("-test", type=str, default="test",help="test dir")
parser.add_argument("-min_freq", type=int, default=0,help="minimum word frequency")
parser.add_argument("-max_vocab_size", type=int, default=15000,help="max vocab size")

config = parser.parse_args()
process_train_data(config)
process_dev_test_data(config)
