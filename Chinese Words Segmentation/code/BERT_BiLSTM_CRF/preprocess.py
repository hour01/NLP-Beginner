import os
import logging
import numpy as np

DATA_DIR = './pku_data/pku_data/'
TRAIN = DATA_DIR+'training.txt'
TEST = DATA_DIR+'test.txt'
PROCESSED_DIR = './processed_data/'
PROCESSED_train = PROCESSED_DIR+'train_dev.npz'
PROCESSED_test = PROCESSED_DIR+'test.npz'

MAX_LEN = 500
sep_word = '@' 
sep_label = 'S' 

def getlables(input_str):
    """
    input_str: the string of phrase
    output_str: list of labels to this phrase,['B','M','E']
    """
    output_str = []
    if len(input_str) == 1:
        output_str.append('S')
    elif len(input_str) == 2:
        output_str = ['B', 'E']
    else:
        M_num = len(input_str) - 2
        M_list = ['M'] * M_num
        output_str.append('B')
        output_str.extend(M_list)
        output_str.append('E')
    return output_str

def get_sub_list(init_list, sublist_len, sep_word):
    '''
    split sequence with sublist_len
    '''
    list_groups = zip(*(iter(init_list),) * sublist_len)
    end_list = [list(i) + list(sep_word) for i in list_groups]
    count = len(init_list) % sublist_len
    if count != 0:
        end_list.append(init_list[-count:])
    else:
        end_list[-1] = end_list[-1][:-1]  # remove the last sep word
    return end_list

def preprocess(path, out_path):
    '''
    read original file from path
    write the processed data to out_path
    Label the raw data with BMES
    '''
    with open(path, 'r', encoding='utf-8') as f:     
        word_list = []
        label_list = []
        num = 0
        sep_num = 0
        for line in f:
            words = []
            line = line.strip()  # remove spaces at the beginning and the end
            if not line:
                continue  # line is None
            for i in range(len(line)):
                if line[i] == " ":
                    continue  # skip space
                words.append(line[i])
            text = line.split(" ")
            labels = []
            for item in text:
                if item == "":
                    continue
                labels.extend(getlables(item))
            if len(words) > MAX_LEN:
                # split the sentence with MAX_LEN
                sub_word_list = get_sub_list(words, MAX_LEN - 5, sep_word)
                sub_label_list = get_sub_list(labels, MAX_LEN - 5, sep_label)
                word_list.extend(sub_word_list)
                label_list.extend(sub_label_list)
                sep_num += 1
            else:
                word_list.append(words)
                label_list.append(labels)
            num += 1
            assert len(labels) == len(words), "labels != words"
            assert len(word_list) == len(label_list), "word_list != label_list "
        print("We have", num, "lines in", path, "file processed")
        print("We have", sep_num, "lines in", path, "file get sep processed")
        # saved as a binary_file
        np.savez_compressed(out_path, words=word_list, labels=label_list)


if __name__ == '__main__':
    # preprocess(TRAIN,PROCESSED_train)
    preprocess(TEST,PROCESSED_test)