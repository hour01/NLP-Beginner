import os
import logging
import numpy as np

DATA_DIR = './pku_data/pku_data/'
TRAIN = DATA_DIR+'training.txt'
TEST = DATA_DIR+'test.txt'
PROCESSED_DIR = './processed_data/'
PROCESSED_train = PROCESSED_DIR+'train_dev.npz'
PROCESSED_test = PROCESSED_DIR+'test.npz'

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
        # each sentence
        for line in f:              
            num += 1
            words = []
            line = line.strip()  # remove spaces at the beginning and the end
            # print(line)
            if not line:
                continue  # line is None
            for i in range(len(line)):
                if line[i] == " ":
                    continue  # skip space
                words.append(line[i])  
            # print(words)
            word_list.append(words)  
            text = line.split(" ")              # text=["共同","创造","美好","的","新","世纪","——","二○○一年","新年","贺词"]
            # print(text)
            labels = []
            for item in text:
                if item == "":
                    continue
                labels.extend(getlables(item))        # label each word ： "二○○一年" to "BMMME"
            # print(labels)
            label_list.append(labels)                # label_list
            assert len(labels) == len(words), "labels_len != words_len"
        print("We have", num, "lines in", path, "file processed")
        # saved as a binary_file
        np.savez_compressed(out_path, words=word_list, labels=label_list)


if __name__ == '__main__':
    # preprocess(TRAIN,PROCESSED_train)
    preprocess(TEST,PROCESSED_test)