import torch
import logging
from torch.utils.data import Dataset
import numpy as np
import random
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer
import gensim
import os

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

def dev_split(dataset_dir):
    '''
    split the data to train and dev
    '''
    data = np.load(dataset_dir, allow_pickle=True)
    words = data["words"]
    labels = data["labels"]
    x_train, x_dev, y_train, y_dev = train_test_split(words, labels, test_size=0.1, random_state=0)
    return x_train, x_dev, y_train, y_dev  

def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.
    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.
    Example:
    ```
    logging.info("Starting training...")
    ```
    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)

class Vocabulary:
    """
    construct Vocabulary
    """
    def __init__(self, config):
        self.config = config
        self.vocab_path = config.vocab_path
        self.word2id = {}
        self.id2word = None
        self.label2id = config.label2id
        self.id2label = config.id2label

    def __len__(self):
        return len(self.word2id)

    def vocab_size(self):
        return len(self.word2id)

    def label_size(self):
        return len(self.label2id)

    def word_id(self, word):
        return self.word2id[word]

    def id_word(self, idx):
        return self.id2word[idx]

    def label_id(self, word):
        return self.label2id[word]

    def id_label(self, idx):
        return self.id2label[idx]

    def get_vocab(self):
        pass

class SegDataset(Dataset):
    def __init__(self, words, labels, config, word_pad_idx=0, label_pad_idx=-1):
        self.tokenizer = BertTokenizer(config.vocab_file)
        self.label2id = config.vocab.label2id
        self.dataset = self.preprocess(words, labels)
        self.word_pad_idx = word_pad_idx
        self.label_pad_idx = label_pad_idx

    def preprocess(self, origin_sentences, origin_labels):
        '''
        Maps tokens and tags to their indices and stores them in the dict data.
        examples: 
            word:['[CLS]', '浙', '商', '银', '行', '企', '业', '信', '贷', '部']
            sentence:([101, 3851, 1555, 7213, 6121, 821, 689, 928, 6587, 6956],
                        array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10]))
            label:[3, 13, 13, 13, 0, 0, 0, 0, 0]
        '''
        data = []
        sentences = []
        labels = []
        for line in origin_sentences:
            # replace each token by its index
            words = []
            word_lens = []
            # replace the token to [UNK] which is not in bert_vocab.
            for token in line:
                words.append(self.tokenizer.tokenize(token))
                word_lens.append(len(token))
            # append the [CLS] token before the sentence
            words = ['[CLS]'] + [item for token in words for item in token]
            token_start_idxs = 1 + np.cumsum([0] + word_lens[:-1])
            # 拼接原句
            sentences.append(((self.tokenizer.convert_tokens_to_ids(words), token_start_idxs), line))
        for tag in origin_labels:
            label_id = [self.label2id.get(t) for t in tag]
            labels.append(label_id)
        for sentence, label in zip(sentences, labels):
            data.append((sentence, label))
        return data

    def __getitem__(self, idx):
        """sample data to get batch"""
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        """get dataset size"""
        return len(self.dataset)

    def collate_fn(self, batch):
        """
        process batch data, including:
            1. padding
            2. aligning: 找到每个sentence sequence里面有label项, 文本与label对齐
        """
        sentences = [x[0][0] for x in batch]
        ori_sents = [x[0][1] for x in batch]
        labels = [x[1] for x in batch]

        # batch length
        batch_len = len(sentences)

        # compute length of longest sentence in batch
        max_len = max([len(s[0]) for s in sentences])
        max_label_len = 0

        # padding data 初始化
        batch_data = self.word_pad_idx * np.ones((batch_len, max_len))
        batch_label_starts = []

        # padding and aligning
        for j in range(batch_len):
            cur_len = len(sentences[j][0])
            batch_data[j][:cur_len] = sentences[j][0]
            # 找到有标签的数据的index（[CLS]不算）
            label_start_idx = sentences[j][-1]
            label_starts = np.zeros(max_len)
            label_starts[[idx for idx in label_start_idx if idx < max_len]] = 1
            batch_label_starts.append(label_starts)
            max_label_len = max(int(sum(label_starts)), max_label_len)

        # padding label
        batch_labels = self.label_pad_idx * np.ones((batch_len, max_label_len))
        for j in range(batch_len):
            cur_tags_len = len(labels[j])
            batch_labels[j][:cur_tags_len] = labels[j]

        # convert data to torch LongTensors
        batch_data = torch.tensor(batch_data, dtype=torch.long)
        batch_label_starts = torch.tensor(np.array(batch_label_starts), dtype=torch.long)
        batch_labels = torch.tensor(batch_labels, dtype=torch.long)
        return [batch_data, batch_label_starts, batch_labels, ori_sents]