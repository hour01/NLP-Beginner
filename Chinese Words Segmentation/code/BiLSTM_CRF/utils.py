import torch
import logging
from torch.utils.data import Dataset
import numpy as np
import random
from sklearn.model_selection import train_test_split
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

def get_embedding(config):
    if os.path.exists(config.word_vec_path):
        return torch.load(config.word_vec_path)
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format(
        config.embedding_path, binary=False, encoding='utf-8')
    vocab_size = len(config.vocab)
    embed_size = config.emb_size
    weight = torch.zeros(vocab_size+1, embed_size)
    cnt = 0
    for i in range(len(word2vec_model.index_to_key)):
        try:
            index = config.vocab.word_id(word2vec_model.index_to_key[i])
        except:
            continue
        cnt += 1
        weight[index, :] = torch.from_numpy(word2vec_model.get_vector(
            config.vocab.id_word(config.vocab.word_id(word2vec_model.index_to_key[i]))))
    torch.save(weight,config.word_vec_path)
    logging.info("--------Pretrained Embedding Loaded ! ({}/{})--------".format(cnt, len(config.vocab)))
    return weight

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
        if os.path.exists(self.vocab_path):
            data = np.load(self.vocab_path, allow_pickle=True)
            self.word2id = data["word2id"][()]
            self.id2word = data["id2word"][()]
            logging.info("-------- Vocabulary Loaded! --------")
            return
        word_freq = {}
        data = np.load(self.config.data_dir+self.config.train_file, allow_pickle=True)
        word_list = data["words"]
        for line in word_list:
            for ch in line:
                if ch in word_freq:
                    word_freq[ch] += 1
                else:
                    word_freq[ch] = 1
        word_freq['<UNK>'] = 1
        sorted_word = sorted(word_freq.items(), key=lambda e: e[1], reverse=True)
        # 构建word2id字典
        for elem in sorted_word:
            self.word2id[elem[0]] = len(self.word2id)
        # id2word保存
        self.id2word = {_idx: _word for _word, _idx in list(self.word2id.items())}
        # 保存为二进制文件
        np.savez_compressed(self.vocab_path, word2id=self.word2id, id2word=self.id2word)
        logging.info("-------- Vocabulary Build! --------")

class SegDataset(Dataset):
    def __init__(self, words, labels, vocab, label2id):
        self.vocab = vocab
        self.dataset = self.preprocess(words, labels)
        self.label2id = label2id

    def preprocess(self, words, labels):
        """convert the data to ids"""
        processed = []
        for (word, label) in zip(words, labels):
            word_id = [self.vocab.word_id(u_) if u_ in self.vocab.word2id else self.vocab.word_id('<UNK>') for u_ in word]
            label_id = [self.vocab.label_id(l_) for l_ in label]
            processed.append((word_id, label_id))
        return processed

    def __getitem__(self, idx):
        word = self.dataset[idx][0]
        label = self.dataset[idx][1]
        return [word, label]

    def __len__(self):
        return len(self.dataset)

    def get_long_tensor(self, words, labels, batch_size):
        token_len = max([len(x) for x in labels])
        word_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        label_tokens = torch.LongTensor(batch_size, token_len).fill_(0)
        mask_tokens = torch.ByteTensor(batch_size, token_len).fill_(0)

        for i, s in enumerate(zip(words, labels)):
            word_tokens[i, :len(s[0])] = torch.LongTensor(s[0])
            label_tokens[i, :len(s[1])] = torch.LongTensor(s[1])
            mask_tokens[i, :len(s[0])] = torch.tensor([1] * len(s[0]), dtype=torch.uint8)

        return word_tokens, label_tokens, mask_tokens

    def collate_fn(self, batch):

        words = [x[0] for x in batch]
        labels = [x[1] for x in batch]
        lens = [len(x) for x in labels]
        batch_size = len(batch)

        word_ids, label_ids, input_mask = self.get_long_tensor(words, labels, batch_size)

        return [word_ids, label_ids, input_mask, lens]