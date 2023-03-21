import argparse

parser = argparse.ArgumentParser()

# Training configs
parser.add_argument('--epoch', type=int, default=110, help="epoch")
parser.add_argument('--lr', type=float, default=0.001, help="Learning rate")
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
parser.add_argument('--lr_steps', type=int, default=3)
parser.add_argument('--lr_gamma', type=float, default=0.5)
parser.add_argument('--batch_size', type=int, default=32, help="Batch size")
parser.add_argument('--mode', type=str, default='train', help="Train or test")
parser.add_argument('--restore', type=str, default='', help="Restoring model path,ie. CKPT_DIR/checkpoint_latest.pt")

# Model configs
parser.add_argument('--hidden_size', type=int, default=256, help="hidden size of lstm")
parser.add_argument('--emb_size', type=int, default=300)
parser.add_argument('--layers', type=int, default=2, help="layers of lstm")
parser.add_argument('--lstm_dropout', type=float, default=0.2, help="Dropout rate")

# path and data configs
parser.add_argument('--data_dir', default='./processed_data/', help='The dataset path.', type=str)
parser.add_argument('--model_path', default='./CKPT', help='The model will be saved to this path.', type=str)
parser.add_argument('--train_file', default='train_dev.npz', type=str)
parser.add_argument('--test_file', default='test.npz', type=str)
parser.add_argument('--log_path', default='./logs/train.log', help='The log will be saved to this path.', type=str)
parser.add_argument('--vocab_path', default='./processed_data/vocab.npz', type=str)
parser.add_argument('--output_dir', default='./output.txt', type=str)
parser.add_argument('--case_dir', default='./bad_case.txt', type=str)
parser.add_argument('--word_vec_path', default='./processed_data/word_vec.pt', type=str)
parser.add_argument('--embedding_path', default='./processed_data/sgns.target.word-character.char1-2.dynwin5.thr10.neg5.dim300.iter5.bz2', type=str)

config = parser.parse_args()

config.label2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
config.id2label = {_id: _label for _label, _id in list(config.label2id.items())}