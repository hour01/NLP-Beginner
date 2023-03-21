import argparse

parser = argparse.ArgumentParser()

# Training configs
parser.add_argument('--epoch', type=int, default=30, help="epoch")
parser.add_argument('--lr', type=float, default=1e-5, help="Learning rate")
parser.add_argument('--betas', type=tuple, default=(0.9, 0.999))
parser.add_argument('--lr_steps', type=int, default=3)
parser.add_argument('--lr_gamma', type=float, default=0.5)
parser.add_argument('--weight_decay', type=float, default=0.01)
parser.add_argument('--clip_grad', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=4, help="Batch size")
parser.add_argument('--mode', type=str, default='train', help="Train or test")
parser.add_argument('--restore', type=str, default='', help="Restoring model path,ie. CKPT_DIR/checkpoint_latest.pt")
parser.add_argument('--fine_tune', type=str, default='not_full', help='fine tune bert or not')

# Model configs
parser.add_argument('--init_weight',type=float, default=0.08,help='initailization weight')
parser.add_argument('--hidden_size', type=int, default=256, help="hidden size of lstm")
parser.add_argument('--emb_size', type=int, default=768, help="hidden size of lstm")
parser.add_argument('--layers', type=int, default=2, help="layers of lstm")
parser.add_argument('--lstm_dropout', type=float, default=0.2, help="Dropout rate")
parser.add_argument('--dropout', type=float, default=0.2, help="Dropout rate")

# path and data configs
parser.add_argument('--data_dir', default='./processed_data/', help='The dataset path.', type=str)
parser.add_argument('--model_path', default='./CKPT', help='The model will be saved to this path.', type=str)
parser.add_argument('--train_file', default='train_dev.npz', type=str)
parser.add_argument('--test_file', default='test.npz', type=str)
parser.add_argument('--log_dir', default='./logs/train.log', type=str)
parser.add_argument('--vocab_path', default='./processed_data/vocab.npz', type=str)
parser.add_argument('--vocab_file', default='./bert-base-chinese/vocab.txt', type=str)
parser.add_argument('--bert_path', default='./bert-base-chinese/', type=str)
parser.add_argument('--output_dir', default='./output.txt', type=str)
parser.add_argument('--case_dir', default='./bad_case.txt', type=str)

config = parser.parse_args()

config.label2id = {'B': 0, 'M': 1, 'E': 2, 'S': 3}
config.id2label = {_id: _label for _label, _id in list(config.label2id.items())}