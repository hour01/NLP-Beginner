import argparse

parser = argparse.ArgumentParser()

# Training configs
parser.add_argument('--epoch', type=int, default=110, help="epoch")
parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
parser.add_argument('--decay_rate',type=float, default=0.95,help='decay rate for rmsprop')
parser.add_argument('--learning_rate_decay',type=float, default=0.97,help='learning rate decay')
parser.add_argument('--learning_rate_decay_after',type=int, default=5,help='in number of epochs, when to start decaying the learning rate')
parser.add_argument('--grad_clip',type=int, default=5,help='clip gradients at this value')
parser.add_argument('--batch_size', type=int, default=1, help="Batch size")
parser.add_argument('--mode', type=str, default='train', help="Train or test")
parser.add_argument('--restore', type=str, default='', help="Restoring model path,ie. CKPT_DIR/checkpoint_latest.pt")

# path and data configs
parser.add_argument('--data_path', default='./data', help='data path.', type=str)
parser.add_argument('--data_dir', default='./data/processed_data', help='The processed dataset path.', type=str)
parser.add_argument('--word_vec_file', default='glove.6B.100d.txt', help='glove embedding file', type=str)
parser.add_argument('--model_path', default='./CKPT', help='The model will be saved to this path.', type=str)
parser.add_argument('--train_file', default='train.conll', type=str)
parser.add_argument('--val_file', default='dev.conll', type=str)
parser.add_argument('--test_file', default='test.conll', type=str)

config = parser.parse_args()

# Default configurations.
config.configuration = {'unk_prob': 0.1,
                 'dropout_rate': 0.01,
                 'token_dim': 100,
                 'pos_dim': 100,
                 'lu_dim': 100,
                 'lu_pos_dim': 100,
                 'lstm_input_dim': 100,
                 'lstm_dim': 100,
                 'lstm_depth': 2,
                 'hidden_dim': 100,
                 'pretrained_embedding_dim': 100,
                 'patience': 25,
                 'eval_after_every_epochs': 100}