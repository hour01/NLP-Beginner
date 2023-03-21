import argparse

parser = argparse.ArgumentParser()

# Training configs
parser.add_argument('--epoch', type=int, default=110, help="epoch")
parser.add_argument('--lr', type=float, default=0.01, help="Learning rate")
parser.add_argument('--decay_rate',type=float, default=0.95,help='decay rate for rmsprop')
parser.add_argument('--learning_rate_decay',type=float, default=0.97,help='learning rate decay')
parser.add_argument('--learning_rate_decay_after',type=int, default=5,help='in number of epochs, when to start decaying the learning rate')
parser.add_argument('--grad_clip',type=int, default=5,help='clip gradients at this value')
parser.add_argument('--batch_size', type=int, default=20, help="Batch size")
parser.add_argument('--eval_steps', type=int, default=200, help='Total number of training epochs to perform.')
parser.add_argument('--mode', type=str, default='train', help="Train or test")
parser.add_argument('--restore', type=str, default='', help="Restoring model path,ie. CKPT_DIR/checkpoint_latest.pt")
parser.add_argument('--dec_seq_length',type=int, default=100,help='number of timesteps to unroll for')

# Model configs
parser.add_argument('--init_weight',type=float, default=0.08,help='initailization weight')
parser.add_argument('--hidden_size', type=int, default=200, help="hidden size of lstm")
parser.add_argument('--layers', type=int, default=1, help="layers of lstm")
parser.add_argument('--dropout', type=float, default=0.3, help="Dropout rate")
parser.add_argument('--dropoutrec',type=int,default=0,help='dropout for regularization, used after each c_i. 0 = no dropout')

# path and data configs
parser.add_argument('--data_dir', default='./processed_data', help='The dataset path.', type=str)
parser.add_argument('--model_path', default='./CKPT', help='The model will be saved to this path.', type=str)
parser.add_argument('--train_file', default='train.pkl', type=str)
parser.add_argument('--val_file', default='dev.pkl', type=str)
parser.add_argument('--test_file', default='test.pkl', type=str)

parser.add_argument('--display', type=int, default=0, help='whether display on console')
parser.add_argument('--output', type=str, default='', help='path of prediction on test to file')

config = parser.parse_args()