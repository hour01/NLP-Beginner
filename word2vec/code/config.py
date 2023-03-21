import argparse


parser = argparse.ArgumentParser()

parser.add_argument('-mode', type=str, default='train', help="mode")
parser.add_argument('-window_size', type=int, default=5, help="context window")
parser.add_argument('-neg_samples', type=int, default=15, help="negative sample")
parser.add_argument('-epoch', type=int, default=2, help="")
parser.add_argument('-vocab', type=int, default=10000, help="vocab size")
parser.add_argument('-emb', type=int, default=100, help="embedding size")
parser.add_argument('-batch', type=int, default=32, help="")
parser.add_argument('-lr', type=float, default=0.002, help="")
parser.add_argument('-data_path', type=str, default='./data/corpus.txt', help="data path")
parser.add_argument('-processed_data', type=str, default='./processed_data', help="processed_data path")
parser.add_argument('-load_path', type=str, default='', help="load path")
parser.add_argument('-save_path', type=str, default='./CKPT/', help="save path")




config = parser.parse_args()