import argparse, os
from model.models import Model

# Configurations
parser = argparse.ArgumentParser(description="Experiment Settings.")

parser.add_argument('-gpus', default=[], nargs='+', type=int,
                    help='Use CUDA on the listed devices.')

parser.add_argument('-epochs', default=50, type=int,
                    help='Training epochs.')

parser.add_argument('-batch_size', default=8, type=int,
                    help='The size of one mini batch.')

parser.add_argument('-lr', default=0.001, type=float,
                    help='Learing rate.')

parser.add_argument('-sen_len', default=300, type=int,
                    help='The length of truncated sentences.')

parser.add_argument('-label_size', default=414, type=int,
                    help='The capacity of label space.')

parser.add_argument('-vocab_size', default=20000, type=int,
                    help='The capacity of vocabulary space.')

parser.add_argument('-embed_size', default=128, type=int,
                    help='The size of the word vectors.')

parser.add_argument('-iterations', default=3, type=int,
                    help='Number of iteration times.')

parser.add_argument('-loss_fn', default='margin_loss_v1', type=str,
                    help='loss functions.')

parser.add_argument('-dataname', default='aapd', type=str,
                    help='Name of dataset.')

parser.add_argument('-bestpath', default=r'./data/saved/best/best_model.ckpt', type=str,
                    help='File Path of the best models.')

cfg = parser.parse_args()

if __name__ == "__main__":
    # print(cfg.gpus[0])
    model = Model(cfg)
    # model.BuildArch()
    model.train()