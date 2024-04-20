import argparse
import torch

torch.set_float32_matmul_precision('medium')

def init_opts(parser: argparse.ArgumentParser):
    parser.add_argument('-seed', type=int, default=1536)
    
    parser.add_argument('-name', type=str, default='train')

    #加入了伪标签训练的embedding哈
    parser.add_argument('-embeddings_path', type=str, default='/home/jovyan/work/embedding_step_two/')
    parser.add_argument('-output_dir', type=str, default='/home/jovyan/work/re_step_two/')
    
def train_opts(parser: argparse.ArgumentParser):
    
    parser.add_argument('-epoch_num', default=120, type=int)
    parser.add_argument('-batch_size', default=256, type=int)
    parser.add_argument('-patience', default=4, type=int)
    parser.add_argument('-lr', default=6e-4, type=float)
    parser.add_argument('-dropout', default=0.3, type=float)
    parser.add_argument('-in_size', default=768, type=int)
    parser.add_argument('-hidden_size', default=256, type=int)
    parser.add_argument('-dr_dialog_sample', default=80, type=int)
    parser.add_argument('-neg_sample', default=20, type=int)
    parser.add_argument('-warmup_ratio', default=0.1, type=float)

def eval_opts(parser: argparse.ArgumentParser):
    parser.add_argument('-eval_model', default="best_model.pt", type=str)

def multihead_att_opts(parser: argparse.ArgumentParser):
    parser.add_argument('-head_num', default=6, type=int)
    parser.add_argument('-add_self_att_on', default="none", type=str)    
