import json
from collections import Counter
import argparse
import time
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm, trange


from MulAttDataset import MulAttDataset
from docterReco_Lightning import docterReco_Lightning
from config import init_opts, train_opts, multihead_att_opts, eval_opts

import lightning.pytorch as pl

parser = argparse.ArgumentParser('train MUL-ATT model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
init_opts(parser)
train_opts(parser)
eval_opts(parser)
multihead_att_opts(parser)
args = parser.parse_args()


def main():
    
    pl.seed_everything(args.seed, workers=True)
    
    print(f'Loadding embeddings from {args.embeddings_path}...')
    with open(f'{args.embeddings_path}/train_profile_embedding.json', 'r', encoding='utf-8') as f:
        profile_embeddings = json.load(f)
    print('Loadding profile embedding over')
    time.sleep(5)
    with open(f'{args.embeddings_path}/query_embeddings.json', 'r', encoding='utf-8') as f:
        query_embeddings = json.load(f)
    print('loadding query embedding over')
    time.sleep(5)
    with open(f'{args.embeddings_path}/dialog_professional_embeddings.json', 'r', encoding='utf-8') as f:
        dialog_professional_embeddings = json.load(f)
    print('loadding dialogue embedding over')
    time.sleep(5)
    with open(f'{args.embeddings_path}/dialog_emotion_embeddings.json', 'r', encoding='utf-8') as f:
        dialog_emotion_embeddings = json.load(f)
    print('loadding dialogue embedding over')
    print('Done')

    #1536 jwei V_3 output_14 dialog_sample 80 weight 1:6
    model_path = 'ckpt/model-epoch=85-val_loss=0.8025.ckpt'
    
    
    model = docterReco_Lightning.load_from_checkpoint(model_path)
    

    test_set = pd.read_csv('/home/jovyan/input/my_data_002/dr_re_data_update/test_age.csv', encoding='utf-8', dtype={'dr_id':str})
    test_dataset = MulAttDataset(
        'test', test_set, profile_embeddings, query_embeddings, dialog_professional_embeddings, dialog_emotion_embeddings,
        dr_dialog_sample=args.dr_dialog_sample, neg_sample=args.neg_sample, output=args.output_dir)
    
    Mydata = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=16)
    
    trainer = pl.Trainer(devices=1)

    pred_score = trainer.predict(model, Mydata)
    pred_score = torch.cat(pred_score)
    df = pd.DataFrame(pred_score.tolist())
    
    df.to_csv(f'{args.output_dir}/test_{args.eval_model}_score.txt', index=False, header=None)

if __name__ == '__main__':
    main()
