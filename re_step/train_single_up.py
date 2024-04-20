                                                                                                    
##单卡没有问题，优化器改为warmup
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
from config import init_opts, train_opts, multihead_att_opts

import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping


parser = argparse.ArgumentParser('train model', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
init_opts(parser)
train_opts(parser)
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
    
    print('Building training dataset and dataloader...')
    train_set = pd.read_csv('/home/jovyan/input/my_data_002/dr_re_data_update/train_age.csv', encoding='utf-8', dtype={'dr_id':str})
    #专业在前，情感在后
    train_dataset = MulAttDataset(
        'train', train_set, profile_embeddings, query_embeddings, dialog_professional_embeddings, dialog_emotion_embeddings,
        dr_dialog_sample=args.dr_dialog_sample, neg_sample=args.neg_sample
    )
    train_dataLoader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    del train_set, train_dataset
    print('Done')
    
    print('Building validation dataset and dataloader...')
    valid_set = pd.read_csv('/home/jovyan/input/my_data_002/dr_re_data_update/valid_age.csv', encoding='utf-8', dtype={'dr_id':str})
    val_dataset = MulAttDataset(
        'valid', valid_set, profile_embeddings, query_embeddings, dialog_professional_embeddings, dialog_emotion_embeddings,
        dr_dialog_sample=args.dr_dialog_sample, neg_sample=args.neg_sample
    )
    val_dataLoader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=16)
    del valid_set, val_dataset, profile_embeddings, query_embeddings, dialog_professional_embeddings, dialog_emotion_embeddings
    print('Done')
    
    model = docterReco_Lightning(args.in_size, args.hidden_size, args.dropout, 
                 args.head_num, args.seed, args.batch_size, args.patience, 
                 args.lr, args.dr_dialog_sample, args.neg_sample, args.warmup_ratio)

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='val_loss',
        dirpath='/home/jovyan/work/re_step_two/ckpt/',
        filename='model-{epoch}-{val_loss:.4f}',
        save_top_k = 1,
        mode='min')
    
    early_stop_callback = EarlyStopping(
        monitor="val_loss", 
        min_delta=0.00, 
        patience=args.patience, 
        verbose=False, 
        mode="min")


    trainer = pl.Trainer(max_epochs=args.epoch_num,
                         devices=1,
                         callbacks=[checkpoint_callback,early_stop_callback]
                        ,gradient_clip_val=20, gradient_clip_algorithm="value"
                        )

    trainer.fit(model=model, train_dataloaders=train_dataLoader, 
                val_dataloaders=val_dataLoader
               )

if __name__ == '__main__':
    main()
