import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModel, get_scheduler
from sklearn.metrics import accuracy_score
from tqdm import tqdm, trange
from info_nce import InfoNCE

class config:
    #profile_model_name = 'freedomking/mc-bert'
    profile_model_name = 'ernie-health-zh'
    dialog_model_name = 'bert-base-chinese'
    

    #未加入伪标签的结果来训练embedding
    #emotion_model_checkpoint = '/home/jovyan/work/train_step_two/emo_ckpt/model-epoch=9-macro_f1=0.5818.ckpt'
    #加入伪标签的结果来训练embedding
    emotion_model_checkpoint = '/home/jovyan/work/train_step_two/emo_ckpt/model-epoch=1-macro_f1=0.5939.ckpt'
    
    
    dialog_tokenizer = AutoTokenizer.from_pretrained(dialog_model_name)
    profile_tokenizer = AutoTokenizer.from_pretrained(profile_model_name)
    epochs = 3
    learning_rate = 8e-5
    dropout = 0.2
    batch_size = 64
    warmup_ratio = 0.1
    val_step = 2
    accumulation_steps = 2
    seed = 0
    save_params_path = "/kaggle/working"
    snapshot = 'snapshot'
    save_every=2
    #对比学校batch内采样的个数
    sample_cl = 1
    debug = False
    CE_loss = nn.CrossEntropyLoss()
    #bacth 内负样本
    temperature=0.05
    loss_weight = 0.1
    #pip install info-nce-pytorch
    InfoNCE_loss = InfoNCE(temperature)
    accuracy_score = accuracy_score
    #device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
