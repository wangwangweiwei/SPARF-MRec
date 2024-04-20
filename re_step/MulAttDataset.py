from random import sample
from collections import Counter
import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
import json
import numpy as np

class MulAttDataset(Dataset):
    def __init__(self, split, dataset, profile_embeddings, query_embeddings, d_professional_embeddings,
                 d_emotion_embeddings,
                 dr_dialog_sample=100, neg_sample=10, embed_size=768, output=''):
        self.split = split
        self.dataset = dataset
        self.q_list = dataset.dialog_id.tolist() # query id - the same as dialogue id
        self.dr_list = dataset.dr_id.unique().tolist()
        self.q_dr_match = dict(zip(dataset.dialog_id, dataset.dr_id))
        self.profile_emb = profile_embeddings
        self.q_emb = query_embeddings
        self.dialog_p_emb = d_professional_embeddings
        self.dialog_e_emb = d_emotion_embeddings

        train_set = pd.read_csv('/home/jovyan/input/my_data_002/dr_re_data_update/train_age.csv', encoding='utf-8', dtype={'dr_id':str})
        #抽取概率
        p = train_set.groupby('dr_id').size()
        self.d_samplepro = dict(zip(p.index, p.values))
        
        with open(f'/home/jovyan/input/my_data_003/neg_drs/{split}_q2dr_list.json', 'r') as fp:
            self.neg_drs = json.load(fp)
            
        self.most_common_drs = [dr for dr, _ in Counter(train_set.dr_id.tolist()).most_common()]
        self.train_q_dr_match = dict(zip(train_set.dialog_id, train_set.dr_id))
        del train_set, p
        self.dr_dialog_sample = dr_dialog_sample
        self.neg_sample = neg_sample
        self.embed_size = embed_size
        self.output = output
        self.dr_feature = {}
        self.features = []
        self.labels = []
        for dr in tqdm(self.dr_list, desc='packing doctor features'):
            self.pack_dr_features(dr)
        self.pack_dataset()

    def __getitem__(self, index):
        return torch.FloatTensor(self.features[index]), torch.FloatTensor([self.labels[index]])[0]

    def __len__(self):
        return len(self.labels)

    def pack_dr_features(self, dr_id):
        feature = []
        feature_profile = self.profile_emb[dr_id]
        feature.append(feature_profile)
        records = [dialog_id for (dialog_id, doctor_id) in self.train_q_dr_match.items() if doctor_id == dr_id]
        if len(records) > self.dr_dialog_sample:
            sample_records = sample(records, self.dr_dialog_sample)
            for idx in sample_records:
                feature.append(self.dialog_p_emb[idx])
                feature.append(self.dialog_e_emb[idx])
        else:

            pad_size = 2*(self.dr_dialog_sample - len(records))
            for idx in records:
                feature.append(self.dialog_p_emb[idx])
                feature.append(self.dialog_e_emb[idx])
            feature.extend([[0] * self.embed_size] * pad_size)
        self.dr_feature[dr_id] = feature
        return feature

    def pack_dataset(self):
     # 负例基于召回的数据集       
        if self.split == "test":
            test_dat = open(f'{self.output}/test.dat', 'w', encoding='utf-8')
        for (q_idx, q) in enumerate(tqdm(self.q_list, desc=f'pack {self.split} dataset')):
            q_feature = self.q_emb[q]
            pos_dr = self.q_dr_match[q]
            pos_feature = self.dr_feature[pos_dr][:]
            pos_feature.append(q_feature)
            if self.split == 'test':
                print(f'# query {q_idx+1} {q} {pos_dr}', file=test_dat)
                print(f"1 'qid':{q_idx+1} # doctor: {pos_dr}", file=test_dat)
            self.features.append(pos_feature)
            self.labels.append(1)

            # negtive sampling
            neg_drs = self.neg_drs[q][:10]
            for neg_dr in neg_drs:
                neg_feature = self.dr_feature[neg_dr][:]
                neg_feature.append(q_feature)
                if self.split == 'test':
                    print(f"0 'qid':{q_idx+1} # doctor: {neg_dr}", file=test_dat)
                self.features.append(neg_feature)
                self.labels.append(0)
        if self.split == 'test':
            test_dat.close()
