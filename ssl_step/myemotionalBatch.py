from config import config
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import json
import random
import numpy as np
import pandas as pd
import torch

class myemotionalBatch(Dataset):
    def __init__(self, split):
        super(myemotionalBatch, self).__init__()
        

        #基于召回的5个医生负例
        self.data = pd.read_csv(f'/home/jovyan/input/my_data_002/dr_re_data_update/{split}_emotional_fusion_recall.csv')
        self.dr_profile = pd.read_csv('/home/jovyan/input/my_data_002/dr_re_data_update/dr_profile.csv', index_col=0)
        self.query = pd.read_csv(f'/home/jovyan/input/my_data_002/dr_re_data_update/{split}_age.csv', index_col=1)
        with open('/home/jovyan/input/my_data_002/dr_re_data_update/dialog_role_index.json', 'r') as fp:
            self.dialog = json.load(fp)
        if config.debug:
            self.data = self.data[:1024]
        
            
    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, index):
        dialog_id = self.data.loc[index, 'dialog_id']
        dr_id = self.data.loc[index, 'dr_id']
        
        
        urrerances = self.dialog[dialog_id]['dialog']
        roles = self.dialog[dialog_id]['role']
        if len(urrerances) > 32:
            urrerances = urrerances[:16] + urrerances[-16:]
            roles = roles[:16] + roles[-16:]
        
        dialog_token = config.dialog_tokenizer(urrerances
                                           ,padding='max_length'
                                           ,truncation = True
                                           ,max_length = 128
                                           ,return_tensors = 'pt')
            
        patient_like = self.dr_profile.loc[dr_id, ['clinic',"title","serviceNum","goodRate","peerRec", "patientLike"]].tolist()
        
        goodat = self.dr_profile.loc[dr_id, 'goodat']
        q_text = self.query.loc[dialog_id,'q']
        
        goodat_tokens = config.profile_tokenizer(goodat
                                           ,padding='max_length'
                                           ,truncation = True
                                           ,max_length = 128
                                           ,return_tensors = 'pt')
        q_text_tokens = config.profile_tokenizer(q_text
                                           ,padding='max_length'
                                           ,truncation = True
                                           ,max_length = 128
                                           ,return_tensors = 'pt')
        
        q_feature = self.query.loc[dialog_id, ['age', 'gender_female', 'gender_man']].tolist()
        ##专业的token
        d_utterance = []
        p_utterance = []
        for i, j in zip(roles, urrerances):
            if i == 0:
                p_utterance.append(j)
            else:
                d_utterance.append(j)
        d_utterance_token = config.profile_tokenizer(''.join(d_utterance)
                                           ,padding = 'max_length'
                                           ,truncation = True
                                           ,max_length = 512
                                           ,return_tensors = 'pt')
        p_utterance_token = config.profile_tokenizer(''.join(p_utterance)
                                           ,padding = 'max_length'
                                           ,truncation = True
                                           ,max_length = 512
                                           ,return_tensors = 'pt')
        valid_length = len(roles)
        label = self.data.loc[index, 'label']
        
        return torch.tensor(roles), dialog_token['input_ids'],dialog_token['attention_mask'], \
               d_utterance_token['input_ids'],d_utterance_token['attention_mask'],\
               p_utterance_token['input_ids'], p_utterance_token['attention_mask'],\
               torch.tensor(patient_like), goodat_tokens['input_ids'], goodat_tokens['attention_mask'],\
               q_text_tokens['input_ids'], q_text_tokens['attention_mask'],\
               torch.tensor(q_feature), torch.tensor(valid_length), torch.tensor(label)
    

    
    def collate_fn(self, sample):
        data = pd.DataFrame(sample)
        #batch_size, max_utterance
        roles = pad_sequence(data[0], True)
        #input_ids
        #batch_size, max_utterance, 128
        dialog_token_ids = pad_sequence(data[1], True)
        #batch_size, max_utterance, 128
        dialog_token_amask = pad_sequence(data[2], True)
        
        d_utterance_token_ids = torch.tensor(np.concatenate(data[3]))
        d_utterance_token_amask = torch.tensor(np.concatenate(data[4]))
        
        p_utterance_token_ids = torch.tensor(np.concatenate(data[5]))
        p_utterance_token_amask = torch.tensor(np.concatenate(data[6]))
        
        patient_like = torch.tensor(np.stack(data[7]), dtype=torch.float32)
        
        goodat_tokens_ids = torch.tensor(np.concatenate(data[8]))
        goodat_tokens_amask = torch.tensor(np.concatenate(data[9]))
        
        q_text_tokens_ids = torch.tensor(np.concatenate(data[10]))
        q_text_tokens_amask = torch.tensor(np.concatenate(data[11]))
        q_feature = torch.tensor(np.stack(data[12]), dtype=torch.float32)
        valid_length = torch.tensor(np.stack(data[13]))
        label = torch.tensor(np.stack(data[14]))
        
        return {'roles': roles,'dialog_token_ids':dialog_token_ids, 'dialog_token_amask':dialog_token_amask,\
               'd_utterance_token_ids':d_utterance_token_ids, 'd_utterance_token_amask':d_utterance_token_amask,\
               'p_utterance_token_ids':p_utterance_token_ids, 'p_utterance_token_amask':p_utterance_token_amask,\
               'patient_like':patient_like, 'goodat_tokens_ids':goodat_tokens_ids,'goodat_tokens_amask':goodat_tokens_amask,\
               'q_text_tokens_ids':q_text_tokens_ids, 'q_text_tokens_amask':q_text_tokens_amask,\
               'q_feature':q_feature, 'valid_length': valid_length, 'label':label}
