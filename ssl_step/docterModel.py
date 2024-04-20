from config import config
import torch
import torch.nn as nn
from transformers import AutoModel

class docterModel(nn.Module):
    def __init__(self, bertModel):
        super(docterModel, self).__init__()
        
        self.bert = bertModel
        #6为其他的子特征个数
        self.liner_one = nn.Linear(6, 32)
        self.liner_two = nn.Linear(800, 800)
        self.liner_three = nn.Linear(800, 768)
        self.drop = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()
    
    def forward(self, goodat_tokens_ids, goodat_tokens_amask, patient_like):
        #感觉这个顺序不会改变哈
        #batch_size, num_hidden

        goodat_logit = self.bert(input_ids = goodat_tokens_ids
                                ,attention_mask = goodat_tokens_amask)['pooler_output']
        #batch_size, 32
        patient_like = self.liner_one(patient_like)
        input_x = self.drop(self.relu(self.liner_two(torch.cat([patient_like, goodat_logit], dim = 1))))
        output = self.liner_three(input_x)
        return output
