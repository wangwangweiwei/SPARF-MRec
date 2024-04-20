from config import config
import torch
import torch.nn as nn

class queryModel(nn.Module):
    def __init__(self, bertModel):
        super(queryModel, self).__init__()
        
        self.bert = bertModel
        #6为其他的子特征个数
        self.liner_one = nn.Linear(3, 32)
        self.liner_two = nn.Linear(800, 800)
        self.liner_three = nn.Linear(800, 768)
        self.drop = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()
    
    def forward(self, q_text_tokens_ids, q_text_tokens_amask, q_feature):
        #batch_size, num_hidden

        q_text_logit = self.bert(input_ids = q_text_tokens_ids
                                ,attention_mask = q_text_tokens_amask)['pooler_output']
        #batch_size, 32
        q_feature = self.liner_one(q_feature)
        input_x = self.drop(self.relu(self.liner_two(torch.cat([q_text_logit, q_feature], dim = 1))))
        output = self.liner_three(input_x)
        return output
