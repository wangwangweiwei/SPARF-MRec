import torch.nn as nn
from transformers import AutoModel
import torch

class professionalDocter(nn.Module):
    def __init__(self, bertModel , num_hidden=768):
        super(professionalDocter, self).__init__()

        self.bert = bertModel
    
        self.proliner_one = nn.Linear(num_hidden * 6, num_hidden * 2)
        self.proliner_two = nn.Linear(num_hidden * 2, num_hidden)
        #self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        
    def forward(self, d_token, d_amask, p_doken, p_mask):

        d_logit = self.bert(input_ids=d_token
                            , attention_mask=d_amask)
        p_logit = self.bert(input_ids=p_doken
                            , attention_mask=p_mask)
#         #医生的专业                   
#         #batch_size, 768=num_hidden
#         a = torch.max(d_logit[-11], dim=1)[0]
#         b = torch.max(d_logit[-12], dim=1)[0]
#         #患者补充的医生表现                    
#         c = torch.max(p_logit[-11], dim=1)[0]
#         d = torch.max(p_logit[-12], dim=1)[0]
        
#         #batch_size , 768*4=num_hidden*4
#         output = torch.cat([a,c,b,d],dim=-1)
        
#         output = self.proliner_two(self.dropout(self.relu(self.proliner_one(output))))
#         #batch_size, 768   
#         return {'dialog_embeddings':output}
        a = torch.cat([d_logit['hidden_states'][-11][:,0,:], \
    d_logit['hidden_states'][-12][:,0,:], d_logit['pooler_output']], dim=-1)
        b = torch.cat([p_logit['hidden_states'][-11][:,0,:], \
    p_logit['hidden_states'][-12][:,0,:], p_logit['pooler_output']], dim=-1)
        #batch_size , 768*6=num_hidden*6
        output = torch.cat([a,b],dim=-1)
        output = self.proliner_two(self.dropout(self.relu(self.proliner_one(output))))
        #batch_size, 768   
        return {'dialog_embeddings':output}
