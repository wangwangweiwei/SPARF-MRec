
from config import config
from docterModel import docterModel
from bertBiGRU_Batch import bertBiGRU_Batch
from dialogCNN import dialogCNN
from queryModel import queryModel
from professionalDocter import professionalDocter
from MultiHeadAtt import MultiHeadAtt

import torch.nn as nn
import torch
from collections import OrderedDict
from transformers import AutoModel

class fusion_Model(nn.Module):
    def __init__(self):
        super(fusion_Model, self).__init__()
        self.pro_q_bert = AutoModel.from_pretrained(config.profile_model_name, output_hidden_states=True)
        unfreeze_layers = ['layer.10', 'layer.11', 'bert.pooler', 'out.']
        for name, param in self.pro_q_bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    # 加快遍历的速度
                    break

        self.docterModel = docterModel(self.pro_q_bert)
        self.queryModel = queryModel(self.pro_q_bert)
        self.professionalDocter = professionalDocter(self.pro_q_bert)
        

        model_checkpoint = torch.load(config.emotion_model_checkpoint, map_location=torch.device('cpu'))
        new_state_dict = OrderedDict()
        for k , v in model_checkpoint['state_dict'].items():
            name = k[22:]
            new_state_dict[name] = v
        self.bertBiGRU_Batch = bertBiGRU_Batch(model_name = model_checkpoint['hyper_parameters']['model_name'], 
                window_size = model_checkpoint['hyper_parameters']['window_size'],
                num_layer = model_checkpoint['hyper_parameters']['num_layer'], 
                input_size = model_checkpoint['hyper_parameters']['input_size'], 
                num_hidden = model_checkpoint['hyper_parameters']['num_hidden'], 
                dropout = model_checkpoint['hyper_parameters']['dropout'])
        self.bertBiGRU_Batch.load_state_dict(new_state_dict, strict=False)
        
        for name, param in self.bertBiGRU_Batch.named_parameters():
            param.requires_grad = False
        
        self.dialogCNN = dialogCNN()
        
        
        self.MultiHeadAtt = MultiHeadAtt(768 * 2, 4, config.dropout)
        self.layerNorm = nn.LayerNorm(768 * 2)
        #当总数据能整除batch_size时是正确的
        self.liner_one = nn.Linear(768*5, 1024)
        #self.liner_one = nn.Linear(2304+batch_size, 1024)
        self.dropout = nn.Dropout(config.dropout)
        self.relu = nn.ReLU()
        self.liner_two = nn.Linear(1024, 2)
        #情感表现
        self.emotion_liner_one = nn.Linear(768*2, 768)
        self.emotion_liner_two = nn.Linear(768, 2)
        #专业表现
        self.pro_liner_one = nn.Linear(768*2, 768)
        self.pro_liner_two = nn.Linear(768, 2)

    def forward(self, input_x):
        
        #batch_size, 768=num_hidden
        profile = self.docterModel(input_x['goodat_tokens_ids'], input_x['goodat_tokens_amask'], input_x['patient_like'])
        #batch_size, 768=num_hidden
        query = self.queryModel(input_x['q_text_tokens_ids'], input_x['q_text_tokens_amask'], input_x['q_feature'])
        
        #batch_size, max_utterance, 256=num_hidden
        dialog_emotion = self.bertBiGRU_Batch(input_x['roles'], input_x['dialog_token_ids'], \
                                              input_x['dialog_token_amask'], input_x['valid_length'])
        #batch_size, 768
        dialog_emotion = self.dialogCNN(dialog_emotion)['features']
        
        #batch_size, 768
        dialog_professional = self.professionalDocter(input_x['d_utterance_token_ids'],input_x['d_utterance_token_amask'],\
                                                     input_x['p_utterance_token_ids'],input_x['p_utterance_token_amask'])['dialog_embeddings']
        #batch_size, 1, 768*2=num_hidden*2
        dialog_all = torch.cat([dialog_emotion, dialog_professional], dim=-1).unsqueeze(1)
        
        dialog_all_att = self.MultiHeadAtt(dialog_all, dialog_all, dialog_all)[0]
        #batch_size, 768*2=num_hidden*2
        dialog_all = dialog_all.squeeze(1) + dialog_all_att.squeeze(1)
        dialog_all = self.layerNorm(dialog_all)
            

        output_all = torch.cat([dialog_all, dialog_emotion, dialog_professional, profile], dim=1)
        emo = torch.cat([dialog_emotion, query], dim=-1)
        pro = torch.cat([dialog_professional, query], dim=-1)
        output_all = self.liner_two(self.dropout(self.relu(self.liner_one(output_all))))
        output_emo = self.emotion_liner_two(self.dropout(self.relu(self.emotion_liner_one(emo))))
        output_pro = self.pro_liner_two(self.dropout(self.relu(self.pro_liner_one(pro))))
        
        output = output_all+output_emo+output_pro
        
        return {"logits":output}
        # return {"logits":output,
        #         "query": query,
        #         "profile":profile,
        #         "dialog_emotion": dialog_emotion,
        #         "dialog_professional": dialog_professional}
