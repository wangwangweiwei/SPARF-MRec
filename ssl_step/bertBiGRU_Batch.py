
import torch
import torch.nn as nn
from transformers import AutoModel
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
'''
带有角色情感的专业属性输出
'''
class bertBiGRU_Batch(nn.Module):
    def __init__(self, model_name, window_size = 5, 
                 num_layer = 1, input_size = 768, 
                 num_hidden = 256, dropout = 0.3):
        super(bertBiGRU_Batch, self).__init__()
        self.dialog_max = 0
        
        #局部窗口的大小
        self.window_size = window_size
        self.input_size = input_size
        
        self.bert = AutoModel.from_pretrained(model_name)
        '''训练后两层的参数'''
        unfreeze_layers = ['layer.10','layer.11','bert.pooler','out.']
        for name ,param in self.bert.named_parameters():
            param.requires_grad = False
            for ele in unfreeze_layers:
                if ele in name:
                    param.requires_grad = True
                    #加快遍历的速度
                    break
        self.dropout = nn.Dropout(dropout)
        #309个角色信息
        self.role = nn.Embedding(2, input_size)
        
        self.utterance_gru = nn.GRU(input_size=input_size*2
                          , num_layers=num_layer
                          , hidden_size=num_hidden
                          , bidirectional=True
                          , batch_first = True)
        
#         self.globe_emotion_gru = nn.GRU(input_size=2*num_hidden
#                                  , num_layers=num_layer
#                                  , hidden_size=num_hidden)
        
        self.local_emotion_gru = nn.GRU(input_size=input_size*2
                                 , num_layers=num_layer
                                 , hidden_size=num_hidden
                                 , batch_first =True)
        #聚合情感信息
        self.liner_one = nn.Linear(num_hidden*5, num_hidden)
        #self.liner_two = nn.Linear(num_hidden, 5)
        self.relu = nn.ReLU()

        
    def forward(self, roles, dialog_token_ids, dialog_token_amask, valid_length):
        '''
        #input_x = [roles, input_ids, attention_mask, labels(valid_length)]
        '''
        batch_size = roles.shape[0]
        num_utterance = dialog_token_ids.shape[1]
        #batch_size, max_utterance
        roles = roles
        #batch_size, max_utterance, 128=hidden
        input_ids = dialog_token_ids
        #batch_size, max_utterance, 128=hidden
        attention_mask = dialog_token_amask
        #batch_size * max_utterance, 128=hidden
        input_ids = input_ids.reshape(-1, input_ids.shape[2])
        attention_mask = attention_mask.reshape(-1, attention_mask.shape[2])
        
        #batch_size * max_utterance, 768=hidden
        d_logit = self.bert(input_ids = input_ids
                            ,attention_mask = attention_mask)['pooler_output']
        
        #batch_size, max_utterance, 768=hidden
        d_logit = d_logit.reshape(batch_size, -1, d_logit.shape[-1])
        #融合角色信息
        #batch_size, max_utterance, 768*2=hidden*2
        d_logit = torch.cat([self.role(roles), d_logit], dim=-1)
        
        d_logit_one = pack_padded_sequence(d_logit, lengths=valid_length.cpu(), 
                             batch_first=True, 
                             enforce_sorted=False)
        #num_layer*2, batch_size, num_hidden
        #假设输出的hidden没有改变batch的顺序哈，后面还要验证一下
        utterance_feature, h_n = self.utterance_gru(d_logit_one)
        #batch_size, max_utterance, num_hidden*2
        #分布式的时候total_length很重要
        utterance_feature = pad_packed_sequence(utterance_feature, 
                                                batch_first = True, 
                                                total_length=num_utterance)[0]
        #batch_size, num_hidden*2
        globel_emotion = torch.cat([h_n[-1], h_n[-2]],dim=-1)
        #print(globel_emotion.shape)
        #debug 没有赋值导致维度没有更新
        globel_emotion = globel_emotion.unsqueeze(1)
        #batch_size, max_utterance, num_hidden*2
        globel_emotion = globel_emotion.repeat(1,utterance_feature.shape[1],1)
        #print(utterance_feature.shape[1])
            
        #batch_size, max_utterance+窗口-1, 768*2=hidden*2
        pad = self.window_size-1
        extend_zero_d_logit = torch.cat([torch.zeros((batch_size, pad, self.input_size*2)).to(d_logit), d_logit], dim=1)
        
        context = []
        for i in range(num_utterance):
            context.append(extend_zero_d_logit[:,i:i+self.window_size-1,:])
        #batch_size, max_utterance, 窗口-1 , 768*2=hidden*2
        extend_zero_d_logit = torch.stack(context, dim=1)
        
        extend_zero_d_logit = extend_zero_d_logit.reshape(-1, extend_zero_d_logit.shape[2], extend_zero_d_logit.shape[3])
        #num_layer, batch_size*max_utterance, num_hidden
        local_emotion = self.local_emotion_gru(extend_zero_d_logit)[1]
        #batch_size*max_utterance, num_hidden
        local_emotion = local_emotion[-1]
        #batch_size, max_utterance, num_hidden
        local_emotion = local_emotion.reshape(batch_size, num_utterance, -1)
#         print(utterance_feature.shape)
#         print(local_emotion.shape)
#         print(globel_emotion.shape)
        #batch_size, max_utterance, num_hidden*2+num_hidden+num_hidden*2
        utterance_with_emotion = torch.cat([utterance_feature, local_emotion, globel_emotion], dim=-1)
        #dialog_label = self.liner_two(self.dropout(self.relu(self.liner_one(utterance_with_emotion))))
        
        dialog_label = self.dropout(self.relu(self.liner_one(utterance_with_emotion)))
        #batch_size, max_utterance, 256=num_hiddden
        return dialog_label
