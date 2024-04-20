import torch
import torch.nn as nn
import torch.nn.functional as F
class docterReco_v3(nn.Module):
    def __init__(self, input_size, hidden_size, dropout, num_head):
        super(docterReco_v3, self).__init__()
        
        self.encoder_layer_one = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_head, batch_first=True)
        self.encoder_layer_two = nn.TransformerEncoderLayer(d_model=input_size, nhead=num_head, batch_first=True)

        self.docter_liner_one = nn.Linear(input_size*3,hidden_size)
        self.docter_liner_two = nn.Linear(hidden_size,1)
        #self.layer_norm_one = nn.LayerNorm(hidden_size)
        self.layer_norm_one = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, x, rcd_num):
        profile = x[:, 0]
        professional_emb = x[:, 1: 1 + rcd_num]
        emotion_emb = x[:, 1+rcd_num: 2*rcd_num+1]
        query = x[:, -1]
        
        all_emb_one = torch.cat([profile.unsqueeze(1), professional_emb], dim = 1)
        all_emb_one = self.encoder_layer_one(all_emb_one)
        
        all_emb_two = torch.cat([profile.unsqueeze(1), emotion_emb], dim = 1)
        all_emb_two = self.encoder_layer_two(all_emb_two)
        
        profile_professional = all_emb_one[:, 0]
        profile_emotion = all_emb_two[:,0]
        #batch_size, 768*3
        input_x = torch.cat([profile_professional, profile_emotion, query], dim =-1)
        
        output_docter = self.sigmoid(self.docter_liner_two(self.dropout(self.relu(self.layer_norm_one(self.docter_liner_one(input_x))))))
        
        return output_docter
