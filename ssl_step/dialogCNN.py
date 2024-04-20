import torch
import torch.nn as nn

class  dialogCNN(nn.Module):
    def __init__(self, num_hidden=256):
        super(dialogCNN, self).__init__()
        self.num_channerls = [128, 64, 64]
        self.kernel_sizes = [2, 3, 4]
        self.dropout = nn.Dropout(0.5)
        self.decoder = nn.Linear(sum(self.num_channerls), 768)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.relu = nn.ReLU()
        self.convs = nn.ModuleList()
        for c, k in zip(self.num_channerls, self.kernel_sizes):
            self.convs.append(nn.Conv1d(num_hidden, c, k))

    def forward(self, x):
 
        dialog_embedding = x.permute(0,2,1)
        encoding = torch.cat([
                    torch.squeeze(self.relu(self.pool(conv(dialog_embedding))), dim=-1)
                    for conv in self.convs], dim=1)
        #batch_size, 768
        output = self.decoder(encoding)
        
        
        return {"features": output}
