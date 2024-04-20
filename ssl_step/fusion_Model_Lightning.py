import lightning.pytorch as pl
import torch
from transformers import get_linear_schedule_with_warmup
from config import config
from fusion_Model import fusion_Model
from sklearn.metrics import f1_score, precision_score
import numpy as np
from collections import OrderedDict

torch.set_float32_matmul_precision('medium')

def myloss(pre, label):
    loss_ce = config.CE_loss(pre['logits'], label)
    
#     q = pre['profile']
#     k = q+pre['dialog_emotion']+pre['dialog_professional']+pre['query']
#     loss_cl = config.InfoNCE_loss(q, k)
    
#     loss = loss_ce + config.loss_weight * loss_cl
    loss = loss_ce
    return loss


class fusion_Model_Lightning(pl.LightningModule):
    def __init__(self, epochs, learning_rate, dropout,
                batch_size, warmup_ratio, val_step, accumulation_steps,
                temperature, loss_weight):
        super(fusion_Model_Lightning, self).__init__()
        self.fusion_Model = fusion_Model()
        
        
        #断点训练
        # model_checkpoint = torch.load('/home/jovyan/work/ckpt/model-epoch=8-accuracy=0.9115.ckpt',map_location='cpu')
        # new_state_dict = OrderedDict()
        # for k , v in model_checkpoint['state_dict'].items():
        #     name = k[13:]
        #     new_state_dict[name] = v
        # self.fusion_Model.load_state_dict(new_state_dict)
        
        self.save_hyperparameters()
        self.validation_step_pres = []
        self.validation_step_labels = []
        
        self.train_loss = []
        self.val_loss = []
    
    def forward(self, x):
        self.model(x)
        
    def training_step(self, batch, batch_idx):
        label = batch['label']
        output = self.fusion_Model(batch)
        loss = myloss(output, label)
        
        self.train_loss.append(loss.item())
        
        return loss

    def validation_step(self, batch, batch_idx):
        label = batch['label']
        output = self.fusion_Model(batch)
        loss = myloss(output, label)
        self.val_loss.append(loss.item())
        
        logits = output['logits'].argmax(-1).detach().cpu().numpy().tolist()
        label = batch['label'].tolist() 
        self.validation_step_pres.extend(logits)
        self.validation_step_labels.extend(label)
        
        
    def on_validation_epoch_end(self):
        
        self.log('accuracy', config.accuracy_score(self.validation_step_labels,
        self.validation_step_pres),  prog_bar=True, sync_dist=True)
        
        self.log('f1_score', f1_score(self.validation_step_labels,
        self.validation_step_pres),  sync_dist=True)
        
        self.log('precision_score', precision_score(self.validation_step_labels,
        self.validation_step_pres), sync_dist=True)
        #print(self.)
        self.log('train_loss', np.mean(self.train_loss), sync_dist=True)
        self.log('val_loss', np.mean(self.val_loss), sync_dist=True)
        self.validation_step_pres.clear()  # free memory
        self.validation_step_labels.clear()  # free memory
        
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
        
    def on_test_epoch_end(self):
        self.on_validation_epoch_end()
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.fusion_Model.parameters()), lr=config.learning_rate)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.hparams.warmup_ratio,
            num_training_steps=self.trainer.estimated_stepping_batches,
        )
        return [optimizer], [scheduler]
