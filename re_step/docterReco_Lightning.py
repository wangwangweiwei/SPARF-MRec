from docterReco_v3 import docterReco_v3
import torch
import numpy as np
import lightning.pytorch as pl


def weighted_class_bceloss(output, target, weights=None):
    if weights is not None:
        assert len(weights) == 2

        loss = weights[1] * target * torch.log(torch.clamp(output, min=1e-15, max=1)) + \
               weights[0] * (1 - target) * torch.log(torch.clamp(1 - output, min=1e-15, max=1))

    else:
        loss = target * torch.log(output) + (1 - target) * torch.log(1 - output)

    return torch.neg(torch.mean(loss))


class docterReco_Lightning(pl.LightningModule):
    
    def __init__(self, in_size, hidden_size, dropout, 
                 head_num, seed, batch_size, patience, 
                 learning_rate, dr_dialog_sample, neg_sample, warmup_ratio):
        super(docterReco_Lightning, self).__init__()
        self.dr_dialog_sample = dr_dialog_sample
        self.learning_rate = learning_rate
        self.patience = patience

        self.model = docterReco_v3(in_size, 
                                hidden_size, 
                                dropout, 
                                head_num)
        
#         model_checkpoint = torch.load('ckpt/model-epoch=99-val_loss=1.0546.ckpt')
        
#         new_state_dict = OrderedDict()
#         for k , v in model_checkpoint['state_dict'].items():
#             name = k[6:]
#             new_state_dict[name] = v
#         self.model.load_state_dict(new_state_dict)
        
        self.save_hyperparameters()
        
        self.weights = torch.Tensor([1, 6])
        
        self.train_loss = []
        self.val_loss = []
        
    
    def forward(self, x):
        self.model(x, self.dr_dialog_sample)
    
    def training_step(self, batch, batch_idx):
        label = batch[1]
        logit = self.model(batch[0], self.dr_dialog_sample)
        
        loss_docter = weighted_class_bceloss(logit, label.reshape(-1, 1), self.weights)
        #loss_docter = weighted_class_bceloss(logit, label.reshape(-1, 1))
       
        loss = loss_docter
        
        self.train_loss.append(loss.item())
        return loss
        
    def validation_step(self, batch, batch_idx):
        label = batch[1]
        logit = self.model(batch[0], self.dr_dialog_sample)
        
        loss_docter = weighted_class_bceloss(logit, label.reshape(-1, 1), self.weights)
        #loss_docter = weighted_class_bceloss(logit, label.reshape(-1, 1))
        loss = loss_docter
        
        self.val_loss.append(loss.item())
        
    def on_validation_epoch_end(self):
        train_loss = np.mean(self.train_loss)
        valid_loss = np.mean(self.val_loss)
        
        self.log('val_loss', valid_loss, prog_bar=True, sync_dist=True)
        
        values = {'train_loss': train_loss, 'valid_loss': valid_loss}
        self.log_dict(values, sync_dist=True)
    
    def test_step(self, batch, batch_idx):
        self.validation_step(batch, batch_idx)
    
    def on_test_epoch_end(self):
        self.on_validation_epoch_end()
    
#     def predict_step(self, batch, batch_idx, dataloader_idx=0):
#         # this calls forward
#         return self(batch)
    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        # this calls forward
        return self.model(batch[0], self.dr_dialog_sample)
        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()), 
                                     lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                               mode="min", 
                                                               factor=0.1, 
                                                               patience=3, 
                                                               min_lr=1e-9)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}
        # scheduler = get_linear_schedule_with_warmup(
        #     optimizer,
        #     num_warmup_steps=self.hparams.warmup_ratio,
        #     num_training_steps=self.trainer.estimated_stepping_batches,
        # )
        # return [optimizer], [scheduler]
