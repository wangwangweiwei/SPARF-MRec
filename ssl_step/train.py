import lightning.pytorch as pl
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from fusion_Model_Lightning import fusion_Model_Lightning
from myemotionalBatch import myemotionalBatch
from torch.utils.data import DataLoader
from config import config

def main():
    pl.seed_everything(42, workers=True)

    train_data = myemotionalBatch('train')
    dev_data = myemotionalBatch('valid')

    train_dataloader = DataLoader(train_data, batch_size=config.batch_size, shuffle=True, collate_fn=train_data.collate_fn, num_workers=16)

    dev_dataloader = DataLoader(dev_data, batch_size=config.batch_size, collate_fn=dev_data.collate_fn, num_workers=16)
    del train_data, dev_data

    model_L = fusion_Model_Lightning(config.epochs, config.learning_rate, config.dropout,
    config.batch_size, config.warmup_ratio, config.val_step, config.accumulation_steps,
    config.temperature, config.loss_weight)


    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor='accuracy',
        dirpath='/home/jovyan/work/train_step_two/ckpt/',
        filename='model-{epoch}-{accuracy:.4f}',
        save_top_k = 2,
        mode='max')

    early_stop_callback = EarlyStopping(
        monitor="accuracy", 
        min_delta=0.00, 
        patience=3, 
        verbose=False, 
        mode="max")

    trainer = pl.Trainer(max_epochs=30, 
                         precision="16-mixed",
                        callbacks=[checkpoint_callback, early_stop_callback],
                        accumulate_grad_batches = config.accumulation_steps
                        )



    trainer.fit(model = model_L, 
                train_dataloaders=train_dataloader, val_dataloaders=dev_dataloader
               )
    
if __name__=='__main__':
    main()
