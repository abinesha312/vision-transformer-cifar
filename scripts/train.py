# vision-transformer-cifar/scripts/train.py
import hydra
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger

from models.hybrid_vit import HybridViT
from data.cifar10 import CIFAR10DataModule

class LitModel(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = HybridViT(**cfg.model)
        self.criterion = torch.nn.CrossEntropyLoss(label_smoothing=cfg.training.label_smoothing)
        self.train_acc = pl.metrics.Accuracy()
        self.val_acc = pl.metrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.train_acc(logits.softmax(-1), y)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.train_acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        self.val_acc(logits.softmax(-1), y)
        self.log('val_loss', loss, prog_bar=True)
        self.log('val_acc', self.val_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.cfg.training.lr,
            weight_decay=self.cfg.training.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.cfg.training.max_epochs
        )
        return [optimizer], [scheduler]

@hydra.main(config_path="../configs", config_name="default")
def main(cfg):
    pl.seed_everything(42)
    
    dm = CIFAR10DataModule(**cfg.data)
    model = LitModel(cfg)
    
    logger = WandbLogger(project=cfg.logging.project, name=cfg.logging.name)
    
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=cfg.training.max_epochs,
        accelerator=cfg.training.accelerator,
        devices=cfg.training.devices,
        precision=cfg.training.precision,
        gradient_clip_val=cfg.training.gradient_clip_val,
        deterministic=True
    )
    
    trainer.fit(model, dm)

if __name__ == "__main__":
    main()
