# vision-transformer-cifar/data/cifar10.py
import torch
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader, random_split
import pytorch_lightning as pl
from torchvision import transforms as T
from .augmentations import MixUp, CutMix, AutoAugment

class CIFAR10DataModule(pl.LightningModule):
    def __init__(self, batch_size=128, augment=True, num_workers=4):
        super().__init__()
        self.batch_size = batch_size
        self.augment = augment
        self.num_workers = num_workers
        self.num_classes = 10

        # Base transformations
        self.base_transform = T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
        ])

        # Augmentation policy
        self.train_transform = T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            AutoAugment() if augment else T.Lambda(lambda x: x),
            self.base_transform
        ])

    def prepare_data(self):
        CIFAR10(root='./data', train=True, download=True)
        CIFAR10(root='./data', train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            cifar_full = CIFAR10(root='./data', train=True, transform=self.train_transform)
            self.train_ds, self.val_ds = random_split(cifar_full, [45000, 5000])
        
        if stage == 'test' or stage is None:
            self.test_ds = CIFAR10(root='./data', train=False, transform=self.base_transform)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, 
                         shuffle=True, num_workers=self.num_workers,
                         pin_memory=True, persistent_workers=True)

    def val_dataloader(self):
        return DataLoader(self.val_ds, batch_size=self.batch_size,
                         num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_ds, batch_size=self.batch_size,
                         num_workers=self.num_workers, pin_memory=True)

    def on_after_batch_transfer(self, batch, dataloader_idx):
        if self.trainer.training and self.augment:
            batch = MixUp()(*batch)
            batch = CutMix()(*batch)
        return batch
