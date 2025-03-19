# vision-transformer-cifar/data/augmentations.py
import torch
import numpy as np
from albumentations import Compose, HorizontalFlip, ShiftScaleRotate, CoarseDropout

class AutoAugment:
    def __init__(self):
        self.aug = Compose([
            HorizontalFlip(p=0.5),
            ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15),
            CoarseDropout(max_holes=1, max_height=16, max_width=16, fill_value=0.5)
        ])

    def __call__(self, img):
        img = np.array(img)
        return self.aug(image=img)['image']

class MixUp:
    def __init__(self, alpha=0.8):
        self.alpha = alpha

    def __call__(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        mixed_x = lam * x + (1 - lam) * x[index]
        y_a, y_b = y, y[index]
        return mixed_x, (y_a, y_b, lam)

class CutMix:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def __call__(self, x, y):
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = x.size(0)
        index = torch.randperm(batch_size)
        
        h, w = x.size()[2:]
        cx, cy = np.random.randint(w), np.random.randint(h)
        cut_ratio = np.sqrt(1. - lam)
        cut_w, cut_h = int(w * cut_ratio), int(h * cut_ratio)
        
        x1 = np.clip(cx - cut_w // 2, 0, w)
        x2 = np.clip(cx + cut_w // 2, 0, w)
        y1 = np.clip(cy - cut_h // 2, 0, h)
        y2 = np.clip(cy + cut_h // 2, 0, h)
        
        x[:, :, y1:y2, x1:x2] = x[index, :, y1:y2, x1:x2]
        lam = 1 - ((x2 - x1) * (y2 - y1) / (w * h))
        
        return x, (y, y[index], lam)
