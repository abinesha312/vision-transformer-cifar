# vision-transformer-cifar/models/utils.py
import torch
from torch import nn

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_checkpoint(model, ckpt_path):
    state_dict = torch.load(ckpt_path, map_location='cpu')
    model.load_state_dict(state_dict['model'])
    print(f"Loaded checkpoint from {ckpt_path}")
    return model

def label_smoothing_loss(preds, targets, epsilon=0.1):
    n_classes = preds.size(-1)
    one_hot = torch.zeros_like(preds).scatter_(1, targets.unsqueeze(1), 1)
    smoothed_labels = one_hot * (1 - epsilon) + epsilon / n_classes
    return (-smoothed_labels * F.log_softmax(preds, dim=-1)).sum(dim=1).mean()
