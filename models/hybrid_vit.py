# vision-transformer-cifar/models/hybrid_vit.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

class HybridViT(nn.Module):
    def __init__(self, 
                 image_size=32,
                 patch_size=4,
                 num_classes=10,
                 dim=256,
                 depth=6,
                 heads=8,
                 mlp_dim=512,
                 dropout=0.1,
                 sd_prob=0.2):
        super().__init__()
        
        # CNN Stem
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.MaxPool2d(2)
        )
        
        # ViT Components
        self.patch_embed = nn.Conv2d(128, dim, patch_size, patch_size)
        num_patches = (image_size//4)**2 // (patch_size**2)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        
        # Transformer Blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout, sd_prob)
            for _ in range(depth)
        ])
        
        # Head
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )

    def forward(self, x, return_attn=False):
        # CNN Feature Extraction
        x = self.cnn_backbone(x)
        
        # Patch Embedding
        x = self.patch_embed(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        
        # Add CLS Token
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=x.shape[0])
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embed
        
        # Transformer Blocks
        attn_maps = []
        for blk in self.blocks:
            x, attn = blk(x)
            attn_maps.append(attn)
            
        # Classification
        cls_out = x[:, 0]
        logits = self.mlp_head(cls_out)
        
        return (logits, torch.stack(attn_maps)) if return_attn else logits

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, sd_prob):
        super().__init__()
        self.sd_prob = sd_prob
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Stochastic Depth
        if self.training and torch.rand(1) < self.sd_prob:
            return x, None
        
        # Attention
        residual = x
        x = self.norm1(x)
        x_attn, attn_weights = self.attn(x, x, x)
        x = residual + x_attn
        
        # MLP
        residual = x
        x = self.norm2(x)
        x = residual + self.mlp(x)
        
        return x, attn_weights.detach()
