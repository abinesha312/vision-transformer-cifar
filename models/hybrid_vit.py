import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, return_attn=False):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        out = self.to_out(out)
        
        if return_attn:
            return out, attn
        return out

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff = FeedForward(dim, mlp_dim, dropout=dropout)

    def forward(self, x, return_attn=False):
        if return_attn:
            attn_out, attn = self.attn(self.norm1(x), return_attn=True)
            x = x + attn_out
            x = x + self.ff(self.norm2(x))
            return x, attn
        else:
            x = x + self.attn(self.norm1(x))
            x = x + self.ff(self.norm2(x))
            return x

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(TransformerBlock(dim, heads, dim_head, mlp_dim, dropout))

    def forward(self, x, return_attn=False):
        attns = []
        for layer in self.layers:
            if return_attn:
                x, attn = layer(x, return_attn=True)
                attns.append(attn)
            else:
                x = layer(x)
        
        if return_attn:
            return x, attns
        return x

class CNNBackbone(nn.Module):
    def __init__(self, in_channels=3, channels=[32, 64]):
        super().__init__()
        layers = []
        prev_channels = in_channels
        
        for ch in channels:
            layers.extend([
                nn.Conv2d(prev_channels, ch, kernel_size=3, padding=1),
                nn.BatchNorm2d(ch),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            ])
            prev_channels = ch
            
        self.cnn_backbone = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.cnn_backbone(x)

class HybridViT(nn.Module):
    def __init__(
        self, 
        image_size=32, 
        patch_size=4, 
        num_classes=10, 
        dim=192, 
        depth=6, 
        heads=6, 
        mlp_dim=384, 
        channels=3,
        dim_head=64, 
        dropout=0.1, 
        sd_prob=0.2
    ):
        super().__init__()
        
        # CNN Backbone with channels matching checkpoint [32, 64]
        self.cnn_backbone = CNNBackbone(channels, channels=[32, 64])
        
        # Calculate feature map size after CNN backbone
        # Initial 32x32 image with two 2x2 maxpool layers becomes 8x8
        feature_size = image_size // 4
        
        # Calculate number of patches
        self.num_patches = (feature_size // patch_size) ** 2
        patch_dim = 64 * patch_size * patch_size  # 64 is the last CNN channel
        
        # Patch embedding
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_size, p2=patch_size),
            nn.Linear(patch_dim, dim),
        )
        
        # Class token and positional embedding
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches + 1, dim))
        
        # Transformer encoder
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)
        
        # Classification head
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
    def forward(self, img, return_attn=False):
        # CNN feature extraction
        x = self.cnn_backbone(img)
        
        # Patch embedding
        x = self.to_patch_embedding(x)
        b, n, _ = x.shape
        
        # Prepend class token and add position embedding
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed[:, :(n + 1)]
        
        # Transformer encoding
        if return_attn:
            x, attns = self.transformer(x, return_attn=True)
            # Classification using [CLS] token
            x = x[:, 0]
            x = self.to_latent(x)
            return self.mlp_head(x), attns
        else:
            x = self.transformer(x)
            # Classification using [CLS] token
            x = x[:, 0]
            x = self.to_latent(x)
            return self.mlp_head(x)
