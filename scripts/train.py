# Updated train.py with ViT improvements for CIFAR-10
import sys
import os
import math
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import numpy as np
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import matplotlib.pyplot as plt
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# 1. Improved CNN-ViT Hybrid Architecture
class HybridViT(nn.Module):
    def __init__(self, 
                 image_size=32,
                 patch_size=4,
                 num_classes=10,
                 dim=192,
                 depth=6,
                 heads=8,
                 mlp_dim=384,
                 dropout=0.1,
                 sd_prob=0.1):  # Reduced stochastic depth for small dataset
        super().__init__()
        
        # Enhanced CNN stem for better feature extraction
        self.cnn_backbone = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),
        )
        
        # Vision Transformer components
        num_patches = (image_size//2)**2 // (patch_size**2)
        self.patch_embed = nn.Conv2d(128, dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim) * 0.02)
        
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, heads, mlp_dim, dropout, sd_prob)
            for _ in range(depth)
        ])
        
        self.norm = nn.LayerNorm(dim)
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        )
        
        # Proper initialization
        self.initialize_weights()

    def initialize_weights(self):
        # Initialize patch embedding like a CNN layer
        nn.init.kaiming_normal_(self.patch_embed.weight)
        
        # Initialize position embeddings (smaller scale for stability)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize transformer blocks
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x, return_attn=False):
        # CNN Feature Extraction
        x = self.cnn_backbone(x)
        
        # Patch Embedding
        x = self.patch_embed(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        
        # Add CLS Token
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = x + self.pos_embed
        
        # Transformer Blocks
        attn_maps = []
        for block in self.blocks:
            x, attn = block(x)
            attn_maps.append(attn)
        
        # Classification
        x = self.norm(x)
        cls_out = x[:, 0]
        logits = self.mlp_head(cls_out)
        
        return (logits, torch.stack(attn_maps)) if return_attn else logits

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, mlp_dim, dropout, sd_prob):
        super().__init__()
        self.sd_prob = sd_prob
        self.ln1 = nn.LayerNorm(dim)
        self.self_attn = nn.MultiheadAttention(dim, heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        # Stochastic Depth
        if self.training and torch.rand(1).item() < self.sd_prob:
            return x, None
        
        # Self Attention with Pre-Norm
        identity = x
        x_norm = self.ln1(x)
        attn_output, attn_weights = self.self_attn(x_norm, x_norm, x_norm)
        x = identity + attn_output
        
        # MLP with Pre-Norm
        identity = x
        x = self.ln2(x)
        x = identity + self.mlp(x)
        
        return x, attn_weights

# 2. Data Augmentation Functions
class Cutmix(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, batch):
        images, labels = batch
        
        # Generate mixed sample
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        # Get bounding box coordinates
        bbx1, bby1, bbx2, bby2 = self._rand_bbox(images.size(), lam)
        images[:, :, bbx1:bbx2, bby1:bby2] = images[index, :, bbx1:bbx2, bby1:bby2]
        
        # Adjust lambda to exactly match pixel ratio
        lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (images.size()[-1] * images.size()[-2]))
        
        # Create target
        targets_a, targets_b = labels, labels[index]
        return images, (targets_a, targets_b, lam)
    
    def _rand_bbox(self, size, lam):
        W, H = size[2], size[3]
        cut_rat = np.sqrt(1. - lam)
        cut_w = np.int64(W * cut_rat)
        cut_h = np.int64(H * cut_rat)
        
        # Uniform
        cx = np.random.randint(W)
        cy = np.random.randint(H)
        
        bbx1 = np.clip(cx - cut_w // 2, 0, W)
        bby1 = np.clip(cy - cut_h // 2, 0, H)
        bbx2 = np.clip(cx + cut_w // 2, 0, W)
        bby2 = np.clip(cy + cut_h // 2, 0, H)
        
        return bbx1, bby1, bbx2, bby2

class Mixup(object):
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        
    def __call__(self, batch):
        images, labels = batch
        
        # Generate mixed sample
        lam = np.random.beta(self.alpha, self.alpha)
        batch_size = images.size(0)
        index = torch.randperm(batch_size)
        
        mixed_images = lam * images + (1 - lam) * images[index]
        targets_a, targets_b = labels, labels[index]
        
        return mixed_images, (targets_a, targets_b, lam)

# 3. Training Function with Learning Rate Warmup
def train_model(model, trainloader, testloader, num_epochs=100, warmup_epochs=5):
    # Device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Criterion with label smoothing
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    # Optimizer
    optimizer = AdamW(
        model.parameters(),
        lr=5e-4,  # Lower initial learning rate
        weight_decay=0.05,  # Higher weight decay for regularization
        betas=(0.9, 0.999)
    )
    
    # Learning rate scheduler with warmup
    base_scheduler = CosineAnnealingLR(optimizer, T_max=num_epochs, eta_min=1e-6)
    
    # Data augmentation
    cutmix = Cutmix(alpha=1.0)
    mixup = Mixup(alpha=0.8)
    
    # Training stats
    train_losses = []
    val_accuracies = []
    best_val_acc = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Apply warmup for first few epochs
        if epoch < warmup_epochs:
            lr_scale = min(1.0, (epoch + 1) / warmup_epochs)
            for pg in optimizer.param_groups:
                pg['lr'] = lr_scale * 5e-4
        
        # Progress bar
        pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Apply Cutmix or Mixup randomly
            r = np.random.rand(1)
            if r < 0.5:
                inputs, (targets_a, targets_b, lam) = cutmix((inputs, targets))
                aug_type = "cutmix"
            else:
                inputs, (targets_a, targets_b, lam) = mixup((inputs, targets))
                aug_type = "mixup"
            
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Mixed loss
            loss = lam * criterion(outputs, targets_a) + (1 - lam) * criterion(outputs, targets_b)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            # Stats
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            
            # For progress display
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({
                'loss': f"{avg_loss:.3f}",
                'aug': aug_type,
                'lr': f"{optimizer.param_groups[0]['lr']:.6f}"
            })
        
        # Adjust learning rate
        if epoch >= warmup_epochs:
            base_scheduler.step()
            
        # Validation
        model.eval()
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in testloader:
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = model(inputs)
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        val_accuracies.append(val_acc)
        train_losses.append(avg_loss)
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train Loss: {avg_loss:.3f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'checkpoints/best_model.pth')
            print(f"New best model saved! Validation Accuracy: {val_acc:.2f}%")
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.savefig('Results/training_curves.png')
    
    return model, train_losses, val_accuracies

# Main function
def main():
    # Create directories
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('Results', exist_ok=True)
    
    # Set seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Strong data augmentation for CIFAR10
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(num_ops=2, magnitude=9),  # RandAugment
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    # Dataset and dataloader
    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=128, shuffle=True, num_workers=4, pin_memory=True)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=4, pin_memory=True)
    
    # Create model
    model = HybridViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=192,
        depth=6,
        heads=8,
        mlp_dim=384,
        dropout=0.1,
        sd_prob=0.1
    )
    
    # Train model
    model, train_losses, val_accuracies = train_model(
        model, 
        trainloader, 
        testloader, 
        num_epochs=100,  # Longer training
        warmup_epochs=5   # Learning rate warmup
    )
    
    # Final model save
    torch.save({
        'model_state_dict': model.state_dict(),
        'val_acc': val_accuracies[-1],
    }, 'checkpoints/final_model.pth')
    
    print(f"Final validation accuracy: {val_accuracies[-1]:.2f}%")
    print("Training completed!")

if __name__ == "__main__":
    main()


# # vision-transformer-cifar/scripts/train.py
# import sys
# import os
# # Add the project root to Python path
# sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# import torch
# import torchvision
# import torchvision.transforms as transforms
# import hydra
# from omegaconf import OmegaConf
# import numpy as np
# from tqdm import tqdm
# import matplotlib.pyplot as plt
# from datetime import datetime
# import multiprocessing
# from typing import Dict, Any, Optional, List

# from models.hybrid_vit import HybridViT

# # Get optimal CPU count for data loading
# CPU_COUNT = max(2, multiprocessing.cpu_count() - 1)

# def create_data_loaders(batch_size=32, num_workers=4, augment=True):
#     """Create CIFAR-10 data loaders with optional augmentation"""
#     # Base transforms
#     transform_test = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#     ])
    
#     # Training transforms with optional augmentation
#     if augment:
#         transform_train = transforms.Compose([
#             transforms.RandomCrop(32, padding=4),
#             transforms.RandomHorizontalFlip(),
#             transforms.ToTensor(),
#             transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
#         ])
#     else:
#         transform_train = transform_test
    
#     # Create datasets
#     trainset = torchvision.datasets.CIFAR10(
#         root='./data', train=True, download=True, transform=transform_train)
#     testset = torchvision.datasets.CIFAR10(
#         root='./data', train=False, download=True, transform=transform_test)
    
#     # Create data loaders
#     trainloader = torch.utils.data.DataLoader(
#         trainset, batch_size=batch_size, shuffle=True, 
#         num_workers=num_workers, pin_memory=True)
#     testloader = torch.utils.data.DataLoader(
#         testset, batch_size=batch_size, shuffle=False, 
#         num_workers=num_workers, pin_memory=True)
    
#     return trainloader, testloader

# def create_model(model_config):
#     """Create HybridViT model from config"""
#     # Default model parameters
#     model_params = {
#         'image_size': 32,
#         'patch_size': 4,
#         'num_classes': 10,
#         'dim': 192,
#         'depth': 6,
#         'heads': 6,
#         'mlp_dim': 384,
#         'dropout': 0.1,
#         'sd_prob': 0.2
#     }
    
#     # Update with config values if they exist
#     if model_config:
#         for k, v in model_config.items():
#             model_params[k] = v
    
#     return HybridViT(**model_params)

# def train_model(model, trainloader, testloader, config, device='cpu'):
#     """Train the model with standard PyTorch"""
#     # Training parameters
#     epochs = config.training.max_epochs if hasattr(config, 'training') and hasattr(config.training, 'max_epochs') else 5
#     lr = config.training.lr if hasattr(config, 'training') and hasattr(config.training, 'lr') else 3e-4
#     weight_decay = config.training.weight_decay if hasattr(config, 'training') and hasattr(config.training, 'weight_decay') else 0.05
    
#     # Create optimizer and criterion
#     optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
#     criterion = torch.nn.CrossEntropyLoss()
#     scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
#     # Create folders for checkpoints and Results
#     os.makedirs('checkpoints', exist_ok=True)
#     os.makedirs('Results', exist_ok=True)
    
#     # Training and validation metrics
#     best_acc = 0.0
#     train_losses = []
#     val_accs = []
    
#     # Log start of training
#     print(f"\n=== Starting training for {epochs} epochs ===")
#     print(f"Training on {device} with model size: {sum(p.numel() for p in model.parameters() if p.requires_grad):,} parameters")
    
#     # Training loop
#     for epoch in range(epochs):
#         model.train()
#         running_loss = 0.0
#         correct = 0
#         total = 0
        
#         # Progress bar for training
#         pbar = tqdm(trainloader, desc=f"Epoch {epoch+1}/{epochs}")
#         for batch_idx, (inputs, targets) in enumerate(pbar):
#             inputs, targets = inputs.to(device), targets.to(device)
            
#             # Forward pass
#             optimizer.zero_grad()
#             outputs = model(inputs)
#             loss = criterion(outputs, targets)
            
#             # Backward pass
#             loss.backward()
#             optimizer.step()
            
#             # Update metrics
#             running_loss += loss.item()
#             _, predicted = outputs.max(1)
#             total += targets.size(0)
#             correct += predicted.eq(targets).sum().item()
            
#             # Update progress bar
#             avg_loss = running_loss / (batch_idx + 1)
#             train_acc = 100. * correct / total
#             pbar.set_postfix({'loss': f"{avg_loss:.3f}", 'acc': f"{train_acc:.2f}%"})
        
#         # Save training metrics
#         train_losses.append(avg_loss)
        
#         # Validation
#         model.eval()
#         val_correct = 0
#         val_total = 0
        
#         with torch.no_grad():
#             for inputs, targets in tqdm(testloader, desc="Validating"):
#                 inputs, targets = inputs.to(device), targets.to(device)
#                 outputs = model(inputs)
#                 _, predicted = outputs.max(1)
#                 val_total += targets.size(0)
#                 val_correct += predicted.eq(targets).sum().item()
        
#         # Calculate validation accuracy
#         val_acc = 100. * val_correct / val_total
#         val_accs.append(val_acc)
        
#         # Print epoch summary
#         print(f"Epoch {epoch+1}/{epochs} - Train Loss: {avg_loss:.3f}, Train Acc: {train_acc:.2f}%, Val Acc: {val_acc:.2f}%")
        
#         # Save best model
#         if val_acc > best_acc:
#             best_acc = val_acc
#             torch.save({
#                 'epoch': epoch,
#                 'model_state_dict': model.state_dict(),
#                 'optimizer_state_dict': optimizer.state_dict(),
#                 'val_acc': val_acc,
#             }, 'checkpoints/best_model.pth')
#             print(f"New best model saved! Validation Accuracy: {val_acc:.2f}%")
        
#         # Update learning rate
#         scheduler.step()
    
#     # Plot training curves
#     plt.figure(figsize=(12, 5))
#     plt.subplot(1, 2, 1)
#     plt.plot(train_losses)
#     plt.title('Training Loss')
#     plt.xlabel('Epoch')
#     plt.ylabel('Loss')
    
#     plt.subplot(1, 2, 2)
#     plt.plot(val_accs)
#     plt.title('Validation Accuracy')
#     plt.xlabel('Epoch')
#     plt.ylabel('Accuracy (%)')
    
#     timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
#     plt.savefig(f'Results/training_curves_{timestamp}.png')
    
#     # Final model save
#     torch.save({
#         'epoch': epochs,
#         'model_state_dict': model.state_dict(),
#         'val_acc': val_accs[-1],
#     }, f'checkpoints/final_model_{timestamp}.pth')
    
#     print("\n=== Training completed successfully! ===")
#     print(f"Best validation accuracy: {best_acc:.2f}%")
#     print(f"Final validation accuracy: {val_accs[-1]:.2f}%")
#     print(f"Models saved in ./checkpoints/")
#     print(f"Training curves saved in ./Results/")
    
#     return model

# @hydra.main(config_path="../configs", config_name="default", version_base=None)
# def main(cfg):
#     # Set random seed for reproducibility
#     torch.manual_seed(42)
#     np.random.seed(42)
    
#     # Apply quick test settings if enabled
#     if os.environ.get("QUICK_TEST", "0") == "1":
#         print("Running quick test mode (5 epochs)")
        
#         # Ensure training config exists
#         if not hasattr(cfg, 'training'):
#             cfg.training = OmegaConf.create({})
        
#         # Set quick training parameters
#         cfg.training.max_epochs = 5
        
#         # Ensure data config exists
#         if not hasattr(cfg, 'data'):
#             cfg.data = OmegaConf.create({})
            
#         # Set reduced batch size for quick testing
#         cfg.data.batch_size = 16
    
#     # Print the active configuration
#     print("\n=== Active Configuration ===")
#     print(OmegaConf.to_yaml(cfg))
#     print("===========================\n")
    
#     # Data loading parameters
#     batch_size = cfg.data.batch_size if hasattr(cfg, 'data') and hasattr(cfg.data, 'batch_size') else 32
#     num_workers = cfg.data.num_workers if hasattr(cfg, 'data') and hasattr(cfg.data, 'num_workers') else CPU_COUNT
#     augment = cfg.data.augment if hasattr(cfg, 'data') and hasattr(cfg.data, 'augment') else True
    
#     # Create data loaders
#     print("Creating data loaders...")
#     trainloader, testloader = create_data_loaders(
#         batch_size=batch_size,
#         num_workers=num_workers,
#         augment=augment
#     )
    
#     # Create model
#     print("Creating model...")
#     model = create_model(cfg.model if hasattr(cfg, 'model') else None)
    
#     # Determine device
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     model = model.to(device)
    
#     # Train model
#     train_model(model, trainloader, testloader, cfg, device=device)
    
#     return 0

# if __name__ == "__main__":
#     main()
