import sys
import os
import math
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import numpy as np

from models.hybrid_vit import HybridViT

def adjust_pos_embed(pos_embed, current_pos_embed):
    """Adjust position embeddings to match target shape."""
    # Extract dimensions
    old_embed = pos_embed.squeeze(0)
    new_embed = current_pos_embed.squeeze(0)
    
    # Separate class token and spatial tokens
    cls_token_old = old_embed[:1]
    spatial_tokens_old = old_embed[1:]
    
    # Calculate grid sizes
    old_grid_size = int(math.sqrt(spatial_tokens_old.shape[0]))
    new_grid_size = int(math.sqrt(new_embed.shape[0] - 1))
    
    # Reshape old spatial tokens to 2D grid for interpolation
    spatial_tokens_old = spatial_tokens_old.reshape(old_grid_size, old_grid_size, -1)
    spatial_tokens_old = spatial_tokens_old.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
    
    # Use bilinear interpolation
    spatial_tokens_new = torch.nn.functional.interpolate(
        spatial_tokens_old, 
        size=(new_grid_size, new_grid_size),
        mode='bilinear',
        align_corners=False
    )
    
    # Reshape back to token format
    spatial_tokens_new = spatial_tokens_new.squeeze(0).permute(1, 2, 0)  # [H, W, C]
    spatial_tokens_new = spatial_tokens_new.reshape(-1, spatial_tokens_new.size(-1))
    
    # Combine with class token
    new_pos_embed = torch.cat([cls_token_old, spatial_tokens_new], dim=0).unsqueeze(0)
    
    return new_pos_embed

def load_best_model():
    """Load the trained model with correct architecture."""
    # Create model with matching architecture
    model = HybridViT(
        image_size=32,
        patch_size=4,
        num_classes=10,
        dim=192,
        depth=6,
        heads=6,
        mlp_dim=384,
        dropout=0.1,
        sd_prob=0.2
    )
    
    # Load checkpoint
    checkpoint_path = "checkpoints/best_model.pth"
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint['model_state_dict']
    
    # Rename keys for compatibility
    fixed_state_dict = {}
    for k, v in state_dict.items():
        # Convert from old naming to new naming
        new_key = k
        if 'self_attn' in k:
            new_key = k.replace('self_attn', 'attn')
        if 'ln' in k:
            new_key = k.replace('ln', 'norm')
            
        fixed_state_dict[new_key] = v
    
    # Adjust position embeddings
    if 'pos_embed' in fixed_state_dict:
        if fixed_state_dict['pos_embed'].shape != model.pos_embed.shape:
            print(f"Adjusting position embeddings from {fixed_state_dict['pos_embed'].shape} to {model.pos_embed.shape}")
            fixed_state_dict['pos_embed'] = adjust_pos_embed(
                fixed_state_dict['pos_embed'], 
                model.pos_embed
            )
    
    # Load with strict=False to handle missing keys
    missing, unexpected = model.load_state_dict(fixed_state_dict, strict=False)
    
    if missing:
        print(f"Missing keys: {len(missing)} keys")
    if unexpected:
        print(f"Unexpected keys: {len(unexpected)} keys")
    
    print(f"Loaded checkpoint with validation accuracy: {checkpoint.get('val_acc', 'N/A')}%")
    return model

def get_cifar10_sample(index=0):
    """Get a sample from CIFAR-10 test set."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    
    image, label = testset[index]
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return image, classes[label], label

def visualize_attention(model, image, class_name):
    """Visualize attention maps."""
    model.eval()
    
    # Add batch dimension
    image_batch = image.unsqueeze(0).float()
    
    with torch.no_grad():
        try:
            # Forward pass with attention maps
            logits, attention_maps = model(image_batch, return_attn=True)
            
            # Create figure for visualization
            fig = plt.figure(figsize=(12, 6))
            
            # Original image
            ax1 = fig.add_subplot(1, 2, 1)
            img_np = image.permute(1, 2, 0).numpy()
            
            # Properly reshape for broadcasting to avoid dimension errors
            mean = np.array([0.4914, 0.4822, 0.4465]).reshape(1, 1, 3)
            std = np.array([0.2023, 0.1994, 0.2010]).reshape(1, 1, 3)
            img_np = img_np * std + mean
            img_np = np.clip(img_np, 0, 1)
            
            ax1.imshow(img_np)
            ax1.set_title(f"True class: {class_name}")
            ax1.axis('off')
            
            # Attention visualization
            ax2 = fig.add_subplot(1, 2, 2)
            
            if attention_maps and len(attention_maps) > 0:
                # Choose a middle layer for visualization
                layer_idx = min(len(attention_maps) - 1, 3)
                attn = attention_maps[layer_idx]
                
                # Visualize attention from CLS token to patches
                if attn.dim() == 3 and attn.size(1) > 1:
                    # Get attention from first head
                    head_idx = 0
                    cls_attn = attn[head_idx, 0, 1:]  # CLS token's attention to patches
                    
                    # Calculate grid size
                    grid_size = int(math.sqrt(cls_attn.size(0)))
                    
                    if grid_size**2 == cls_attn.size(0):
                        # Reshape to square grid
                        cls_attn = cls_attn.reshape(grid_size, grid_size)
                        ax2.imshow(cls_attn.cpu(), cmap='viridis')
                        ax2.set_title(f"Attention Map (Layer {layer_idx}, Head {head_idx})")
                    else:
                        ax2.text(0.5, 0.5, "Non-square attention grid", ha='center', va='center')
                else:
                    ax2.text(0.5, 0.5, "Invalid attention format", ha='center', va='center')
            else:
                ax2.text(0.5, 0.5, "No attention maps available", ha='center', va='center')
            
            ax2.axis('off')
            
            # Save visualization
            os.makedirs('Results', exist_ok=True)
            plt.savefig(f'Results/attention_{class_name}.png')
            plt.close()
            
            # Print prediction results
            probs = torch.softmax(logits, dim=1)[0]
            predicted_class = ('plane', 'car', 'bird', 'cat', 'deer', 
                              'dog', 'frog', 'horse', 'ship', 'truck')[torch.argmax(probs).item()]
            
            print(f"Predicted: {predicted_class} (confidence: {probs.max().item():.2f})")
            
        except Exception as e:
            print(f"Error visualizing attention: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    print("Loading best model...")
    
    try:
        model = load_best_model()
        
        for i in range(5):
            print(f"Visualizing sample {i}...")
            image, class_name, _ = get_cifar10_sample(i)
            visualize_attention(model, image, class_name)
            print("-" * 50)
            
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()
