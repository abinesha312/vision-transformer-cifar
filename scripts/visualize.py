# vision-transformer-cifar/scripts/visualize.py
import sys
import os
# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
import matplotlib.pyplot as plt
import torchvision
import torchvision.transforms as transforms
import numpy as np

from models.hybrid_vit import HybridViT

def load_best_model():
    # Model configuration matching training settings
    model_params = {
        'image_size': 32,
        'patch_size': 4,
        'num_classes': 10,
        'dim': 192,  # Matches the trained model
        'depth': 6,
        'heads': 6,
        'mlp_dim': 384,
        'dropout': 0.1,
        'sd_prob': 0.2
    }
    
    model = HybridViT(
    dim=192,  # MUST match training configuration
    depth=6,
    heads=6,
    mlp_dim=384
)
    checkpoint_path = "checkpoints/best_model.pth"
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint with validation accuracy: {checkpoint.get('val_acc', 'N/A')}%")
    
    return model

def get_cifar10_sample(index=0):
    """Get a sample from CIFAR-10 test set"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform)
    
    # Get sample image and label
    image, label = testset[index]
    
    # Class names
    classes = ('plane', 'car', 'bird', 'cat', 'deer', 
               'dog', 'frog', 'horse', 'ship', 'truck')
    
    return image, classes[label], label

def visualize_attention(model, image, class_name):
    """Visualize attention maps with robust error handling"""
    # Ensure model is in eval mode
    model.eval()
    
    # Add batch dimension
    image_batch = image.unsqueeze(0)
    
    # Forward pass with attention
    with torch.no_grad():
        try:
            # Try to get attention maps
            logits, attention_maps = model(image_batch, return_attn=True)
            
            # Debug prints to understand attention map structure
            print(f"Attention maps shape: {attention_maps.shape if attention_maps is not None else 'None'}")
            if attention_maps is not None and attention_maps.numel() > 0:
                print(f"First layer attention shape: {attention_maps[0].shape if len(attention_maps) > 0 else 'Empty'}")
        
            # Create visualization
            fig = plt.figure(figsize=(12, 6))
            
            # Original Image
            ax = fig.add_subplot(1, 2, 1)
            img_np = image.permute(1, 2, 0).numpy()  # CHW to HWC
            ax.imshow(img_np)
            ax.set_title(f"True class: {class_name}")
            ax.axis('off')
            
            # Attention Maps (if available)
            ax = fig.add_subplot(1, 2, 2)
            
            # Check if we have valid attention maps
            if attention_maps is not None and attention_maps.numel() > 0:
                # Use first layer attention (safer)
                layer_idx = 0
                if layer_idx < len(attention_maps):
                    # Get attention from first head, focusing on CLS token
                    attn = attention_maps[layer_idx][0]  # [heads, seq_len, seq_len]
                    
                    # Check attention dimensions
                    if attn.dim() >= 2 and attn.size(0) > 0:
                        # Visualize attention from CLS token (index 0) to patches
                        if attn.size(1) > 1:  # Make sure we have enough tokens
                            cls_attn = attn[0, 1:]  # First head, CLS→patches attention
                            
                            # Reshape based on actual size, not hardcoded
                            patch_side = int(math.sqrt(cls_attn.size(0)))
                            cls_attn_reshaped = cls_attn.reshape(patch_side, patch_side)
                            
                            ax.imshow(cls_attn_reshaped.cpu(), cmap='viridis')
                            ax.set_title(f"Attention Map (Layer {layer_idx})")
                        else:
                            ax.text(0.5, 0.5, "Not enough tokens for visualization", 
                                ha='center', va='center')
                    else:
                        ax.text(0.5, 0.5, "Invalid attention dimensions", 
                            ha='center', va='center')
                else:
                    ax.text(0.5, 0.5, f"No attention for layer {layer_idx}", 
                        ha='center', va='center')
            else:
                ax.text(0.5, 0.5, "No attention maps available", 
                    ha='center', va='center')
            
            ax.axis('off')
            
            # Save visualization
            os.makedirs('visualizations', exist_ok=True)
            plt.savefig(f'visualizations/attention_sample_{class_name}.png')
            plt.close()
            
            # Print class predictions
            probs = torch.softmax(logits, dim=1)[0]
            classes = ('plane', 'car', 'bird', 'cat', 'deer', 
                    'dog', 'frog', 'horse', 'ship', 'truck')
            predicted_class = classes[torch.argmax(probs).item()]
            
            print(f"Predicted: {predicted_class} (confidence: {probs.max().item():.2f})")
            
        except Exception as e:
            print(f"Error visualizing attention: {e}")
            print("Check your model's architecture and make sure it matches the training configuration.")

if __name__ == "__main__":
    print("Loading best model...")
    
    try:
        model = load_best_model()
        
        for i in range(5):  # Visualize first five samples from test set
            print(f"Visualizing sample {i}...")
            image, class_name, _ = get_cifar10_sample(i)
            visualize_attention(model, image, class_name)
            
            print("\n")
            
    except Exception as e:
        print(f"Error occurred: {e}")
