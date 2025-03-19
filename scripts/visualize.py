# vision-transformer-cifar/scripts/visualize.py
import torch
import matplotlib.pyplot as plt
from models.hybrid_vit import HybridViT
from models.utils import load_checkpoint
from data.cifar10 import CIFAR10DataModule

def visualize_attention(model, datamodule, sample_idx=0):
    # Get sample
    dm = datamodule()
    dm.setup()
    sample = dm.val_dataloader().dataset[sample_idx][0].unsqueeze(0)
    
    # Forward pass
    logits, attn_maps = model(sample, return_attn=True)
    
    # Plotting
    fig = plt.figure(figsize=(16, 8))
    
    # Original Image
    ax = fig.add_subplot(1, 2, 1)
    ax.imshow(sample.squeeze().permute(1,2,0).numpy())
    ax.set_title("Original Image")
    ax.axis('off')
    
    # Attention Maps
    ax = fig.add_subplot(1, 2, 2)
    avg_attn = attn_maps.mean(dim=0)[0, :, 0, 1:].mean(0)
    ax.imshow(avg_attn.reshape(4,4).detach().cpu(), cmap='viridis')
    ax.set_title("Average Attention Map")
    ax.axis('off')
    
    plt.savefig('attention_visualization.png')
    plt.close()

if __name__ == "__main__":
    # Initialize components
    model = HybridViT()
    model = load_checkpoint(model, "path/to/checkpoint.pt")
    
    # Visualize
    visualize_attention(model, CIFAR10DataModule)
