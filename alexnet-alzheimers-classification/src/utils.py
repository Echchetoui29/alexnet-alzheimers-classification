import torch
import yaml
import matplotlib.pyplot as plt
import numpy as np

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def get_device():
    """Get available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def plot_sample_images(dataloader, class_names):
    """Plot sample images from dataloader"""
    images, labels = next(iter(dataloader))
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for i in range(8):
        ax = axes[i // 4, i % 4]
        image = images[i].numpy().transpose((1, 2, 0))
        # Denormalize if needed
        image = np.clip(image, 0, 1)
        ax.imshow(image)
        ax.set_title(class_names[labels[i]])
        ax.axis('off')
    plt.tight_layout()
    return fig