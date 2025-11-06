"""
Data Preprocessing for Alzheimer MRI Dataset
Handles image resizing, data splitting, and data augmentation
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import json
from sklearn.model_selection import train_test_split

class AlzheimerMRIDataset(Dataset):
    """Custom Dataset class for Alzheimer MRI images"""
    
    def __init__(self, data_dir, transform=None, split='train'):
        """
        Args:
            data_dir (string): Directory with all the images
            transform (callable, optional): Optional transform to be applied on a sample
            split (string): 'train', 'val', or 'test'
        """
        self.data_dir = data_dir
        self.transform = transform
        self.split = split
        
        # Get class names and create mapping
        self.class_names = [d for d in os.listdir(data_dir) 
                           if os.path.isdir(os.path.join(data_dir, d))]
        self.class_names.sort()
        self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(self.class_names)}
        
        # Load image paths and labels
        self.image_paths = []
        self.labels = []
        
        for class_name in self.class_names:
            class_path = os.path.join(data_dir, class_name)
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
            
            for img_name in images:
                self.image_paths.append(os.path.join(class_path, img_name))
                self.labels.append(self.class_to_idx[class_name])
        
        print(f"{split.capitalize()} dataset: {len(self.image_paths)} images")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image
        image = Image.open(img_path).convert('RGB')
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        
        return image, label

def get_data_transforms():
    """Define data transformations for training, validation, and testing"""
    
    # AlexNet expects 224x224 input images
    mean = [0.485, 0.456, 0.406]  # ImageNet mean
    std = [0.229, 0.224, 0.225]   # ImageNet std
    
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
    
    return train_transforms, val_transforms, test_transforms

def find_dataset_folder(base_path):
    """Find the actual dataset folder within the raw directory"""
    possible_paths = [
        base_path,  # Direct structure
        os.path.join(base_path, "augmented-alzheimer-mri-dataset", "AugmentedAlzheimerDataset"),
        os.path.join(base_path, "augmented-alzheimer-mri-dataset", "OriginalDataset"),
        os.path.join(base_path, "AugmentedAlzheimerDataset"),
        os.path.join(base_path, "OriginalDataset")
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            # Check if it contains the expected class folders
            class_folders = [d for d in os.listdir(path) 
                           if os.path.isdir(os.path.join(path, d)) and 
                           any(class_name in d for class_name in ['NonDemented', 'VeryMildDemented', 'MildDemented', 'ModerateDemented'])]
            if len(class_folders) >= 2:  # At least 2 classes found
                print(f"✅ Found dataset at: {path}")
                return path
    
    return None

def create_data_loaders(data_dir, batch_size=32, val_split=0.15, test_split=0.15, random_state=42):
    """
    Create data loaders for training, validation, and testing
    
    Args:
        data_dir (str): Path to the dataset directory
        batch_size (int): Batch size for data loaders
        val_split (float): Fraction of data for validation
        test_split (float): Fraction of data for testing
        random_state (int): Random seed for reproducibility
    """
    
    # Get data transforms
    train_transforms, val_transforms, test_transforms = get_data_transforms()
    
    # Create full dataset
    full_dataset = AlzheimerMRIDataset(data_dir, transform=train_transforms, split='full')
    
    # Calculate split sizes
    dataset_size = len(full_dataset)
    test_size = int(test_split * dataset_size)
    val_size = int(val_split * dataset_size)
    train_size = dataset_size - val_size - test_size
    
    print(f"Dataset split: Train={train_size}, Val={val_size}, Test={test_size}")
    
    # Split dataset
    train_dataset, temp_dataset = random_split(
        full_dataset, [train_size, val_size + test_size],
        generator=torch.Generator().manual_seed(random_state)
    )
    
    val_dataset, test_dataset = random_split(
        temp_dataset, [val_size, test_size],
        generator=torch.Generator().manual_seed(random_state)
    )
    
    # Apply different transforms for val and test
    val_dataset.dataset.transform = val_transforms
    val_dataset.dataset.split = 'val'
    
    test_dataset.dataset.transform = test_transforms
    test_dataset.dataset.split = 'test'
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)  # num_workers=0 for Windows
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    return train_loader, val_loader, test_loader, full_dataset.class_names

def visualize_batch(dataloader, class_names, num_images=8):
    """Visualize a batch of images from dataloader"""
    images, labels = next(iter(dataloader))
    
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    images = images * std + mean
    images = torch.clamp(images, 0, 1)
    
    # Plot images
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    axes = axes.ravel()
    
    for i in range(min(num_images, len(images))):
        img = images[i].numpy().transpose((1, 2, 0))
        axes[i].imshow(img)
        axes[i].set_title(f'{class_names[labels[i]]}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig('../data/processed/preprocessed_samples.png', dpi=300, bbox_inches='tight')
    plt.show()

def main():
    """Test the data preprocessing pipeline"""
    # Set up absolute paths
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_data_path = os.path.join(project_root, "data", "raw")
    
    # Find the actual dataset
    dataset_path = find_dataset_folder(base_data_path)
    
    if not dataset_path:
        print(f"❌ ERROR: Could not find dataset in: {base_data_path}")
        print("Available folders:")
        for root, dirs, files in os.walk(base_data_path):
            print(f"  {root}")
        return
    
    print(f"Using dataset from: {dataset_path}")
    
    # Create output directory
    output_path = os.path.join(project_root, "data", "processed")
    os.makedirs(output_path, exist_ok=True)
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir=dataset_path,
        batch_size=32,
        val_split=0.15,
        test_split=0.15
    )
    
    print(f"Class names: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    
    # Test one batch
    for images, labels in train_loader:
        print(f"Batch image shape: {images.shape}")  # Should be [32, 3, 224, 224]
        print(f"Batch label shape: {labels.shape}")  # Should be [32]
        print(f"Image range: [{images.min():.3f}, {images.max():.3f}]")
        break
    
    # Visualize a batch
    visualize_batch(train_loader, class_names)
    
    print("✅ Data preprocessing pipeline test completed successfully!")

if __name__ == "__main__":
    main()