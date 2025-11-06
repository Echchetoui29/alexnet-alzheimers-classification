"""
Alzheimer MRI Dataset - Exploratory Data Analysis

This script explores the Augmented Alzheimer MRI Dataset to understand 
its structure, class distribution, and image characteristics.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pandas as pd
from collections import Counter
import cv2
import json

def explore_dataset_structure(dataset_path):
    """Explore and print the dataset directory structure"""
    print("Dataset structure:")
    for root, dirs, files in os.walk(dataset_path):
        level = root.replace(dataset_path, '').count(os.sep)
        indent = ' ' * 2 * level
        print(f"{indent}{os.path.basename(root)}/")
        subindent = ' ' * 2 * (level + 1)
        for file in files[:3]:  # Show first 3 files
            print(f"{subindent}{file}")
        if len(files) > 3:
            print(f"{subindent}... and {len(files) - 3} more files")
    print()

def find_dataset_folder(dataset_path):
    """Find the actual dataset folder within the raw directory"""
    possible_paths = [
        dataset_path,  # Direct structure
        os.path.join(dataset_path, "augmented-alzheimer-mri-dataset", "AugmentedAlzheimerDataset"),
        os.path.join(dataset_path, "augmented-alzheimer-mri-dataset", "OriginalDataset"),
        os.path.join(dataset_path, "AugmentedAlzheimerDataset"),
        os.path.join(dataset_path, "OriginalDataset")
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

def analyze_class_distribution(dataset_path):
    """Analyze the distribution of images across classes"""
    class_names = [d for d in os.listdir(dataset_path) 
                   if os.path.isdir(os.path.join(dataset_path, d))]
    class_names.sort()
    
    class_distribution = {}
    image_paths = {}
    image_info = []

    print("Class Distribution:")
    print("-" * 40)

    for class_name in class_names:
        class_path = os.path.join(dataset_path, class_name)
        images = [f for f in os.listdir(class_path) 
                  if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        class_distribution[class_name] = len(images)
        image_paths[class_name] = [os.path.join(class_path, img) for img in images]
        
        # Get sample image info
        if images:
            sample_img_path = os.path.join(class_path, images[0])
            try:
                with Image.open(sample_img_path) as img:
                    image_info.append({
                        'class': class_name,
                        'size': img.size,
                        'mode': img.mode,
                        'format': img.format
                    })
            except Exception as e:
                print(f"Error reading sample image from {class_name}: {e}")
                image_info.append({
                    'class': class_name,
                    'size': 'Unknown',
                    'mode': 'Unknown',
                    'format': 'Unknown'
                })
        
        print(f"{class_name:<20}: {len(images):>6} images")

    total_images = sum(class_distribution.values())
    print("-" * 40)
    print(f"{'TOTAL':<20}: {total_images:>6} images")
    print()
    
    return class_names, class_distribution, image_paths, image_info, total_images

def plot_class_distribution(class_distribution):
    """Create visualization plots for class distribution"""
    plt.figure(figsize=(12, 6))

    # Bar plot
    plt.subplot(1, 2, 1)
    bars = plt.bar(class_distribution.keys(), class_distribution.values())
    plt.title('Class Distribution of MRI Images')
    plt.xlabel('Alzheimer Stage')
    plt.ylabel('Number of Images')
    plt.xticks(rotation=45)

    # Add value labels on bars
    for bar in bars:
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                 f'{int(height)}', ha='center', va='bottom')

    # Pie chart
    plt.subplot(1, 2, 2)
    plt.pie(class_distribution.values(), labels=class_distribution.keys(), autopct='%1.1f%%')
    plt.title('Percentage Distribution')

    plt.tight_layout()
    plt.savefig('../data/processed/class_distribution.png', dpi=300, bbox_inches='tight')
    plt.show()

def display_sample_images(class_names, image_paths, num_samples=5):
    """Display sample images from each class"""
    fig, axes = plt.subplots(len(class_names), num_samples, figsize=(15, 12))
    
    if len(class_names) == 1:
        axes = [axes]
    
    for i, class_name in enumerate(class_names):
        class_images = image_paths[class_name][:num_samples]
        
        for j, img_path in enumerate(class_images):
            try:
                img = Image.open(img_path)
                ax = axes[i][j] if len(class_names) > 1 else axes[j]
                ax.imshow(img, cmap='gray')
                ax.set_title(f'{class_name}\n{img.size}')
                ax.axis('off')
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                # Create empty plot with error message
                ax = axes[i][j] if len(class_names) > 1 else axes[j]
                ax.text(0.5, 0.5, f"Error\nloading", 
                       ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'{class_name}')
                ax.axis('off')
    
    plt.suptitle('Sample MRI Images from Each Class', fontsize=16, y=1.02)
    plt.tight_layout()
    plt.savefig('../data/processed/sample_images.png', dpi=300, bbox_inches='tight')
    plt.show()

def analyze_image_properties(class_names, image_paths):
    """Analyze image dimensions and properties"""
    print("Image Properties Analysis:")
    print("-" * 50)

    all_dimensions = []
    for class_name in class_names:
        class_images = image_paths[class_name][:20]  # Sample 20 images per class
        for img_path in class_images:
            try:
                with Image.open(img_path) as img:
                    all_dimensions.append(img.size)
            except Exception as e:
                print(f"Error analyzing {img_path}: {e}")

    if all_dimensions:
        dimensions_df = pd.DataFrame(all_dimensions, columns=['width', 'height'])
        print(dimensions_df.describe())
    else:
        print("No images could be analyzed")
        dimensions_df = pd.DataFrame(columns=['width', 'height'])
    print()
    
    return dimensions_df, all_dimensions

def check_image_consistency(all_dimensions):
    """Check for consistent image sizes across dataset"""
    if not all_dimensions:
        print("No image dimensions to analyze")
        return set()
        
    unique_sizes = set(all_dimensions)
    print(f"Unique image sizes found: {len(unique_sizes)}")
    if len(unique_sizes) <= 10:  # Only print if not too many
        for size in sorted(unique_sizes):
            count = all_dimensions.count(size)
            print(f"  {size}: {count} images")
    print()
    
    return unique_sizes

def generate_recommendations(class_distribution, total_images, unique_sizes):
    """Generate recommendations based on dataset analysis"""
    print("="*50)
    print("RECOMMENDATIONS FOR NEXT STEPS")
    print("="*50)

    if len(unique_sizes) > 1:
        print("⚠️  Images have different sizes. Need to implement resizing in preprocessing.")
    else:
        print("✅ All images have consistent size. No resizing needed.")

    if total_images < 10000:
        print("⚠️  Dataset is relatively small. Consider using data augmentation during training.")
    else:
        print("✅ Dataset size is substantial.")

    # Check class imbalance
    counts = list(class_distribution.values())
    max_count = max(counts)
    min_count = min(counts)
    imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')

    if imbalance_ratio > 3:
        print(f"⚠️  Significant class imbalance detected (ratio: {imbalance_ratio:.2f}). Consider class weighting.")
    else:
        print("✅ Classes are reasonably balanced.")

    print("\nNext steps:")
    print("1. Implement data preprocessing in src/data_preprocessing.py")
    print("2. Create AlexNet model in src/model.py") 
    print("3. Set up training pipeline in src/train.py")

def main():
    """Main function to run the exploratory data analysis"""
    # Set up paths - FIXED: Using absolute paths based on script location
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    dataset_path = os.path.join(project_root, "data", "raw")
    output_path = os.path.join(project_root, "data", "processed")

    # Create processed directory if it doesn't exist
    os.makedirs(output_path, exist_ok=True)
    
    print("="*60)
    print("ALZHEIMER MRI DATASET - EXPLORATORY DATA ANALYSIS")
    print("="*60)
    print()

    # Find the actual dataset folder
    actual_dataset_path = find_dataset_folder(dataset_path)
    
    if not actual_dataset_path:
        print(f"❌ ERROR: Could not find dataset in: {dataset_path}")
        print("Available folders:")
        for root, dirs, files in os.walk(dataset_path):
            print(f"  {root}")
        return

    print(f"Using dataset from: {actual_dataset_path}")
    print()

    # 1. Explore dataset structure
    explore_dataset_structure(actual_dataset_path)
    
    # 2. Analyze class distribution
    class_names, class_distribution, image_paths, image_info, total_images = analyze_class_distribution(actual_dataset_path)
    
    # 3. Plot class distribution
    plot_class_distribution(class_distribution)
    
    # 4. Display sample images
    display_sample_images(class_names, image_paths)
    
    # 5. Analyze image properties
    dimensions_df, all_dimensions = analyze_image_properties(class_names, image_paths)
    
    # 6. Check image consistency
    unique_sizes = check_image_consistency(all_dimensions)
    
    # 7. Create dataset summary
    summary = {
        'total_images': total_images,
        'number_of_classes': len(class_names),
        'classes': class_names,
        'class_distribution': class_distribution,
        'image_dimensions_summary': dimensions_df.describe().to_dict() if not dimensions_df.empty else {},
        'unique_image_sizes': len(unique_sizes),
        'dataset_path': actual_dataset_path
    }

    print("="*50)
    print("DATASET SUMMARY")
    print("="*50)
    for key, value in summary.items():
        if key not in ['class_distribution', 'image_dimensions_summary']:
            print(f"{key.replace('_', ' ').title()}: {value}")
    print()

    # 8. Save analysis results
    analysis_results = {
        'class_names': class_names,
        'class_distribution': class_distribution,
        'total_images': total_images,
        'image_dimensions': dimensions_df.describe().to_dict() if not dimensions_df.empty else {},
        'dataset_path': actual_dataset_path
    }

    with open(os.path.join(output_path, 'dataset_analysis.json'), 'w') as f:
        json.dump(analysis_results, f, indent=2)

    print(f"Analysis results saved to: {output_path}/dataset_analysis.json")
    print()
    
    # 9. Generate recommendations
    generate_recommendations(class_distribution, total_images, unique_sizes)

if __name__ == "__main__":
    main()