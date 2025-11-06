"""
Training script for AlexNet on Alzheimer MRI dataset
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import time
import os
import json
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import numpy as np
import pandas as pd

# Import our custom modules
from model import AlexNet, SimplifiedAlexNet
from data_preprocessing import create_data_loaders, find_dataset_folder
from utils import get_device, load_config

class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, class_names, config):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.class_names = class_names
        self.config = config
        
        self.device = get_device()
        self.model.to(self.device)
        
        # Loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config['training']['learning_rate'],
            weight_decay=1e-4
        )
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=10, gamma=0.1)
        
        # Training history
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'train_acc': [],
            'val_acc': [],
            'learning_rates': []
        }
        
        # Create models directory
        os.makedirs('../models', exist_ok=True)
    
    def train_epoch(self):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (images, labels) in enumerate(self.train_loader):
            images, labels = images.to(self.device), labels.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            if batch_idx % 100 == 0:
                print(f'Batch {batch_idx}/{len(self.train_loader)}, Loss: {loss.item():.4f}')
        
        epoch_loss = running_loss / len(self.train_loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def validate(self, loader):
        """Validate the model"""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        epoch_loss = running_loss / len(loader)
        epoch_acc = 100. * correct / total
        
        return epoch_loss, epoch_acc
    
    def train(self, epochs):
        """Main training loop"""
        print(f"Starting training on {self.device}")
        print(f"Training samples: {len(self.train_loader.dataset)}")
        print(f"Validation samples: {len(self.val_loader.dataset)}")
        
        best_val_acc = 0.0
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc = self.validate(self.val_loader)
            
            # Update scheduler
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Save history
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_acc'].append(val_acc)
            self.history['learning_rates'].append(current_lr)
            
            epoch_time = time.time() - epoch_start
            
            print(f'Epoch {epoch+1}/{epochs}:')
            print(f'  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
            print(f'  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')
            print(f'  LR: {current_lr:.6f}, Time: {epoch_time:.2f}s')
            
            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'config': self.config
                }, '../models/best_model.pth')
                print(f'  🎯 New best model saved with val_acc: {val_acc:.2f}%')
            
            print('-' * 60)
        
        total_time = time.time() - start_time
        print(f'Training completed in {total_time:.2f}s')
        print(f'Best validation accuracy: {best_val_acc:.2f}%')
    
    def evaluate(self):
        """Evaluate on test set"""
        print("\nEvaluating on test set...")
        
        # Load best model
        checkpoint = torch.load('../models/best_model.pth')
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        test_loss, test_acc = self.validate(self.test_loader)
        
        # Detailed classification report
        all_preds = []
        all_labels = []
        
        self.model.eval()
        with torch.no_grad():
            for images, labels in self.test_loader:
                images, labels = images.to(self.device), labels.to(self.device)
                outputs = self.model(images)
                _, predicted = outputs.max(1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        # Classification report
        print(f"\nTest Accuracy: {test_acc:.2f}%")
        print(f"Test Loss: {test_loss:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(all_labels, all_preds, target_names=self.class_names))
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names, 
                   yticklabels=self.class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('../data/processed/confusion_matrix.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        return test_acc, test_loss
    
    def plot_training_history(self):
        """Plot training history"""
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss
        ax1.plot(self.history['train_loss'], label='Train Loss')
        ax1.plot(self.history['val_loss'], label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy
        ax2.plot(self.history['train_acc'], label='Train Acc')
        ax2.plot(self.history['val_acc'], label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate
        ax3.plot(self.history['learning_rates'])
        ax3.set_title('Learning Rate')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)
        
        # Combined
        ax4.plot(self.history['train_loss'], label='Train Loss', alpha=0.7)
        ax4.plot(self.history['val_loss'], label='Val Loss', alpha=0.7)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss', color='blue')
        ax4.tick_params(axis='y', labelcolor='blue')
        ax4.legend(loc='upper left')
        
        ax4_twin = ax4.twinx()
        ax4_twin.plot(self.history['train_acc'], label='Train Acc', color='red', alpha=0.7)
        ax4_twin.plot(self.history['val_acc'], label='Val Acc', color='orange', alpha=0.7)
        ax4_twin.set_ylabel('Accuracy (%)', color='red')
        ax4_twin.tick_params(axis='y', labelcolor='red')
        ax4_twin.legend(loc='upper right')
        ax4.set_title('Loss and Accuracy')
        ax4.grid(True)
        
        plt.tight_layout()
        plt.savefig('../data/processed/training_history.png', dpi=300, bbox_inches='tight')
        plt.show()

def main():
    """Main training function"""
    # Load configuration
    config = load_config()
    
    # Find and load dataset
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    base_data_path = os.path.join(project_root, "data", "raw")
    
    dataset_path = find_dataset_folder(base_data_path)
    if not dataset_path:
        print(f"❌ ERROR: Could not find dataset")
        return
    
    # Create data loaders
    train_loader, val_loader, test_loader, class_names = create_data_loaders(
        data_dir=dataset_path,
        batch_size=config['data']['batch_size'],
        val_split=0.15,
        test_split=0.15
    )
    
    # Create model
    model = AlexNet(num_classes=len(class_names))
    # Alternatively, use simplified model for faster training:
    # model = SimplifiedAlexNet(num_classes=len(class_names))
    
    print(f"Model: {model.__class__.__name__}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer and start training
    trainer = Trainer(model, train_loader, val_loader, test_loader, class_names, config)
    trainer.train(epochs=config['training']['epochs'])
    
    # Evaluate and plot results
    trainer.evaluate()
    trainer.plot_training_history()
    
    print("✅ Training completed successfully!")

if __name__ == "__main__":
    main()