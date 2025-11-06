"""
AlexNet Implementation for Alzheimer MRI Classification
Adapted from the original AlexNet paper for 4-class classification
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

class AlexNet(nn.Module):
    """
    AlexNet model adapted for Alzheimer MRI classification
    Original AlexNet was designed for 1000 ImageNet classes
    We adapt it for 4 classes: ['MildDemented', 'ModerateDemented', 'NonDemented', 'VeryMildDemented']
    """
    
    def __init__(self, num_classes=4, dropout=0.5):
        super(AlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # Conv1: Input: 3x224x224, Output: 64x55x55
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),  # LRN layer
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2: Input: 64x27x27, Output: 192x27x27
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.LocalResponseNorm(size=5, alpha=0.0001, beta=0.75, k=2),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3: Input: 192x13x13, Output: 384x13x13
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4: Input: 384x13x13, Output: 256x13x13
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5: Input: 256x13x13, Output: 256x13x13
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            
            nn.Linear(4096, num_classes),
        )
        
        # Initialize weights
        self._initialize_weights()
    
    def forward(self, x):
        # Feature extraction
        x = self.features(x)
        x = self.avgpool(x)
        
        # Flatten for fully connected layers
        x = torch.flatten(x, 1)
        
        # Classification
        x = self.classifier(x)
        return x
    
    def _initialize_weights(self):
        """Initialize weights using method from original paper"""
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0, std=0.01)
                nn.init.constant_(module.bias, 1)

class SimplifiedAlexNet(nn.Module):
    """
    Simplified version of AlexNet with fewer parameters
    Better for smaller datasets and faster training
    """
    
    def __init__(self, num_classes=4, dropout=0.5):
        super(SimplifiedAlexNet, self).__init__()
        
        self.features = nn.Sequential(
            # Conv1
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv2
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            
            # Conv3
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv4
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            
            # Conv5
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            
            nn.Dropout(p=dropout),
            nn.Linear(4096, 1024),  # Reduced from 4096 to 1024
            nn.ReLU(inplace=True),
            
            nn.Linear(1024, num_classes),
        )
    
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

def test_model():
    """Test the model with a sample input"""
    # Create model
    model = AlexNet(num_classes=4)
    
    # Test with sample input
    sample_input = torch.randn(2, 3, 224, 224)  # Batch of 2 images
    output = model(sample_input)
    
    print("✅ Model test successful!")
    print(f"Input shape: {sample_input.shape}")
    print(f"Output shape: {output.shape}")
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test simplified model too
    simple_model = SimplifiedAlexNet(num_classes=4)
    simple_output = simple_model(sample_input)
    print(f"Simplified model parameters: {sum(p.numel() for p in simple_model.parameters()):,}")
    
    return model, simple_model

def count_parameters(model):
    """Count total trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model, simple_model = test_model()