"""
CNN model definition module
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TinyCNN(nn.Module):
    """Compact and practical CNN example for 8x8 input"""
    def __init__(self):
        super(TinyCNN, self).__init__()
        # Input: 1x8x8 → Output: 2x8x8
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=3, padding=1)
        
        # Output: 2x8x8 → MaxPool: 2x4x4
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Output: 2x4x4 → 4x4x4
        self.conv2 = nn.Conv2d(in_channels=2, out_channels=4, kernel_size=3, padding=1)
        
        # Output: 4x4x4 → AvgPool: 4x2x2
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        
        # 1x1 convolution to reduce channel dimension
        self.conv3 = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=1)
        
        # Final convolution that directly outputs 2 features (instead of flattening + linear)
        # Kernel size 2x2 matches our 2x2 feature map after conv3
        self.conv_out = nn.Conv2d(in_channels=2, out_channels=2, kernel_size=2)
    
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))
        # Direct convolution to output features instead of flatten + linear
        x = self.conv_out(x)
        # Remove spatial dimensions (will be 1x1 after conv_out)
        x = x.squeeze(3).squeeze(2)
        return x
