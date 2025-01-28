"""
Created on Tue Oct 29 10:54:50 2024

@author: nimis
"""

import torch
import torch.nn as nn

class MLKA(nn.Module):
    def __init__(self, in_channels, kernel_sizes, dilation):
        super(MLKA, self).__init__()

        # Multi-scale large kernel convolutions with different kernel sizes
        self.conv1 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_sizes[0], 
                               padding=kernel_sizes[0] // 2, dilation=1, groups=in_channels, bias=False)
        
        self.conv2 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_sizes[1], 
                               padding=kernel_sizes[1] // 2 * dilation, dilation=dilation, groups=in_channels, bias=False)
        
        self.conv3 = nn.Conv1d(in_channels, in_channels, kernel_size=kernel_sizes[2], 
                               padding=kernel_sizes[2] // 2 * dilation, dilation=dilation, groups=in_channels, bias=False)

        # 1x1 Pointwise convolution to combine multi-scale features
        self.pointwise_conv = nn.Conv1d(in_channels * 3, in_channels, kernel_size=1, bias=False)
        
        # Batch normalization
        self.bn = nn.BatchNorm1d(in_channels)

        # Attention mechanism
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Apply multi-scale convolutions
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x3 = self.conv3(x)
        
        # Concatenate multi-scale features along channel dimension
        multi_scale_features = torch.cat([x1, x2, x3], dim=1)
        
        # Combine using pointwise convolution
        fused_features = self.pointwise_conv(multi_scale_features)
        
        # Batch norm and attention gate
        attention = self.sigmoid(self.bn(fused_features))
        
        # Element-wise multiplication to apply attention
        return x * attention