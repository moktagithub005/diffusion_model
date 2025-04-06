import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextConditioningStack(nn.Module):
    """
    Context conditioning stack from DGMR that processes input radar sequence.
    Simplified version for testing purposes.
    """
    def __init__(self, input_channels=1, output_channels=192, num_layers=3):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        
        # Simple encoder with 2D convolutions
        layers = []
        current_channels = input_channels
        channels = [64, 128, output_channels]
        
        for i in range(num_layers):
            layers.append(
                nn.Conv2d(current_channels, channels[i], kernel_size=3, padding=1)
            )
            layers.append(nn.GroupNorm(8, channels[i]))
            layers.append(nn.ReLU(inplace=True))
            current_channels = channels[i]
        
        self.encoder = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass for context conditioning.
        
        Args:
            x: Input tensor [B, C, T, H, W] or [B, T, C, H, W]
        
        Returns:
            Context features [B, output_channels, H, W]
        """
        if x.dim() == 5:
            # Check if input is [B, C, T, H, W] and convert to [B, T, C, H, W] if needed
            if x.shape[1] < x.shape[2]:
                x = x.permute(0, 2, 1, 3, 4)
            
            batch_size, seq_len, channels, height, width = x.shape
            
            # Process the latest frame for simplicity
            x_latest = x[:, -1]  # [B, C, H, W]
            
            # Encode context
            context = self.encoder(x_latest)
            
            return context
        else:
            # Handle unexpected input format
            raise ValueError(f"Expected 5D input tensor, got {x.dim()}D")