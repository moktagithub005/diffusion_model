import torch
import torch.nn as nn
import torch.nn.functional as F

class Generator(nn.Module):
    """
    Generator from DGMR that combines context and latent stacks.
    Simplified version for testing purposes.
    """
    def __init__(
        self,
        conditioning_stack,
        latent_stack,
        sampler,
    ):
        super().__init__()
        self.conditioning_stack = conditioning_stack
        self.latent_stack = latent_stack
        self.sampler = sampler
    
    def forward(self, x):
        """
        Forward pass for the generator.
        
        Args:
            x: Input tensor [B, C, T, H, W]
        
        Returns:
            Forecast sequence [B, C, T, H, W]
        """
        # Get context and latent features
        context = self.conditioning_stack(x)
        latent = self.latent_stack(x)
        
        # Generate forecast
        forecast = self.sampler(latent, context)
        
        return forecast