import torch
import torch.nn as nn
import torch.nn.functional as F

class Sampler(nn.Module):
    """
    Sampler module from DGMR that generates forecast frames.
    Simplified version for testing purposes.
    """
    def __init__(
        self,
        forecast_steps=6,
        latent_channels=384,
        context_channels=192,
        output_shape=128,
        output_channels=1,
    ):
        super().__init__()
        self.forecast_steps = forecast_steps
        self.latent_channels = latent_channels
        self.context_channels = context_channels
        self.output_shape = output_shape
        self.output_channels = output_channels
        
        # Simplified decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels + context_channels, 256, kernel_size=3, padding=1),
            nn.GroupNorm(8, 256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, output_channels * forecast_steps, kernel_size=1)
        )
    
    def forward(self, latent, context):
        """
        Forward pass for the sampler.
        
        Args:
            latent: Latent features [B, latent_channels, H, W]
            context: Context features [B, context_channels, H, W]
        
        Returns:
            Forecast sequence [B, C, T, H, W]
        """
        batch_size = latent.shape[0]
        
        # Concatenate latent and context features
        features = torch.cat([latent, context], dim=1)
        
        # Generate all forecast steps at once
        forecast_flat = self.decoder(features)  # [B, C*T, H, W]
        
        # Reshape to separate time steps
        forecast = forecast_flat.reshape(
            batch_size, self.output_channels, self.forecast_steps, 
            self.output_shape, self.output_shape
        )
        
        return forecast