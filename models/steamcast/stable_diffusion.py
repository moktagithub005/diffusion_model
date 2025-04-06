import torch
import torch.nn as nn
import torch.nn.functional as F

class StableDiffusionUNet(nn.Module):
    """
    Modified Stable Diffusion U-Net for hail nowcasting.
    Simplified version for testing purposes.
    """
    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        time_dim=256,
        base_channels=128,
        channel_mults=(1, 2, 4),
        attention_resolutions=(8, 4),
        num_res_blocks=2,
        dropout=0.1
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_dim = time_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 2),
            nn.SiLU(),
            nn.Linear(time_dim * 2, time_dim)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)
        
        # Simplified U-Net structure
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU()
        )
        
        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU()
        )
        
        self.middle = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU(),
            nn.Conv2d(base_channels * 4, base_channels * 4, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 4),
            nn.SiLU()
        )
        
        self.up1 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 4 + base_channels * 2, base_channels * 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels * 2),
            nn.SiLU()
        )
        
        self.up2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(base_channels * 2 + base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU()
        )
        
        # Final layers
        self.final_conv = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, output_channels, kernel_size=1)
        )
    
    def forward(self, x, timesteps):
        """
        Forward pass of the U-Net.
        
        Args:
            x: Input features [B, C, H, W]
            timesteps: Diffusion timesteps [B]
            
        Returns:
            Predicted noise or denoised image
        """
        # Time embedding
        t_emb = self.time_embed(timesteps)
        
        # Initial convolution
        h = self.init_conv(x)
        h1 = h
        
        # Downsampling
        h = self.down1(h)
        h2 = h
        
        h = self.down2(h)
        
        # Middle
        h = self.middle(h)
        
        # Add time embedding to middle layer
        t_emb = t_emb.unsqueeze(-1).unsqueeze(-1)
        h = h + t_emb
        
        # Upsampling with skip connections
        h = torch.cat([h, h2], dim=1)
        h = self.up1(h)
        
        h = torch.cat([h, h1], dim=1)
        h = self.up2(h)
        
        # Final layers
        output = self.final_conv(h)
        
        return output


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for target-reference consistency."""
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm_target = nn.LayerNorm(embed_dim)
        self.norm_ref = nn.LayerNorm(embed_dim)
        self.norm_out = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
    
    def forward(self, target, reference):
        """
        Apply cross-attention between target and reference.
        
        Args:
            target: Target features [B, T, embed_dim, H, W]
            reference: Reference features [B, T, embed_dim, H, W]
            
        Returns:
            Updated target features with cross-attention
        """
        batch_size, seq_len, embed_dim, height, width = target.shape
        
        # Reshape for attention
        target_flat = target.flatten(3).permute(0, 3, 1, 2)  # [B, H*W, T, embed_dim]
        target_flat = target_flat.reshape(-1, seq_len, embed_dim)  # [B*H*W, T, embed_dim]
        
        ref_flat = reference.flatten(3).permute(0, 3, 1, 2)  # [B, H*W, T, embed_dim]
        ref_flat = ref_flat.reshape(-1, seq_len, embed_dim)  # [B*H*W, T, embed_dim]
        
        # Apply cross-attention
        target_norm = self.norm_target(target_flat)
        ref_norm = self.norm_ref(ref_flat)
        
        attn_out, _ = self.cross_attn(target_norm, ref_norm, ref_norm)
        target_flat = target_flat + attn_out
        
        # Apply feed-forward network
        ff_in = self.norm_out(target_flat)
        ff_out = self.ff(ff_in)
        target_flat = target_flat + ff_out
        
        # Reshape back
        target_flat = target_flat.reshape(batch_size, height * width, seq_len, embed_dim)
        target_out = target_flat.permute(0, 2, 3, 1).reshape(batch_size, seq_len, embed_dim, height, width)
        
        return target_out