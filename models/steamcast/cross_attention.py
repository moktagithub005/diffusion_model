import torch
import torch.nn as nn
import torch.nn.functional as F

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