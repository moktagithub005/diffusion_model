import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional

class SpEn(nn.Module):
    """
    Spatiotemporal Encoding (SpEn) module for encoding spatial-temporal information
    as described in the SteamCast paper.
    """
    def __init__(self, input_channels=1, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_channels = input_channels
        self.embed_dim = embed_dim
        
        # Initial convolutional encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, kernel_size=3, padding=1),
            nn.GroupNorm(8, 64),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.SiLU(),
            nn.Conv2d(128, embed_dim, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, embed_dim),
            nn.SiLU()
        )
        
        # Position embedding
        self.pos_embed = PositionEmbedding(embed_dim=embed_dim)
        
        # Time embedding
        self.time_embed = TimeEmbedding(embed_dim=embed_dim)
        
        # Self-attention block
        self.self_attn = SelfAttentionBlock(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        
    def forward(self, x, time_steps=None):
        """
        Forward pass of SpEn.
        
        Args:
            x: Input tensor [B, T, C, H, W]
            time_steps: Optional time steps tensor
            
        Returns:
            Encoded features
        """
        batch_size, seq_len, channels, height, width = x.shape
        
        # Process each time step
        encoded_features = []
        for t in range(seq_len):
            # Extract features using 2D CNN
            features = self.encoder(x[:, t])  # [B, embed_dim, H//4, W//4]
            
            # Add position embedding
            features = self.pos_embed(features)
            
            # Add time embedding if time_steps provided
            if time_steps is not None:
                time_emb = self.time_embed(time_steps[:, t])
                features = features + time_emb.unsqueeze(-1).unsqueeze(-1)
            
            encoded_features.append(features)
        
        # Stack along time dimension
        encoded_features = torch.stack(encoded_features, dim=1)  # [B, T, embed_dim, H//4, W//4]
        
        # Apply self-attention across time
        encoded_features = self.self_attn(encoded_features)
        
        return encoded_features

class PositionEmbedding(nn.Module):
    """Position embedding module for spatial positions."""
    def __init__(self, embed_dim=256):
        super().__init__()
        self.embed_dim = embed_dim
        self.pos_embed = nn.Parameter(torch.zeros(1, embed_dim, 1, 1))
        self.initialize_parameters()
        
    def initialize_parameters(self):
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        """
        Add position embeddings to input features.
        
        Args:
            x: Input features [B, C, H, W]
            
        Returns:
            Features with position embeddings
        """
        return x + self.pos_embed

class TimeEmbedding(nn.Module):
    """Time embedding module for temporal information."""
    def __init__(self, embed_dim=256, max_period=10000):
        super().__init__()
        self.embed_dim = embed_dim
        self.max_period = max_period
        
        # Sinusoidal position embedding followed by MLP
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 2),
            nn.SiLU(),
            nn.Linear(embed_dim * 2, embed_dim)
        )
        
    def forward(self, t):
        """
        Create time embeddings for the given time steps.
        
        Args:
            t: Time steps tensor [B]
            
        Returns:
            Time embeddings [B, embed_dim]
        """
        half_dim = self.embed_dim // 2
        emb = torch.log(torch.tensor(self.max_period, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, embed_dim]
        
        # Further process with MLP
        emb = self.mlp(emb)
        
        return emb

class SelfAttentionBlock(nn.Module):
    """Self-attention block for capturing temporal dependencies."""
    def __init__(self, embed_dim=256, num_heads=8, dropout=0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Feed-forward network
        self.ff = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout)
        )
        
    def forward(self, x):
        """
        Apply self-attention across time dimension.
        
        Args:
            x: Input features [B, T, embed_dim, H, W]
            
        Returns:
            Processed features with self-attention
        """
        batch_size, seq_len, embed_dim, height, width = x.shape
        
        # Reshape for attention
        x_flat = x.flatten(3).permute(0, 3, 1, 2)  # [B, H*W, T, embed_dim]
        x_flat = x_flat.reshape(-1, seq_len, embed_dim)  # [B*H*W, T, embed_dim]
        
        # Apply self-attention
        x_attn = self.norm1(x_flat)
        x_attn, _ = self.self_attn(x_attn, x_attn, x_attn)
        x_flat = x_flat + x_attn
        
        # Apply feed-forward network
        x_ff = self.norm2(x_flat)
        x_ff = self.ff(x_ff)
        x_flat = x_flat + x_ff
        
        # Reshape back
        x_flat = x_flat.reshape(batch_size, height * width, seq_len, embed_dim)
        x_out = x_flat.permute(0, 2, 3, 1).reshape(batch_size, seq_len, embed_dim, height, width)
        
        return x_out

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

class StableDiffusionUNet(nn.Module):
    """
    Modified Stable Diffusion U-Net for hail nowcasting as described in SteamCast.
    """
    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        time_dim=256,
        base_channels=128,
        channel_mults=(1, 2, 4, 8),
        attention_resolutions=(8, 4, 2),
        num_res_blocks=2,
        dropout=0.1,
        use_checkpoint=False
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.time_dim = time_dim
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim)
        )
        
        # Initial convolution
        self.init_conv = nn.Conv2d(input_channels, base_channels, kernel_size=3, padding=1)
        
        # Encoder blocks
        self.encoder_blocks = nn.ModuleList()
        input_ch = base_channels
        channels = [input_ch]
        
        for level, mult in enumerate(channel_mults):
            out_ch = base_channels * mult
            
            for _ in range(num_res_blocks):
                self.encoder_blocks.append(
                    ResBlock(
                        input_ch,
                        out_ch,
                        time_dim,
                        dropout,
                        use_attention=(level in attention_resolutions),
                        use_checkpoint=use_checkpoint
                    )
                )
                input_ch = out_ch
                channels.append(input_ch)
            
            # Downsample except for the last level
            if level < len(channel_mults) - 1:
                self.encoder_blocks.append(Downsample(input_ch))
                channels.append(input_ch)
        
        # Middle blocks
        self.middle_block1 = ResBlock(
            input_ch,
            input_ch,
            time_dim,
            dropout,
            use_attention=True,
            use_checkpoint=use_checkpoint
        )
        self.middle_attn = SelfAttention(input_ch)
        self.middle_block2 = ResBlock(
            input_ch,
            input_ch,
            time_dim,
            dropout,
            use_checkpoint=use_checkpoint
        )
        
        # Decoder blocks
        self.decoder_blocks = nn.ModuleList()
        
        for level, mult in reversed(list(enumerate(channel_mults))):
            out_ch = base_channels * mult
            
            for i in range(num_res_blocks + 1):
                in_channels = channels.pop()
                self.decoder_blocks.append(
                    ResBlock(
                        in_channels + channels[-1],
                        out_ch,
                        time_dim,
                        dropout,
                        use_attention=(level in attention_resolutions),
                        use_checkpoint=use_checkpoint
                    )
                )
                
            # Upsample except for the last level
            if level > 0:
                self.decoder_blocks.append(Upsample(out_ch))
        
        # Final layers
        self.final_res_block = ResBlock(
            base_channels * 2,
            base_channels,
            time_dim,
            dropout,
            use_checkpoint=use_checkpoint
        )
        self.final_conv = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, output_channels, kernel_size=3, padding=1)
        )
    
    def forward(self, x, timesteps):
        """
        Forward pass of the U-Net.
        
        Args:
            x: Input features [B, C, H, W]
            timesteps: Diffusion timesteps [B]
            
        Returns:
            Predicted noise
        """
        # Time embedding
        temb = self.time_embed(timesteps)
        
        # Initial convolution
        h = self.init_conv(x)
        hs = [h]
        
        # Encoder
        for module in self.encoder_blocks:
            h = module(h, temb)
            hs.append(h)
        
        # Middle
        h = self.middle_block1(h, temb)
        h = self.middle_attn(h)
        h = self.middle_block2(h, temb)
        
        # Decoder
        for module in self.decoder_blocks:
            if isinstance(module, ResBlock):
                h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, temb)
        
        # Final layers
        h = torch.cat([h, hs.pop()], dim=1)
        h = self.final_res_block(h, temb)
        h = self.final_conv(h)
        
        return h

class ResBlock(nn.Module):
    """Residual block with optional attention."""
    def __init__(
        self,
        in_channels,
        out_channels,
        time_dim,
        dropout,
        use_attention=False,
        use_checkpoint=False
    ):
        super().__init__()
        self.use_checkpoint = use_checkpoint
        
        self.in_layers = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        )
        
        self.time_emb_proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, out_channels)
        )
        
        self.out_layers = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        )
        
        if in_channels != out_channels:
            self.skip_connection = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        else:
            self.skip_connection = nn.Identity()
        
        if use_attention:
            self.attention = SelfAttention(out_channels)
        else:
            self.attention = None
    
    def forward(self, x, temb):
        """Forward pass of the residual block."""
        if self.use_checkpoint and self.training:
            return torch.utils.checkpoint.checkpoint(self._forward, x, temb)
        else:
            return self._forward(x, temb)
    
    def _forward(self, x, temb):
        h = self.in_layers(x)
        h = h + self.time_emb_proj(temb).unsqueeze(-1).unsqueeze(-1)
        h = self.out_layers(h)
        h = h + self.skip_connection(x)
        
        if self.attention is not None:
            h = self.attention(h)
        
        return h

class SelfAttention(nn.Module):
    """Self-attention module."""
    def __init__(self, channels, num_heads=8):
        super().__init__()
        self.channels = channels
        self.num_heads = num_heads
        
        self.norm = nn.GroupNorm(8, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, kernel_size=1)
        self.proj = nn.Conv2d(channels, channels, kernel_size=1)
    
    def forward(self, x):
        """Forward pass of self-attention."""
        b, c, h, w = x.shape
        
        # Normalize input
        x_norm = self.norm(x)
        
        # Get query, key, value
        qkv = self.qkv(x_norm)
        q, k, v = torch.chunk(qkv, 3, dim=1)
        
        # Reshape for attention
        q = q.reshape(b, self.num_heads, c // self.num_heads, h * w).permute(0, 1, 3, 2)  # [B, num_heads, H*W, C//num_heads]
        k = k.reshape(b, self.num_heads, c // self.num_heads, h * w).permute(0, 1, 2, 3)  # [B, num_heads, C//num_heads, H*W]
        v = v.reshape(b, self.num_heads, c // self.num_heads, h * w).permute(0, 1, 3, 2)  # [B, num_heads, H*W, C//num_heads]
        
        # Attention
        attn = torch.matmul(q, k) * (c // self.num_heads) ** -0.5
        attn = F.softmax(attn, dim=-1)
        
        # Apply attention
        out = torch.matmul(attn, v).permute(0, 1, 3, 2)  # [B, num_heads, C//num_heads, H*W]
        out = out.reshape(b, c, h, w)
        
        # Project back
        out = self.proj(out)
        
        return out + x

class Downsample(nn.Module):
    """Downsampling module."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, stride=2, padding=1)
    
    def forward(self, x):
        return self.conv(x)

class Upsample(nn.Module):
    """Upsampling module."""
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.conv(x)

class SteamCast(nn.Module):
    """
    SteamCast: A Spatial-temporal Deep Probabilistic Diffusion Model for Reliable Hail Nowcasting.
    
    Based on the paper: https://arxiv.org/abs/2104.00954
    """
    def __init__(
        self,
        input_channels=1,
        output_channels=1,
        hidden_channels=128,
        embed_dim=256,
        input_seq_len=5,
        pred_seq_len=6,
        num_diffusion_steps=1000,
        beta_schedule="linear",
        beta_start=1e-4,
        beta_end=2e-2
    ):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = hidden_channels
        self.embed_dim = embed_dim
        self.input_seq_len = input_seq_len
        self.pred_seq_len = pred_seq_len
        self.num_diffusion_steps = num_diffusion_steps
        
        # Set up betas for diffusion
        if beta_schedule == "linear":
            betas = torch.linspace(beta_start, beta_end, num_diffusion_steps)
        elif beta_schedule == "cosine":
            steps = torch.arange(num_diffusion_steps + 1) / num_diffusion_steps
            alpha_bar = torch.cos((steps + 0.008) / 1.008 * torch.pi / 2) ** 2
            alpha_bar = alpha_bar / alpha_bar[0]
            betas = 1 - alpha_bar[1:] / alpha_bar[:-1]
            betas = torch.clamp(betas, max=0.999)
        else:
            raise ValueError(f"Unknown beta schedule: {beta_schedule}")
        
        self.register_buffer("betas", betas)
        alphas = 1 - betas
        self.register_buffer("alphas", alphas)
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.register_buffer("alphas_cumprod", alphas_cumprod)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
        self.register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        self.register_buffer("sqrt_alphas_cumprod", sqrt_alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - alphas_cumprod)
        self.register_buffer("sqrt_one_minus_alphas_cumprod", sqrt_one_minus_alphas_cumprod)
        
        # SpEn encoder for target and reference
        self.target_encoder = SpEn(
            input_channels=input_channels,
            embed_dim=embed_dim
        )
        
        self.reference_encoder = SpEn(
            input_channels=input_channels,
            embed_dim=embed_dim
        )
        
        # Cross-attention for target-reference consistency
        self.cross_attention = CrossAttentionBlock(
            embed_dim=embed_dim
        )
        
        # Stable Diffusion U-Net
        self.unet = StableDiffusionUNet(
            input_channels=input_channels,
            output_channels=output_channels,
            time_dim=embed_dim,
            base_channels=hidden_channels
        )
    
    def forward(self, target, reference, t):
        """
        Forward pass of SteamCast.
        
        Args:
            target: Target patch sequence [B, T, C, H, W]
            reference: Reference patch sequences [B, T, C, H, W]
            t: Diffusion time steps [B]
            
        Returns:
            Predicted noise
        """
        # Encode target and reference patches
        target_features = self.target_encoder(target)
        reference_features = self.reference_encoder(reference)
        
        # Apply cross-attention for consistency
        fused_features = self.cross_attention(target_features, reference_features)
        
        # Extract features for the nowcast time step
        # We use the last timestep of the fused features for nowcasting
        nowcast_features = fused_features[:, -1]  # [B, embed_dim, H//4, W//4]
        
        # Time embedding for diffusion
        time_emb = self.get_time_embedding(t)
        
        # Predict noise using U-Net
        noise_pred = self.unet(nowcast_features, time_emb)
        
        return noise_pred
    
    def get_time_embedding(self, t):
        """Generate time embeddings for diffusion timesteps."""
        half_dim = self.embed_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, embed_dim]
        return emb
    
    def add_noise(self, x, t):
        """
        Add noise to the input according to the diffusion schedule.
        
        Args:
            x: Clean data [B, C, H, W]
            t: Timesteps [B]
            
        Returns:
            Noisy data
        """
        sqrt_alphas_cumprod_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alphas_cumprod_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        
        # Generate noise
        noise = torch.randn_like(x)
        
        # Add noise according to diffusion schedule
        return sqrt_alphas_cumprod_t * x + sqrt_one_minus_alphas_cumprod_t * noise, noise
    
    def sample_timesteps(self, batch_size, device):
        """Sample timesteps uniformly for training."""
        return torch.randint(0, self.num_diffusion_steps, (batch_size,), device=device)
    
    def training_losses(self, target, reference):
        """
        Compute training losses for SteamCast.
        
        Args:
            target: Target patch sequence [B, T, C, H, W]
            reference: Reference patch sequences [B, T, C, H, W]
            
        Returns:
            Dictionary of losses
        """
        batch_size = target.shape[0]
        device = target.device
        
        # Extract nowcast target (last frame)
        nowcast_target = target[:, -1]  # [B, C, H, W]
        
        # Sample timesteps
        t = self.sample_timesteps(batch_size, device)
        
        # Add noise to target
        noisy_target, noise = self.add_noise(nowcast_target, t)
        
        # Predict noise
        noise_pred = self(target, reference, t)
        
        # Simple MSE loss between predicted and actual noise
        loss = F.mse_loss(noise_pred, noise)
        
        return {"loss": loss}
    
    @torch.no_grad()
    def sample(self, target, reference, num_steps=50):
        """
        Sample from the diffusion model.
        
        Args:
            target: Target patch sequence [B, T, C, H, W]
            reference: Reference patch sequences [B, T, C, H, W]
            num_steps: Number of sampling steps
            
        Returns:
            Sampled images
        """
        batch_size = target.shape[0]
        device = target.device
        
        # Start from pure noise
        x = torch.randn(
            batch_size, 
            self.output_channels, 
            target.shape[3], 
            target.shape[4], 
            device=device
        )
        
        # Progressively denoise
        for i in reversed(range(0, num_steps)):
            # Set timestep
            t = torch.ones(batch_size, device=device) * i
            
            # Predict noise
            predicted_noise = self(target, reference, t.long())
            
            # Update sample using predicted noise
            alpha = self.alphas[i]
            alpha_cumprod = self.alphas_cumprod[i]
            beta = self.betas[i]
            
            if i > 0:
                noise = torch.randn_like(x)
            else:
                noise = torch.zeros_like(x)
                
            # Update step
            x = (1 / torch.sqrt(alpha)) * (
                x - (beta / (torch.sqrt(1 - alpha_cumprod))) * predicted_noise
            ) + torch.sqrt(beta) * noise
        
        return x
    
    @torch.no_grad()
    def generate_forecast(self, input_sequence, reference_sequences=None):
        """
        Generate a forecast for the given input sequence.
        
        Args:
            input_sequence: Input radar sequence [B, T, C, H, W]
            reference_sequences: Optional reference sequences
            
        Returns:
            Forecasted radar sequence
        """
        # If no reference sequences provided, use input_sequence
        if reference_sequences is None:
            reference_sequences = input_sequence
        
        # Generate forecast using diffusion sampling
        forecast = self.sample(input_sequence, reference_sequences)
        
        return forecast