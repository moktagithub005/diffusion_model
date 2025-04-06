import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Tuple, Optional, Union

# Import DGMR components
from models.dgmr.context_stack import ContextConditioningStack
from models.dgmr.latent_stack import LatentConditioningStack
from models.dgmr.sampler import Sampler
from models.dgmr.generator import Generator

# Import SteamCast components
from models.steamcast.spen import SpEn
from models.steamcast.stable_diffusion import StableDiffusionUNet
from models.steamcast.cross_attention import CrossAttentionBlock

class IntegratedHailModel(nn.Module):
    """
    Integrated model combining DGMR and SteamCast for hail nowcasting.
    
    This model uses:
    1. DGMR's generator for high-quality precipitation forecasting
    2. SteamCast's spatial-temporal fusion for hail-specific detection
    3. Combined output for reliable hail nowcasting
    """
    def __init__(
    self,
    input_channels: int = 1,
    output_channels: int = 1,
    hidden_dim: int = 128,
    context_channels: int = 192,
    latent_channels: int = 384,
    num_samples: int = 3,
    forecast_steps: int = 6,
    output_shape: int = 128,
    dgmr_weight: float = 0.5,  # Added default
    steamcast_weight: float = 0.5,  # Added default
    use_diffusion: bool = True,
    diffusion_steps: int = 50,
):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_dim = hidden_dim
        self.forecast_steps = forecast_steps
        self.num_samples = num_samples
        self.dgmr_weight = dgmr_weight
        self.steamcast_weight = steamcast_weight
        self.use_diffusion = use_diffusion
        self.diffusion_steps = diffusion_steps
        
        # Initialize DGMR components
        self.context_stack = ContextConditioningStack(
            input_channels=input_channels,
            output_channels=context_channels,
        )
        
        self.latent_stack = LatentConditioningStack(
            input_channels=input_channels,
            output_channels=latent_channels,
        )
        
        self.sampler = Sampler(
            forecast_steps=forecast_steps,
            latent_channels=latent_channels,
            context_channels=context_channels,
            output_shape=output_shape,
            output_channels=output_channels,
        )
        
        self.dgmr_generator = Generator(
            conditioning_stack=self.context_stack,
            latent_stack=self.latent_stack,
            sampler=self.sampler,
        )
        
        # Initialize SteamCast components
        self.target_encoder = SpEn(
            input_channels=input_channels,
            embed_dim=hidden_dim,
        )
        
        self.reference_encoder = SpEn(
            input_channels=input_channels,
            embed_dim=hidden_dim,
        )
        
        self.cross_attention = CrossAttentionBlock(
            embed_dim=hidden_dim,
        )
        
        if self.use_diffusion:
            self.diffusion_model = StableDiffusionUNet(
                input_channels=input_channels + context_channels,  # Concatenate with DGMR features
                output_channels=output_channels,
                time_dim=hidden_dim,
                base_channels=hidden_dim,
            )
            
            # Diffusion parameters
            betas = torch.linspace(1e-4, 2e-2, diffusion_steps)
            self.register_buffer("betas", betas)
            alphas = 1 - betas
            self.register_buffer("alphas", alphas)
            alphas_cumprod = torch.cumprod(alphas, dim=0)
            self.register_buffer("alphas_cumprod", alphas_cumprod)
        
        # Hail prediction head (for classification or regression)
        self.hail_prediction_head = nn.Sequential(
            nn.Conv2d(hidden_dim + context_channels, hidden_dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim),
            nn.SiLU(),
            nn.Conv2d(hidden_dim, hidden_dim // 2, kernel_size=3, padding=1),
            nn.GroupNorm(8, hidden_dim // 2),
            nn.SiLU(),
            nn.Conv2d(hidden_dim // 2, output_channels, kernel_size=1),
        )
        
        # Time embedding
        self.time_embedding = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        
        # Position embedding
        self.position_embedding = nn.Parameter(torch.zeros(1, hidden_dim, output_shape // 4, output_shape // 4))
        
        # Initialize position embedding
        nn.init.normal_(self.position_embedding, std=0.02)
    
    def forward(
        self, 
        x: torch.Tensor, 
        surrounding_blocks: Optional[torch.Tensor] = None,
        return_intermediates: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the integrated model.
        
        Args:
            x: Input radar sequence [B, T, C, H, W]
            surrounding_blocks: Optional surrounding block information [B, T, C, H, W]
            return_intermediates: Whether to return intermediate tensors
            
        Returns:
            Dictionary of outputs
        """
        batch_size, seq_len, channels, height, width = x.shape
        device = x.device
        
        # Reshape for DGMR
        x_dgmr = x.permute(0, 2, 1, 3, 4)  # [B, C, T, H, W]
        
        # Process through DGMR generator
        # Note: DGMR generator returns output of shape [B, C, T_forecast, H, W]
        dgmr_output = []
        for _ in range(self.num_samples):
            sample = self.dgmr_generator(x_dgmr)
            dgmr_output.append(sample)
        
        # Stack samples
        dgmr_output = torch.stack(dgmr_output, dim=1)  # [B, num_samples, C, T_forecast, H, W]
        
        # Process through SteamCast components
        # Use DGMR's context features
        context_features = self.context_stack(x_dgmr)  # [B, context_channels, H, W]
        
        # Encode target sequence
        target_features = self.target_encoder(x)  # [B, T, embed_dim, H//4, W//4]
        
        # Create reference features from surrounding blocks if provided
        if surrounding_blocks is not None:
            reference_features = self.reference_encoder(surrounding_blocks)
        else:
            # Use target as reference if no surrounding blocks
            reference_features = self.reference_encoder(x)
        
        # Apply cross-attention for consistency
        fused_features = self.cross_attention(target_features, reference_features)
        
        # Extract features for the nowcast time step (last timestep)
        nowcast_features = fused_features[:, -1]  # [B, embed_dim, H//4, W//4]
        
        # Add position embedding
        nowcast_features = nowcast_features + self.position_embedding
        
        # Upsample nowcast features to match context features
        nowcast_features = F.interpolate(
            nowcast_features, 
            size=(height, width), 
            mode='bilinear', 
            align_corners=False
        )
        
        # Concatenate context and nowcast features
        combined_features = torch.cat([context_features, nowcast_features], dim=1)
        
        # Generate hail prediction
        hail_predictions = []
        
        if self.use_diffusion:
            # Sample from diffusion model
            for _ in range(self.num_samples):
                # Start from pure noise
                x_noisy = torch.randn(
                    batch_size, 
                    self.output_channels, 
                    height, 
                    width, 
                    device=device
                )
                
                # Progressively denoise
                for i in reversed(range(0, self.diffusion_steps)):
                    # Create timestep embedding
                    t_emb = torch.ones(batch_size, device=device) * i
                    t_emb = self._get_time_embedding(t_emb)
                    
                    # Predict noise
                    noise_pred = self.diffusion_model(
                        torch.cat([x_noisy, combined_features], dim=1),
                        t_emb
                    )
                    
                    # Update sample using predicted noise
                    alpha = self.alphas[i]
                    alpha_cumprod = self.alphas_cumprod[i]
                    beta = self.betas[i]
                    
                    if i > 0:
                        noise = torch.randn_like(x_noisy)
                    else:
                        noise = torch.zeros_like(x_noisy)
                    
                    # Update step
                    x_noisy = (1 / torch.sqrt(alpha)) * (
                        x_noisy - (beta / (torch.sqrt(1 - alpha_cumprod))) * noise_pred
                    ) + torch.sqrt(beta) * noise
                
                hail_predictions.append(x_noisy)
        else:
            # Direct prediction using the hail prediction head
            for _ in range(self.num_samples):
                hail_pred = self.hail_prediction_head(combined_features)
                hail_predictions.append(hail_pred)
        
        # Stack hail predictions
        hail_predictions = torch.stack(hail_predictions, dim=1)  # [B, num_samples, C, H, W]
        
        # Return outputs
        outputs = {
            "weather_forecasts": dgmr_output,  # [B, num_samples, C, T_forecast, H, W]
            "hail_predictions": hail_predictions,  # [B, num_samples, C, H, W]
        }
        
        if return_intermediates:
            outputs.update({
                "context_features": context_features,
                "target_features": target_features,
                "reference_features": reference_features,
                "fused_features": fused_features,
                "combined_features": combined_features,
            })
        
        return outputs
    
    def _get_time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        """Generate time embeddings for diffusion timesteps."""
        half_dim = self.hidden_dim // 2
        emb = torch.log(torch.tensor(10000.0, device=t.device)) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t.unsqueeze(1) * emb.unsqueeze(0)  # [B, half_dim]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1)  # [B, embed_dim]
        
        # Pass through MLP
        emb = self.time_embedding(emb)
        
        return emb
    
    @torch.no_grad()
    def generate_forecast(
        self,
        x: torch.Tensor,
        surrounding_blocks: Optional[torch.Tensor] = None,
        ensemble_method: str = "mean",
    ) -> Dict[str, torch.Tensor]:
        """
        Generate forecast for both weather and hail.
        
        Args:
            x: Input radar sequence [B, T, C, H, W]
            surrounding_blocks: Optional surrounding block information
            ensemble_method: Method to ensemble multiple samples ("mean" or "median")
            
        Returns:
            Dictionary with weather and hail forecasts
        """
        # Run forward pass
        outputs = self.forward(x, surrounding_blocks)
        
        # Ensemble predictions
        if ensemble_method == "mean":
            weather_forecast = outputs["weather_forecasts"].mean(dim=1)
            hail_prediction = outputs["hail_predictions"].mean(dim=1)
        elif ensemble_method == "median":
            weather_forecast = outputs["weather_forecasts"].median(dim=1).values
            hail_prediction = outputs["hail_predictions"].median(dim=1).values
        else:
            raise ValueError(f"Unknown ensemble method: {ensemble_method}")
        
        return {
            "weather_forecast": weather_forecast,  # [B, C, T_forecast, H, W]
            "hail_forecast": hail_prediction,      # [B, C, H, W]
        }
    
    def inference_with_threshold(
        self, 
        x: torch.Tensor,
        surrounding_blocks: Optional[torch.Tensor] = None,
        hail_threshold: float = 0.5,
    ) -> Dict[str, torch.Tensor]:
        """
        Run inference with thresholding for hail prediction.
        
        Args:
            x: Input radar sequence [B, T, C, H, W]
            surrounding_blocks: Optional surrounding block information
            hail_threshold: Threshold for hail detection
            
        Returns:
            Dictionary with weather forecast, hail probability, and binary hail mask
        """
        # Generate forecast
        forecast = self.generate_forecast(x, surrounding_blocks)
        
        # Apply threshold to hail prediction
        hail_probability = forecast["hail_forecast"]
        hail_binary = (hail_probability > hail_threshold).float()
        
        # Add to outputs
        forecast["hail_binary"] = hail_binary
        
        return forecast


class EnsembleHailModel(nn.Module):
    """
    Ensemble model combining multiple hail prediction models.
    
    This can include:
    1. DGMR model
    2. SteamCast model
    3. Integrated model
    4. Traditional ML model (if provided)
    """
    def __init__(
        self,
        models: List[nn.Module],
        model_weights: Optional[List[float]] = None,
        voting_threshold: int = 2,
    ):
        super().__init__()
        self.models = nn.ModuleList(models)
        
        # If weights not provided, use equal weighting
        if model_weights is None:
            self.model_weights = torch.ones(len(models)) / len(models)
        else:
            assert len(model_weights) == len(models), "Number of weights must match number of models"
            self.model_weights = torch.tensor(model_weights) / sum(model_weights)
        
        self.voting_threshold = voting_threshold
    
    def forward(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the ensemble model.
        
        Args:
            x: Input data
            **kwargs: Additional keyword arguments for models
            
        Returns:
            Ensemble predictions
        """
        all_predictions = []
        
        # Get predictions from all models
        for model in self.models:
            with torch.no_grad():
                pred = model(x, **kwargs)
                all_predictions.append(pred)
        
        return self._ensemble_predictions(all_predictions)
    
    def _ensemble_predictions(self, predictions: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        """
        Ensemble predictions from multiple models.
        
        Args:
            predictions: List of predictions from different models
            
        Returns:
            Ensembled predictions
        """
        # Extract hail predictions
        hail_preds = []
        for pred in predictions:
            if "hail_predictions" in pred:
                hail_preds.append(pred["hail_predictions"].mean(dim=1))  # Average over samples
            elif "hail_forecast" in pred:
                hail_preds.append(pred["hail_forecast"])
        
        # Weight and combine predictions
        weighted_hail_preds = []
        for i, pred in enumerate(hail_preds):
            weighted_hail_preds.append(pred * self.model_weights[i])
        
        # Sum weighted predictions
        ensemble_hail_pred = sum(weighted_hail_preds)
        
        # Create binary predictions
        binary_preds = [(pred > 0.5).float() for pred in hail_preds]
        
        # Voting ensemble
        sum_binary = sum(binary_preds)
        voting_result = (sum_binary >= self.voting_threshold).float()
        
        return {
            "hail_forecast": ensemble_hail_pred,
            "hail_binary": voting_result,
            "model_votes": sum_binary,
        }
    
    @torch.no_grad()
    def generate_forecast(self, x: torch.Tensor, **kwargs) -> Dict[str, torch.Tensor]:
        """
        Generate ensemble forecast.
        
        Args:
            x: Input radar sequence
            **kwargs: Additional keyword arguments for models
            
        Returns:
            Ensemble forecast
        """
        all_forecasts = []
        
        # Get forecasts from all models
        for model in self.models:
            if hasattr(model, "generate_forecast"):
                forecast = model.generate_forecast(x, **kwargs)
                all_forecasts.append(forecast)
            else:
                # If model doesn't have generate_forecast method, use forward
                pred = model(x, **kwargs)
                all_forecasts.append(pred)
        
        return self._ensemble_predictions(all_forecasts)
    
    @torch.no_grad()
    def inference_with_threshold(
        self, 
        x: torch.Tensor,
        hail_threshold: float = 0.5,
        **kwargs
    ) -> Dict[str, torch.Tensor]:
        """
        Run ensemble inference with thresholding.
        
        Args:
            x: Input radar sequence
            hail_threshold: Threshold for hail detection
            **kwargs: Additional keyword arguments for models
            
        Returns:
            Ensemble forecast with thresholding
        """
        # Generate forecast
        forecast = self.generate_forecast(x, **kwargs)
        
        # Apply threshold to ensemble prediction
        hail_probability = forecast["hail_forecast"]
        hail_binary = (hail_probability > hail_threshold).float()
        
        # Update binary prediction
        forecast["hail_binary"] = hail_binary
        
        return forecast