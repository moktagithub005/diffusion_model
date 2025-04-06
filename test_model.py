# Test script to verify model functionality
import torch
import sys
import os

# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.integrated_model import IntegratedHailModel

def test_model():
    print("Creating model instance...")
    
    # Create a simplified model for testing
    model = IntegratedHailModel(
        input_channels=1,
        output_channels=1,
        hidden_dim=64,  # Reduced size for testing
        context_channels=64,
        latent_channels=128,
        num_samples=1,  # Just one sample for faster testing
        forecast_steps=4,
        output_shape=128,
        use_diffusion=False,  # No diffusion for faster testing
    )
    
    print("Model created successfully!")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create dummy input data (batch, time, channel, height, width)
    print("Creating test data...")
    test_data = torch.randn(1, 4, 1, 128, 128)
    print(f"Test data shape: {test_data.shape}")
    
    # Run inference
    print("Running inference...")
    with torch.no_grad():
        try:
            output = model(test_data)
            print("Inference successful!")
            print("Output keys:", output.keys())
            print("Weather forecasts shape:", output["weather_forecasts"].shape)
            print("Hail predictions shape:", output["hail_predictions"].shape)
        except Exception as e:
            print(f"Error during inference: {e}")
            raise
    
    print("Model test completed successfully!")

if __name__ == "__main__":
    test_model()