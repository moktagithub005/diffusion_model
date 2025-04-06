#!/usr/bin/env python3
"""
Hail Detection System for Raspberry Pi

This script runs a lightweight version of the hail detection model on a Raspberry Pi
and communicates with an ESP module to trigger protective actions when hail is detected.

Usage:
    python run_service.py --config config/deploy_config.yaml
"""

import os
import argparse
import yaml
import time
import logging
import threading
import queue
import numpy as np
import torch
import torch.nn as nn
import requests
from datetime import datetime

from models.integrated_model import IntegratedHailModel, EnsembleHailModel
import utils.model_utils as model_utils
import utils.data_utils as data_utils
from utils.visualization import create_visualization
from esp_communication import ESPCommunicator

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hail_detection.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('hail_detection')

class OptimizedHailModel(nn.Module):
    """
    Optimized version of the Hail Detection model for Raspberry Pi deployment.
    This model uses quantization and other optimizations for faster inference.
    """
    def __init__(self, model_path, model_type="integrated", device="cpu"):
        super().__init__()
        self.model_type = model_type
        self.device = device
        
        # Load the model
        logger.info(f"Loading {model_type} model from {model_path}")
        
        if model_type == "integrated":
            self.model = IntegratedHailModel(
                input_channels=1,
                output_channels=1,
                hidden_dim=64,  # Reduced size for Pi
                context_channels=64,
                latent_channels=128,
                num_samples=1,  # Only one sample for deployment
                forecast_steps=6,
                output_shape=128,
                use_diffusion=False,  # No diffusion for faster inference
            )
        elif model_type == "ensemble":
            # For ensemble, we need to load multiple optimized models
            self.model = self._load_ensemble_model(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
            
        # Load state dict
        if model_type == "integrated":
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            
        # Optimize the model for inference
        self.model.eval()
        
        # Quantize the model for faster inference on CPU
        if device == "cpu":
            self.model = torch.quantization.quantize_dynamic(
                self.model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            
        logger.info(f"Model loaded and optimized for {device}")
    
    def _load_ensemble_model(self, model_path):
        """Load an ensemble of optimized models."""
        # Load ensemble configuration
        with open(model_path, 'r') as f:
            config = yaml.safe_load(f)
        
        # Load individual models
        models = []
        for model_config in config['models']:
            model = OptimizedHailModel(
                model_config['path'],
                model_config['type'],
                self.device
            ).model
            models.append(model)
        
        # Create ensemble
        ensemble = EnsembleHailModel(
            models=models,
            model_weights=config.get('weights', None),
            voting_threshold=config.get('voting_threshold', 2)
        )
        
        return ensemble
    
    def forward(self, x):
        with torch.no_grad():
            return self.model(x)
    
    @torch.no_grad()
    def predict(self, radar_data, threshold=0.5):
        """
        Make a prediction with the model.
        
        Args:
            radar_data: Radar data tensor [1, T, C, H, W]
            threshold: Threshold for hail detection
            
        Returns:
            Dictionary with prediction results
        """
        # Ensure data is on the correct device
        radar_data = radar_data.to(self.device)
        
        # Make prediction
        if hasattr(self.model, 'inference_with_threshold'):
            output = self.model.inference_with_threshold(
                radar_data, hail_threshold=threshold
            )
        else:
            output = self.model(radar_data)
            
            # Apply threshold if needed
            if 'hail_forecast' in output and 'hail_binary' not in output:
                output['hail_binary'] = (output['hail_forecast'] > threshold).float()
        
        return output

class HailDetectionService:
    """
    Service for continuous hail detection using radar data.
    
    This service:
    1. Fetches radar data at regular intervals
    2. Runs the hail detection model
    3. Communicates with ESP module if hail is detected
    4. Logs predictions and actions
    """
    def __init__(self, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Set up device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")
        
        # Load model
        self.model = OptimizedHailModel(
            self.config['model']['path'],
            self.config['model']['type'],
            self.device
        )
        
        # Set up ESP communication
        self.esp_communicator = ESPCommunicator(
            self.config['esp']['ip_address'],
            self.config['esp']['port']
        )
        
        # Set up data fetcher
        self.data_fetcher = RadarDataFetcher(
            self.config['data']['source'],
            self.config['data']['api_key'],
            self.config['data']['location'],
            self.config['data']['cache_dir']
        )
        
        # Detection parameters
        self.detection_threshold = self.config['detection']['threshold']
        self.consecutive_detections_required = self.config['detection']['consecutive_detections']
        self.detection_interval = self.config['detection']['interval_seconds']
        
        # Protection parameters
        self.protection_duration = self.config['protection']['duration_minutes'] * 60
        self.cooldown_period = self.config['protection']['cooldown_minutes'] * 60
        
        # State variables
        self.is_running = False
        self.is_protected = False
        self.protection_start_time = 0
        self.consecutive_detections = 0
        self.last_detection_time = 0
        
        # Create processing queue
        self.queue = queue.Queue()
        
        # Visualization settings
        self.enable_visualization = self.config['visualization']['enabled']
        self.visualization_path = self.config['visualization']['path']
        
        # Create visualization directory
        if self.enable_visualization:
            os.makedirs(self.visualization_path, exist_ok=True)
    
    def start(self):
        """Start the hail detection service."""
        if self.is_running:
            logger.warning("Service is already running")
            return
        
        self.is_running = True
        
        # Start the data fetching thread
        self.fetch_thread = threading.Thread(target=self._fetch_data_loop)
        self.fetch_thread.daemon = True
        self.fetch_thread.start()
        
        # Start the processing thread
        self.process_thread = threading.Thread(target=self._process_data_loop)
        self.process_thread.daemon = True
        self.process_thread.start()
        
        logger.info("Hail detection service started")
    
    def stop(self):
        """Stop the hail detection service."""
        self.is_running = False
        logger.info("Hail detection service stopped")
    
    def _fetch_data_loop(self):
        """Loop for fetching radar data."""
        while self.is_running:
            try:
                # Fetch radar data
                radar_data = self.data_fetcher.fetch_latest_data()
                
                # Put data in the queue
                self.queue.put(radar_data)
                
                # Sleep until next fetch
                time.sleep(self.detection_interval)
                
            except Exception as e:
                logger.error(f"Error fetching radar data: {e}")
                time.sleep(10)  # Shorter sleep on error
    
    def _process_data_loop(self):
        """Loop for processing radar data."""
        while self.is_running:
            try:
                # Get data from the queue
                radar_data = self.queue.get(timeout=30)
                
                # Process the data
                self._process_radar_data(radar_data)
                
                # Mark task as done
                self.queue.task_done()
                
            except queue.Empty:
                logger.warning("No data received in the last 30 seconds")
            except Exception as e:
                logger.error(f"Error processing radar data: {e}")
    
    def _process_radar_data(self, radar_data):
        """
        Process radar data to detect hail.
        
        Args:
            radar_data: Radar data in numpy format
        """
        logger.info("Processing radar data")
        
        # Convert to tensor
        radar_tensor = torch.from_numpy(radar_data).float().unsqueeze(0)  # Add batch dimension
        
        # Make prediction
        prediction = self.model.predict(radar_tensor, threshold=self.detection_threshold)
        
        # Extract hail binary prediction
        hail_binary = prediction['hail_binary'].cpu().numpy()
        
        # Check if hail is detected (any positive prediction in the frame)
        hail_detected = np.any(hail_binary > 0)
        
        # Update consecutive detection count
        current_time = time.time()
        if hail_detected:
            if current_time - self.last_detection_time < self.detection_interval * 2:
                self.consecutive_detections += 1
            else:
                self.consecutive_detections = 1
            
            self.last_detection_time = current_time
        else:
            self.consecutive_detections = 0
        
        # Log detection
        logger.info(f"Hail detected: {hail_detected}, Consecutive: {self.consecutive_detections}/{self.consecutive_detections_required}")
        
        # Create visualization if enabled
        if self.enable_visualization:
            self._create_visualization(radar_data, prediction, hail_detected)
        
        # Take action if needed
        self._handle_detection_state()
    
    def _handle_detection_state(self):
        """Handle the current detection state and take appropriate action."""
        current_time = time.time()
        
        # Check if protection is currently active
        if self.is_protected:
            # Check if protection duration has elapsed
            if current_time - self.protection_start_time > self.protection_duration:
                # End protection
                self._end_protection()
            return
        
        # Check if we have enough consecutive detections
        if self.consecutive_detections >= self.consecutive_detections_required:
            # Activate protection
            self._activate_protection()
    
    def _activate_protection(self):
        """Activate hail protection measures."""
        if self.is_protected:
            return
        
        logger.info("Activating hail protection")
        
        # Send command to ESP module
        success = self.esp_communicator.send_command("activate")
        
        if success:
            self.is_protected = True
            self.protection_start_time = time.time()
            logger.info(f"Protection activated for {self.protection_duration / 60} minutes")
        else:
            logger.error("Failed to activate protection")
    
    def _end_protection(self):
        """End hail protection measures."""
        if not self.is_protected:
            return
        
        logger.info("Ending hail protection")
        
        # Send command to ESP module
        success = self.esp_communicator.send_command("deactivate")
        
        if success:
            self.is_protected = False
            logger.info("Protection deactivated")
        else:
            logger.error("Failed to deactivate protection")
    
    def _create_visualization(self, radar_data, prediction, hail_detected):
        """Create visualization of the radar data and prediction."""
        try:
            # Create timestamp string
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Create filename
            filename = f"{timestamp}_{'hail' if hail_detected else 'normal'}.png"
            filepath = os.path.join(self.visualization_path, filename)
            
            # Create visualization
            create_visualization(
                radar_data,
                prediction,
                filepath,
                title=f"Hail Detection: {'POSITIVE' if hail_detected else 'NEGATIVE'}"
            )
            
            logger.debug(f"Visualization saved to {filepath}")
        
        except Exception as e:
            logger.error(f"Error creating visualization: {e}")

class RadarDataFetcher:
    """
    Class for fetching radar data from various sources.
    
    Supports:
    - Local files
    - Weather API
    - NEXRAD data
    - Custom data source
    """
    def __init__(self, source_type, api_key=None, location=None, cache_dir="./data/cache"):
        self.source_type = source_type
        self.api_key = api_key
        self.location = location
        self.cache_dir = cache_dir
        
        # Create cache directory
        os.makedirs(cache_dir, exist_ok=True)
        
        logger.info(f"Initialized radar data fetcher with source: {source_type}")
    
    def fetch_latest_data(self):
        """
        Fetch the latest radar data.
        
        Returns:
            Radar data as numpy array [T, C, H, W]
        """
        if self.source_type == "demo":
            return self._get_demo_data()
        elif self.source_type == "weather_api":
            return self._fetch_from_weather_api()
        elif self.source_type == "local_file":
            return self._load_from_local_file()
        elif self.source_type == "nexrad":
            return self._fetch_from_nexrad()
        else:
            raise ValueError(f"Unknown source type: {self.source_type}")
    
    def _get_demo_data(self):
        """Get demo radar data for testing."""
        logger.info("Using demo radar data")
        
        # Create simple demo data
        # Sequence of 4 frames, 1 channel, 128x128 resolution
        data = np.zeros((4, 1, 128, 128), dtype=np.float32)
        
        # Add some random radar echoes
        for t in range(4):
            # Create random cells
            num_cells = np.random.randint(3, 8)
            for _ in range(num_cells):
                x = np.random.randint(20, 108)
                y = np.random.randint(20, 108)
                intensity = np.random.uniform(0.3, 1.0)
                radius = np.random.randint(5, 15)
                
                # Create Gaussian cell
                y_grid, x_grid = np.ogrid[-y:128-y, -x:128-x]
                mask = x_grid*x_grid + y_grid*y_grid <= radius*radius
                data[t, 0, mask] = intensity
        
        # Add noise
        data += np.random.uniform(0, 0.1, size=data.shape)
        
        # Clip to valid range
        data = np.clip(data, 0, 1)
        
        return data
    
    def _fetch_from_weather_api(self):
        """Fetch radar data from a weather API."""
        logger.info(f"Fetching radar data from weather API for location: {self.location}")
        
        try:
            # Example API URL (replace with actual API endpoint)
            url = f"https://api.weatherapi.com/v1/radar.json?key={self.api_key}&q={self.location}"
            
            # Make request
            response = requests.get(url)
            response.raise_for_status()
            
            # Process response
            data = response.json()
            
            # Convert API response to radar data
            # This will be specific to the API format
            radar_data = self._process_api_response(data)
            
            return radar_data
            
        except Exception as e:
            logger.error(f"Error fetching from weather API: {e}")
            # Return demo data as fallback
            return self._get_demo_data()
    
    def _process_api_response(self, data):
        """
        Process API response to extract radar data.
        
        This is a placeholder - implementation will depend on the specific API.
        """
        # Placeholder implementation
        return self._get_demo_data()
    
    def _load_from_local_file(self):
        """Load radar data from a local file."""
        logger.info("Loading radar data from local file")
        
        try:
            # Find the most recent file in the cache directory
            files = os.listdir(self.cache_dir)
            radar_files = [f for f in files if f.endswith('.npy')]
            
            if not radar_files:
                logger.warning("No radar files found in cache directory")
                return self._get_demo_data()
            
            # Sort by modification time (most recent first)
            radar_files.sort(key=lambda f: os.path.getmtime(os.path.join(self.cache_dir, f)), reverse=True)
            
            # Load the most recent file
            file_path = os.path.join(self.cache_dir, radar_files[0])
            radar_data = np.load(file_path)
            
            logger.info(f"Loaded radar data from {file_path}")
            return radar_data
            
        except Exception as e:
            logger.error(f"Error loading from local file: {e}")
            # Return demo data as fallback
            return self._get_demo_data()
    
    def _fetch_from_nexrad(self):
        """Fetch radar data from NEXRAD."""
        logger.info("Fetching radar data from NEXRAD")
        
        try:
            # This would require a library like nexradaws
            # Placeholder implementation
            return self._get_demo_data()
            
        except Exception as e:
            logger.error(f"Error fetching from NEXRAD: {e}")
            # Return demo data as fallback
            return self._get_demo_data()


def main():
    """Main entry point for the hail detection service."""
    parser = argparse.ArgumentParser(description="Hail Detection Service for Raspberry Pi")
    parser.add_argument("--config", type=str, default="config/deploy_config.yaml", 
                        help="Path to configuration file")
    args = parser.parse_args()
    
    # Initialize and start the service
    service = HailDetectionService(args.config)
    service.start()
    
    try:
        # Keep the main thread alive
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down service")
        service.stop()

if __name__ == "__main__":
    main()