import streamlit as st
import torch
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import io
import sys
import pandas as pd
from datetime import datetime
from PIL import Image
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
from matplotlib.colors import ListedColormap

# Add parent directory to path for module import
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import your model classes
from models.integrated_model import IntegratedHailModel, EnsembleHailModel
from deploy.raspberry_pi.esp_communication import ESPCommunicator

# Constants
MODEL_PATH = None  # Not using pre-trained weights for demo
DEFAULT_THRESHOLD = 0.5

# Streamlit config
st.set_page_config(
    page_title="Hail Detection System",
    page_icon="🌦️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-title {
        font-size: 40px !important;
        text-align: center;
        margin-bottom: 30px;
    }
    .sub-title {
        font-size: 24px !important;
        margin-top: 20px;
        margin-bottom: 10px;
    }
    .highlight {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
    }
    .detection-positive {
        color: #ff4b4b;
        font-weight: bold;
        font-size: 20px;
    }
    .detection-negative {
        color: #0068c9;
        font-weight: bold;
        font-size: 20px;
    }
    .status-circle {
        display: inline-block;
        width: 15px;
        height: 15px;
        border-radius: 50%;
        margin-right: 5px;
    }
    .status-active {
        background-color: #00cc66;
    }
    .status-inactive {
        background-color: #ff4b4b;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("<h1 class='main-title'>Hail Detection System</h1>", unsafe_allow_html=True)

# Initialize session state
if 'model_loaded' not in st.session_state:
    st.session_state.model_loaded = False
if 'protection_active' not in st.session_state:
    st.session_state.protection_active = False
if 'detection_history' not in st.session_state:
    st.session_state.detection_history = []
if 'last_detection_time' not in st.session_state:
    st.session_state.last_detection_time = None

# Sidebar
st.sidebar.markdown("## System Control")

model_type = st.sidebar.selectbox(
    "Select Model Type",
    ["Integrated Model", "DGMR", "SteamCast", "Ensemble"],
    index=0
)

# Add this right after the model_type dropdown but before the threshold slider
scenario = st.sidebar.selectbox(
    "Demo Scenario",
    ["None", "Approaching Severe Storm", "False Alarm", "Early Detection Success"],
    index=0
)

if scenario != "None":
    # Override demo mode when scenario is selected
    st.sidebar.info(f"Using pre-defined scenario: {scenario}")
    demo_mode = "Custom"

threshold = st.sidebar.slider(
    "Detection Threshold",
    0.0, 1.0, value=DEFAULT_THRESHOLD, step=0.05
)

demo_mode = st.sidebar.selectbox(
    "Demo Mode",
    ["Normal Weather", "Hail Storm", "Mixed Conditions"],
    index=0
)

# ESP connection settings
st.sidebar.markdown("## Hardware Control")
esp_ip = st.sidebar.text_input("ESP IP Address", "192.168.1.100")
esp_port = st.sidebar.number_input("ESP Port", value=8080)

if st.sidebar.button("Deploy Protection Manually"):
    st.sidebar.success("Manual protection deployment initiated!")
    st.session_state.protection_active = True
    st.session_state.last_detection_time = datetime.now()

if st.sidebar.button("Retract Protection Manually"):
    st.sidebar.success("Manual protection retraction initiated!")
    st.session_state.protection_active = False



# Visualization functions
def visualize_radar_sequence(radar_data, fig_size=(12, 3)):
    """Simple visualization function for radar data."""
    # Create a blank image
    fig, axes = plt.subplots(1, 4, figsize=fig_size)
    
    # Generate simple images
    for i, ax in enumerate(axes):
        # Create a simple gradient image
        img = np.zeros((128, 128))
        for y in range(128):
            for x in range(128):
                img[y, x] = (x + y) / 256
        
        # Add some random cells to make it look like radar
        num_cells = np.random.randint(3, 8)
        for _ in range(num_cells):
            cx, cy = np.random.randint(20, 108, 2)
            radius = np.random.randint(5, 20)
            intensity = np.random.uniform(0.5, 1.0)
            
            # Create a circular pattern
            for y in range(max(0, cy-radius), min(128, cy+radius)):
                for x in range(max(0, cx-radius), min(128, cx+radius)):
                    dist = np.sqrt((y-cy)**2 + (x-cx)**2)
                    if dist < radius:
                        img[y, x] = intensity * (1 - dist/radius)
        
        ax.imshow(img, cmap='jet', vmin=0, vmax=1)
        ax.set_title(f"T-{4-i}")
        ax.axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def visualize_prediction(weather_forecast, hail_prediction, fig_size=(12, 6)):
    """Simple visualization function for forecasts."""
    # Create a blank image
    fig, axes = plt.subplots(2, 6, figsize=fig_size)
    
    # Generate simple images for weather forecast
    for i in range(6):
        # Create a simple gradient image
        img = np.zeros((128, 128))
        for y in range(128):
            for x in range(128):
                img[y, x] = (x + y) / 256
        
        # Add some random cells
        num_cells = np.random.randint(3, 8)
        for _ in range(num_cells):
            cx, cy = np.random.randint(20, 108, 2)
            radius = np.random.randint(5, 20)
            intensity = np.random.uniform(0.5, 1.0)
            
            # Create a circular pattern
            for y in range(max(0, cy-radius), min(128, cy+radius)):
                for x in range(max(0, cx-radius), min(128, cx+radius)):
                    dist = np.sqrt((y-cy)**2 + (x-cx)**2)
                    if dist < radius:
                        img[y, x] = intensity * (1 - dist/radius)
        
        axes[0, i].imshow(img, cmap='jet', vmin=0, vmax=1)
        axes[0, i].set_title(f"T+{i+1}")
        axes[0, i].axis('off')
    
    # Generate simple images for hail prediction
    for i in range(6):
        # Create a simple hail probability image
        img = np.zeros((128, 128))
        
        # Only add hail signatures in "Hail Storm" mode or in some cases for "Mixed Conditions"
        if demo_mode == "Hail Storm" or (demo_mode == "Mixed Conditions" and np.random.random() > 0.5):
            # Add a few hail cells
            num_cells = np.random.randint(1, 4)
            for _ in range(num_cells):
                cx, cy = np.random.randint(20, 108, 2)
                radius = np.random.randint(5, 15)
                intensity = np.random.uniform(0.7, 1.0)
                
                # Create a circular pattern
                for y in range(max(0, cy-radius), min(128, cy+radius)):
                    for x in range(max(0, cx-radius), min(128, cx+radius)):
                        dist = np.sqrt((y-cy)**2 + (x-cx)**2)
                        if dist < radius:
                            img[y, x] = intensity * (1 - dist/radius)
        
        axes[1, i].imshow(img, cmap='RdYlBu_r', vmin=0, vmax=1)
        axes[1, i].set_title(f"Hail T+{i+1}")
        axes[1, i].axis('off')
    
    plt.tight_layout()
    
    # Convert plot to image
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plt.close(fig)
    
    return buf

def create_radar_sweep_animation(data, fig_size=(12, 3)):
    """Create a radar sweep animation to make visualizations more authentic."""
    # Create figure
    fig, axes = plt.subplots(1, 4, figsize=fig_size)
    
    # Set up base images
    images = []
    for i, ax in enumerate(axes):
        # Display the base radar image
        im = ax.imshow(data[i, 0], cmap='jet', vmin=0, vmax=1)
        ax.set_title(f"T-{4-i}")
        ax.axis('off')
        images.append(im)
    
    plt.tight_layout()
    
    # Create animation frames
    def update(frame):
        # Calculate angle for sweep line
        angle = frame * 6  # 6 degrees per frame
        
        # Update all subplots with a sweep line
        for i, ax in enumerate(axes):
            # Clear previous lines
            for line in ax.lines:
                line.remove()
            
            # Calculate line endpoint
            center = (64, 64)  # Center of 128x128 image
            radius = 70  # Line length
            end_x = center[0] + radius * np.cos(np.radians(angle))
            end_y = center[1] + radius * np.sin(np.radians(angle))
            
            # Draw sweep line
            ax.plot([center[0], end_x], [center[1], end_y], 'white', linewidth=1.5, alpha=0.7)
            
            # Add sweep circle
            circle = plt.Circle(center, 70, fill=False, color='white', alpha=0.3, linestyle='--')
            ax.add_patch(circle)
        
        return images + [line for ax in axes for line in ax.lines] + [patch for ax in axes for patch in ax.patches]
    
    # Create animation
    from matplotlib.animation import FuncAnimation
    ani = FuncAnimation(fig, update, frames=60, interval=50, blit=True)
    
    # Convert to HTML for Streamlit
    from IPython.display import HTML
    plt.close()
    
    return ani

def compare_model_predictions(data, threshold=0.5, fig_size=(15, 8)):
    """Create side-by-side comparisons of different model predictions."""
    fig, axes = plt.subplots(3, 2, figsize=fig_size)
    
    # Generate slightly different predictions for each model type to show differences
    # DGMR - good at temporal patterns, may miss stationary hail
    dgmr_pred = np.zeros((128, 128))
    # Moving cells in a sequence (simulating temporal sensitivity)
    for i in range(3):
        x, y = 40 + i*15, 50 + i*10
        radius = 15
        for cy in range(max(0, y-radius), min(128, y+radius)):
            for cx in range(max(0, x-radius), min(128, x+radius)):
                dist = np.sqrt((cy-y)**2 + (cx-x)**2)
                if dist < radius:
                    dgmr_pred[cy, cx] = max(dgmr_pred[cy, cx], 0.8 * (1 - dist/radius))
    
    # SteamCast - good at spatial patterns, high resolution
    steamcast_pred = np.zeros((128, 128))
    # Complex spatial patterns (simulating texture sensitivity)
    for i in range(4):
        x, y = np.random.randint(30, 100, 2)
        radius = np.random.randint(5, 15)
        intensity = np.random.uniform(0.7, 0.9)
        
        for cy in range(max(0, y-radius), min(128, y+radius)):
            for cx in range(max(0, x-radius), min(128, x+radius)):
                dist = np.sqrt((cy-y)**2 + (cx-x)**2)
                if dist < radius:
                    # Create texture pattern
                    texture = np.sin(cx/3) * np.cos(cy/3) * 0.2 + 0.8
                    steamcast_pred[cy, cx] = max(steamcast_pred[cy, cx], 
                                               intensity * texture * (1 - dist/radius))
    
    # Integrated model - combines strengths
    integrated_pred = np.zeros((128, 128))
    # Moving cells with texture
    for i in range(3):
        x, y = 40 + i*15, 50 + i*10
        radius = 15
        for cy in range(max(0, y-radius), min(128, y+radius)):
            for cx in range(max(0, x-radius), min(128, x+radius)):
                dist = np.sqrt((cy-y)**2 + (cx-x)**2)
                if dist < radius:
                    # Add both temporal and texture patterns
                    texture = np.sin(cx/3) * np.cos(cy/3) * 0.2 + 0.8
                    integrated_pred[cy, cx] = max(integrated_pred[cy, cx], 
                                               0.9 * texture * (1 - dist/radius))
    
    # Display input data
    im = axes[0, 0].imshow(data[-1, 0], cmap='jet', vmin=0, vmax=1)
    axes[0, 0].set_title("Input Radar Data")
    axes[0, 0].axis('off')
    
    # Display prediction overlays
    # DGMR
    axes[0, 1].imshow(data[-1, 0], cmap='jet', vmin=0, vmax=1)
    dgmr_masked = np.ma.masked_where(dgmr_pred < threshold, dgmr_pred)
    axes[0, 1].imshow(dgmr_masked, cmap='autumn_r', vmin=0, vmax=1, alpha=0.7)
    axes[0, 1].set_title("DGMR Prediction")
    axes[0, 1].axis('off')
    
    # SteamCast
    axes[1, 0].imshow(data[-1, 0], cmap='jet', vmin=0, vmax=1)
    steam_masked = np.ma.masked_where(steamcast_pred < threshold, steamcast_pred)
    axes[1, 0].imshow(steam_masked, cmap='autumn_r', vmin=0, vmax=1, alpha=0.7)
    axes[1, 0].set_title("SteamCast Prediction")
    axes[1, 0].axis('off')
    
    # Integrated
    axes[1, 1].imshow(data[-1, 0], cmap='jet', vmin=0, vmax=1)
    int_masked = np.ma.masked_where(integrated_pred < threshold, integrated_pred)
    axes[1, 1].imshow(int_masked, cmap='autumn_r', vmin=0, vmax=1, alpha=0.7)
    axes[1, 1].set_title("Integrated Model Prediction")
    axes[1, 1].axis('off')
    
    # Hail probability maps
    cmap = plt.cm.RdYlBu_r
    im2 = axes[2, 0].imshow(dgmr_pred, cmap=cmap, vmin=0, vmax=1)
    axes[2, 0].set_title("DGMR Hail Probability")
    axes[2, 0].axis('off')
    
    axes[2, 1].imshow(integrated_pred, cmap=cmap, vmin=0, vmax=1)
    axes[2, 1].set_title("Integrated Hail Probability")
    axes[2, 1].axis('off')
    
    # Add colorbar
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(im2, cax=cbar_ax)
    cbar.set_label('Hail Probability')
    
    plt.tight_layout(rect=[0, 0, 0.9, 1])
    
    # Convert to image for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    
    return buf

def visualize_hail_risk(data, hail_pred, threshold=0.5, fig_size=(12, 8)):
    """Create visually striking visualization of hail risk areas."""
    fig, axes = plt.subplots(2, 1, figsize=fig_size, gridspec_kw={'height_ratios': [3, 1]})
    
    # Create base radar image
    im1 = axes[0].imshow(data[-1, 0], cmap='Blues', vmin=0, vmax=1)
    
    # Create hail risk overlay with vibrant colors
    # Define custom colormap for high-contrast
    from matplotlib.colors import ListedColormap
    hail_colors = plt.cm.get_cmap('RdYlBu_r')(np.linspace(0.3, 1.0, 256))
    hail_cmap = ListedColormap(hail_colors)
    
    # Make low values transparent
    hail_masked = np.ma.masked_where(hail_pred < threshold, hail_pred)
    im2 = axes[0].imshow(hail_masked, cmap=hail_cmap, vmin=0, vmax=1, alpha=0.8)
    
    # Add contour lines to emphasize risk boundaries
    contour_levels = np.linspace(threshold, 1.0, 5)
    cs = axes[0].contour(hail_pred, levels=contour_levels, colors='red', linewidths=1.5, alpha=0.7)
    
    # Add dynamic annotations
    high_risk_threshold = 0.8
    high_risk_areas = np.where(hail_pred > high_risk_threshold)
    if len(high_risk_areas[0]) > 0:
        # Find centroid of high risk area
        y_center = int(np.mean(high_risk_areas[0]))
        x_center = int(np.mean(high_risk_areas[1]))
        axes[0].annotate("HIGH RISK", xy=(x_center, y_center), 
                         xytext=(x_center+20, y_center-20),
                         arrowprops=dict(facecolor='red', shrink=0.05),
                         color='white', fontsize=12, fontweight='bold',
                         bbox=dict(boxstyle="round,pad=0.3", fc="red", ec="none", alpha=0.8))
    
    # Add radar-like decorations
    # Range rings
    center = (64, 64)
    for r in [20, 40, 60]:
        circle = plt.Circle(center, r, fill=False, color='white', alpha=0.3, linestyle='--')
        axes[0].add_patch(circle)
    
    # Add compass directions
    for direction, (x, y) in [('N', (64, 5)), ('S', (64, 123)), ('E', (123, 64)), ('W', (5, 64))]:
        axes[0].text(x, y, direction, color='white', ha='center', va='center', fontweight='bold')
    
    axes[0].set_title("Weather Radar with Hail Risk Overlay", fontsize=14)
    axes[0].axis('off')
    
    # Risk level legend/key
    axes[1].axis('off')
    risk_levels = ["None", "Low", "Moderate", "High", "Extreme"]
    risk_colors = [plt.cm.get_cmap('RdYlBu_r')(i) for i in [0.0, 0.3, 0.6, 0.8, 0.95]]
    
    for i, (level, color) in enumerate(zip(risk_levels, risk_colors)):
        rect = plt.Rectangle((0.1 + i*0.18, 0.4), 0.15, 0.3, fc=color)
        axes[1].add_patch(rect)
        axes[1].text(0.175 + i*0.18, 0.2, level, ha='center')
    
    axes[1].text(0.5, 0.8, "Hail Risk Levels", ha='center', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Convert to image for Streamlit
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    buf.seek(0)
    plt.close(fig)
    
    return buf

def display_radar_animation(data):
    """Display radar animation in Streamlit."""
    animation = create_radar_sweep_animation(data)
    
    # Save animation as animated GIF
    animation_file = "radar_animation.gif"
    animation.save(animation_file, writer='pillow', fps=10)
    
    # Display in Streamlit
    st.image(animation_file, caption="Real-time Radar Sweep", use_container_width=True)

def show_protection_animation():
    """Display animation of protection system deployment."""
    # Create a simple animation showing nets being deployed
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Draw background - simple orchard
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 6)
    
    # Draw trees
    for x in range(1, 10, 2):
        for y in range(1, 6, 2):
            # Tree trunk
            ax.add_patch(plt.Rectangle((x-0.1, y-0.5), 0.2, 0.5, fc='brown'))
            # Tree canopy
            ax.add_patch(plt.Circle((x, y), 0.5, fc='green'))
    
    # Title and axis settings
    ax.set_title("Protection System Deployment")
    ax.axis('off')
    
    plt.tight_layout()
    
    # Create static image for initial state
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img1 = buf.getvalue()
    
    # Now add netting
    # Horizontal support lines
    for y in range(1, 6, 2):
        ax.plot([0, 10], [y+0.6, y+0.6], 'k-', linewidth=1)
    
    # Vertical poles
    for x in range(0, 11, 2):
        ax.plot([x, x], [0, 6], 'k-', linewidth=2)
    
    # Netting
    for x in range(0, 10, 0.5):
        for y in range(0, 6, 0.5):
            if (x + y) % 1 == 0:
                ax.plot([x, x+0.5], [y, y+0.5], 'b-', alpha=0.3, linewidth=0.5)
                ax.plot([x+0.5, x], [y, y+0.5], 'b-', alpha=0.3, linewidth=0.5)
    
    # Update title
    ax.set_title("Protection System Deployed")
    
    # Save second state
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    img2 = buf.getvalue()
    plt.close()
    
    # Display animation in Streamlit
    st.image(img1, use_container_width=True)
    time.sleep(1)
    st.image(img2, use_container_width=True)

def get_demo_data(mode="Normal Weather"):
    """Get demo radar data based on selected mode."""
    # Sequence of 4 frames, 1 channel, 128x128 resolution
    data = np.zeros((4, 1, 128, 128), dtype=np.float32)
    
    if mode == "Hail Storm":
        # Create intense storm cells with potential hail signatures
        for t in range(4):
            # Create strong cells
            num_cells = np.random.randint(3, 7)
            for _ in range(num_cells):
                x = np.random.randint(20, 108)
                y = np.random.randint(20, 108)
                intensity = np.random.uniform(0.75, 1.0)  # Higher intensity for hail
                radius = np.random.randint(10, 20)
                
                # Create Gaussian cell
                y_grid, x_grid = np.ogrid[-y:128-y, -x:128-x]
                mask = x_grid*x_grid + y_grid*y_grid <= radius*radius
                data[t, 0, mask] = intensity
    
    elif mode == "Mixed Conditions":
        # Mix of normal cells and some potential hail
        for t in range(4):
            # Create random cells
            num_cells = np.random.randint(4, 8)
            for i in range(num_cells):
                x = np.random.randint(20, 108)
                y = np.random.randint(20, 108)
                # Some cells are intense (potential hail), others are normal
                if i < 2:  # Make just a couple of cells intense
                    intensity = np.random.uniform(0.7, 0.9)
                else:
                    intensity = np.random.uniform(0.2, 0.5)
                radius = np.random.randint(5, 15)
                
                # Create Gaussian cell
                y_grid, x_grid = np.ogrid[-y:128-y, -x:128-x]
                mask = x_grid*x_grid + y_grid*y_grid <= radius*radius
                data[t, 0, mask] = intensity
    
    else:  # Normal Weather
        # Create typical radar echoes without hail signatures
        for t in range(4):
            # Create random cells
            num_cells = np.random.randint(3, 7)
            for _ in range(num_cells):
                x = np.random.randint(20, 108)
                y = np.random.randint(20, 108)
                intensity = np.random.uniform(0.1, 0.4)  # Lower intensity for normal weather
                radius = np.random.randint(5, 12)
                
                # Create Gaussian cell
                y_grid, x_grid = np.ogrid[-y:128-y, -x:128-x]
                mask = x_grid*x_grid + y_grid*y_grid <= radius*radius
                data[t, 0, mask] = intensity
    
    # Add noise
    data += np.random.uniform(0, 0.05, size=data.shape)
    
    # Clip to valid range
    data = np.clip(data, 0, 1)
    
    return data

def update_detection_history(is_detected, confidence, threshold):
    """Update the detection history in session state."""
    timestamp = datetime.now()
    st.session_state.detection_history.append({
        "timestamp": timestamp,
        "detected": is_detected,
        "confidence": confidence,
        "threshold": threshold
    })
    
    # Keep only the last 10 detections
    if len(st.session_state.detection_history) > 10:
        st.session_state.detection_history = st.session_state.detection_history[-10:]
    
    if is_detected:
        st.session_state.last_detection_time = timestamp

# Add this function AFTER get_demo_data
def load_scenario(scenario_name):
    """Load a pre-defined scenario for demonstration."""
    if scenario_name == "Approaching Severe Storm":
        # Sequence showing increasing intensity with clear hail signatures
        data = np.zeros((4, 1, 128, 128), dtype=np.float32)
        # Create intensifying storm cells
        for t in range(4):
            # Add increasing number of cells
            num_cells = 3 + t
            intensity_factor = 0.5 + t * 0.15  # Increasing intensity
            for i in range(num_cells):
                x = 30 + i*15 + t*5  # Cells moving across frame
                y = 40 + i*10
                radius = 10 + t*2
                intensity = np.random.uniform(0.5, 0.9) * intensity_factor
                
                # Create Gaussian cell
                y_grid, x_grid = np.ogrid[-y:128-y, -x:128-x]
                mask = x_grid*x_grid + y_grid*y_grid <= radius*radius
                data[t, 0, mask] = intensity
        
        hail_prob = 0.85  # High probability of hail
    
    elif scenario_name == "False Alarm":
        # Sequence with intense but non-hail producing cells
        data = np.zeros((4, 1, 128, 128), dtype=np.float32)
        # Create large stratiform precip with embedded convection
        for t in range(4):
            # Background precipitation
            for y in range(128):
                for x in range(128):
                    data[t, 0, y, x] = np.random.uniform(0.1, 0.3)
            
            # Add strong cells but with characteristics not typical for hail
            for i in range(5):
                x = np.random.randint(20, 108)
                y = np.random.randint(20, 108)
                radius = np.random.randint(15, 25)  # Large cells
                intensity = np.random.uniform(0.5, 0.7)  # Moderate intensity
                
                y_grid, x_grid = np.ogrid[-y:128-y, -x:128-x]
                mask = x_grid*x_grid + y_grid*y_grid <= radius*radius
                data[t, 0, mask] = intensity
        
        hail_prob = 0.45  # Moderate probability but below typical threshold
    
    elif scenario_name == "Early Detection Success":
        # Sequence with subtle early hail signatures
        data = np.zeros((4, 1, 128, 128), dtype=np.float32)
        # First two frames have subtle signatures
        for t in range(2):
            num_cells = 2
            for i in range(num_cells):
                x = 40 + i*30
                y = 50 + i*20
                radius = 8
                intensity = 0.4 + t*0.15
                
                # Create cell with subtle bounded weak echo region (BWER)
                y_grid, x_grid = np.ogrid[-y:128-y, -x:128-x]
                r_squared = x_grid*x_grid + y_grid*y_grid
                
                # Create ring-like structure (BWER signature)
                inner_mask = r_squared <= (radius/2)*(radius/2)
                outer_mask = r_squared <= radius*radius
                ring_mask = outer_mask & ~inner_mask
                
                data[t, 0, ring_mask] = intensity
                data[t, 0, inner_mask] = intensity * 0.3  # Weak echo region
        
        # Last two frames show rapid intensification
        for t in range(2, 4):
            num_cells = 2
            for i in range(num_cells):
                x = 40 + i*30
                y = 50 + i*20
                radius = 12
                intensity = 0.7 + (t-2)*0.2
                
                # Create cell with clear hail signature
                y_grid, x_grid = np.ogrid[-y:128-y, -x:128-x]
                r_squared = x_grid*x_grid + y_grid*y_grid
                
                # Three-body scatter spike signature
                mask = r_squared <= radius*radius
                data[t, 0, mask] = intensity
                
                # Add spike in one direction
                spike_length = 20
                for j in range(spike_length):
                    sx = x + j
                    sy = y + j
                    if sx < 128 and sy < 128:
                        data[t, 0, sy, sx] = intensity * (1 - j/spike_length)
        
        hail_prob = 0.75  # High probability that becomes evident only in later frames
    
    else:  # Default normal weather
        data = get_demo_data("Normal Weather")
        hail_prob = 0.2
    
    # Add some noise
    data += np.random.uniform(0, 0.05, size=data.shape)
    data = np.clip(data, 0, 1)
    
    return data, hail_prob

def show_roi_calculator():
    """Display ROI calculator for hail protection system."""
    st.subheader("ROI Calculator")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### System Costs")
        system_cost = st.number_input("Installation Cost ($)", value=50000, step=5000)
        annual_maintenance = st.number_input("Annual Maintenance ($)", value=2000, step=500)
        lifespan = st.number_input("System Lifespan (years)", value=10, step=1)
        
    with col2:
        st.markdown("### Damage Prevention")
        orchard_size = st.number_input("Orchard Size (acres)", value=20, step=5)
        crop_value = st.number_input("Crop Value per Acre ($)", value=15000, step=1000)
        hail_probability = st.slider("Annual Hail Probability (%)", 0, 100, 20)
        damage_reduction = st.slider("Damage Reduction with System (%)", 0, 100, 90)
    
    # Calculate ROI
    total_cost = system_cost + (annual_maintenance * lifespan)
    annual_potential_damage = orchard_size * crop_value * (hail_probability / 100)
    annual_savings = annual_potential_damage * (damage_reduction / 100)
    total_savings = annual_savings * lifespan
    net_savings = total_savings - total_cost
    roi_percent = (net_savings / total_cost) * 100 if total_cost > 0 else 0
    payback_years = total_cost / annual_savings if annual_savings > 0 else float('inf')
    
    # Display results
    st.markdown("### ROI Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total System Cost", f"${total_cost:,.2f}")
        st.metric("Total Savings", f"${total_savings:,.2f}")
    
    with col2:
        st.metric("Net Savings", f"${net_savings:,.2f}")
        st.metric("ROI", f"{roi_percent:.1f}%")
    
    with col3:
        st.metric("Payback Period", f"{payback_years:.1f} years")
        st.metric("Annual Savings", f"${annual_savings:,.2f}")
    
    # Visualization
    if net_savings > 0:
        st.success(f"The system will pay for itself in {payback_years:.1f} years and save approximately ${net_savings:,.2f} over its lifetime.")
    else:
        st.warning(f"Based on current inputs, the system costs more than the expected savings. Consider adjusting parameters.")

# Cached model loader
@st.cache_resource
def load_model(model_type="Integrated Model"):
    """Load model for demonstration purposes."""
    try:
        if model_type == "SteamCast":
            # Create a model with SteamCast characteristics
            model = object()  # Just a placeholder
            st.info("Using SteamCast model for demonstration")
        elif model_type == "DGMR":
            # Create a model with DGMR characteristics
            model = object()  # Just a placeholder
            st.info("Using DGMR model for demonstration")
        elif model_type == "Ensemble":
            # Create an ensemble model
            model = object()  # Just a placeholder
            st.info("Using ensemble model for demonstration")
        else:
            # Create an integrated model
            model = object()  # Just a placeholder
            st.info("Using integrated model for demonstration")
        
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return object()  # Just a placeholder

# Main app layout
tabs = st.tabs(["Live Detection", "History", "System Status", "ROI Calculator"])

with tabs[0]:
    # Live Detection tab
    st.markdown("<h2 class='sub-title'>Real-time Hail Detection</h2>", unsafe_allow_html=True)
    
    # Load model
    model = load_model(model_type)
    
    # Create columns for display
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("<h3>Input Radar Data</h3>", unsafe_allow_html=True)

        # Replace with this conditional code
        if scenario != "None":
             radar_data, scenario_hail_prob = load_scenario(scenario)
        else:
             radar_data = get_demo_data(demo_mode)
             scenario_hail_prob = None
        
        
        
        
        # Display standard visualization
        radar_img_buf = visualize_radar_sequence(radar_data)
        st.image(radar_img_buf, caption="Input Radar Sequence", use_container_width=True)
        
        # Add radar sweep animation
        if st.checkbox("Show radar animation", value=False):
            try:
                display_radar_animation(radar_data)
            except Exception as e:
                st.error(f"Could not display animation: {e}")
                
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("<h3>Model Predictions</h3>", unsafe_allow_html=True)
        
        # Run model prediction
        if st.button("Run Detection"):
            with st.spinner("Running hail detection..."):
                # For demo purposes, simulate predictions
                weather_forecast = radar_data
                
                # Generate simulated hail prediction based on demo mode
                if demo_mode == "Hail Storm":
                    # Higher chance of hail detection
                    confidence = np.random.uniform(0.7, 0.95)
                    # Create hail pattern
                    hail_pred = np.zeros((128, 128))
                    for _ in range(3):
                        x, y = np.random.randint(30, 100, 2)
                        r = np.random.randint(10, 20)
                        for cy in range(max(0, y-r), min(128, y+r)):
                            for cx in range(max(0, x-r), min(128, x+r)):
                                dist = np.sqrt((cy-y)**2 + (cx-x)**2)
                                if dist < r:
                                    hail_pred[cy, cx] = max(
                                        hail_pred[cy, cx], 
                                        confidence * (1 - dist/r)
                                    )
                elif demo_mode == "Mixed Conditions":
                    # Medium chance of hail detection
                    confidence = np.random.uniform(0.4, 0.8)
                    # Create hail pattern
                    hail_pred = np.zeros((128, 128))
                    for _ in range(1):
                        x, y = np.random.randint(30, 100, 2)
                        r = np.random.randint(5, 15)
                        for cy in range(max(0, y-r), min(128, y+r)):
                            for cx in range(max(0, x-r), min(128, x+r)):
                                dist = np.sqrt((cy-y)**2 + (cx-x)**2)
                                if dist < r:
                                    hail_pred[cy, cx] = max(
                                        hail_pred[cy, cx], 
                                        confidence * (1 - dist/r)
                                    )
                else:  # Normal Weather
                    # Lower chance of hail detection
                    confidence = np.random.uniform(0.1, 0.5)
                    hail_pred = np.zeros((128, 128))
                
                # Apply threshold to determine detection
                is_hail_detected = np.max(hail_pred) > threshold
                
                # Standard visualization
                pred_img_buf = visualize_prediction(weather_forecast, hail_pred.reshape(1, 1, 128, 128))
                st.image(pred_img_buf, caption="Weather Forecast and Hail Prediction", use_container_width=True)
                
                # Show enhanced visualizations
                st.subheader("Enhanced Visualizations")
                
                # Show hail risk visualization
                hail_risk_buf = visualize_hail_risk(weather_forecast, hail_pred, threshold)
                st.image(hail_risk_buf, caption="High-contrast Hail Risk Visualization", use_container_width=True)
                
                # Model comparison visualizations
                st.subheader("Model Comparison")
                comparison_buf = compare_model_predictions(weather_forecast, threshold)
                st.image(comparison_buf, caption="Comparison of Model Predictions", use_container_width=True)
                
                # Show detection result
                if is_hail_detected:
                    st.markdown("<p class='detection-positive'>⚠️ HAIL DETECTED!</p>", unsafe_allow_html=True)
                    st.markdown(f"Max confidence: {np.max(hail_pred):.2f}")
                    
                    # Activate protection if not already active
                    if not st.session_state.protection_active:
                        st.session_state.protection_active = True
                        st.success("Protection system activated!")
                        with st.spinner("Deploying protection system..."):
                             show_protection_animation()
                        
                        # Add animation of protection deployment
                        st.markdown("### Protection System Deployed")
                        # Simple ASCII animation
                        for i in range(5):
                            progress = "▓" * i + "░" * (5-i)
                            st.text(f"Deploying protective nets: [{progress}]")
                            time.sleep(0.2)
                        st.success("Protection fully deployed!")
                else:
                    st.markdown("<p class='detection-negative'>✓ No hail detected</p>", unsafe_allow_html=True)
                    st.markdown(f"Max confidence: {np.max(hail_pred):.2f}")
        
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[1]:
    # History tab
    st.markdown("<h2 class='sub-title'>Detection History</h2>", unsafe_allow_html=True)
    
    if not st.session_state.detection_history:
        st.info("No detection history available yet. Run some detections first.")
    else:
        # Create a DataFrame for the history
        history_df = pd.DataFrame(st.session_state.detection_history)
        history_df["Time"] = history_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
        history_df["Detection"] = history_df["detected"].apply(lambda x: "✅ HAIL DETECTED" if x else "❌ No Hail")
        history_df["Confidence"] = history_df["confidence"].apply(lambda x: f"{x:.2f}")
        history_df["Threshold"] = history_df["threshold"].apply(lambda x: f"{x:.2f}")
        
        # Display the history
        st.dataframe(
            history_df[["Time", "Detection", "Confidence", "Threshold"]],
            use_container_width=True
        )
        
        # Plot history
        if len(history_df) > 1:
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(history_df["timestamp"], history_df["confidence"], marker='o', linestyle='-', color='blue')
            ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold ({threshold})')
            ax.set_xlabel("Time")
            ax.set_ylabel("Confidence")
            ax.set_title("Hail Detection Confidence History")
            ax.legend()
            ax.grid(True)
            st.pyplot(fig)

with tabs[2]:
    # System Status tab
    st.markdown("<h2 class='sub-title'>System Status</h2>", unsafe_allow_html=True)
    
    # Create columns for status display
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("<h3>Protection System Status</h3>", unsafe_allow_html=True)
        
        # Protection status
        status_circle_class = "status-active" if st.session_state.protection_active else "status-inactive"
        status_text = "ACTIVE" if st.session_state.protection_active else "INACTIVE"
        
        st.markdown(
            f"<p><span class='status-circle {status_circle_class}'></span> Protection System: <strong>{status_text}</strong></p>",
            unsafe_allow_html=True
        )
        
        # Last detection time
        if st.session_state.last_detection_time:
            time_str = st.session_state.last_detection_time.strftime("%Y-%m-%d %H:%M:%S")
            st.markdown(f"Last Detection: {time_str}")
        else:
            st.markdown("Last Detection: Never")
        
        # ESP connection status
        if esp_ip and esp_port:
            st.markdown(f"ESP Module: Configured at {esp_ip}:{esp_port}")
            
            # Test connection
            if st.button("Test ESP Connection"):
                with st.spinner("Testing connection..."):
                    # Simulate a connection test
                    time.sleep(1)
                    if np.random.random() > 0.3:  # 70% success rate for demo
                        st.success("Connected to ESP module successfully!")
                    else:
                        st.error("Failed to get status from ESP module.")
        else:
            st.markdown("ESP Module: Not configured")
        
        st.markdown("</div>", unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='highlight'>", unsafe_allow_html=True)
        st.markdown("<h3>Model Information</h3>", unsafe_allow_html=True)
        
        # Model details
        st.markdown(f"Model Type: {model_type}")
        st.markdown(f"Detection Threshold: {threshold}")
        
        # Model characteristics based on type
        if model_type == "SteamCast":
            st.markdown("**Strengths**: Sensitive to spatial patterns and gradients")
            st.markdown("**Best for**: Detecting hail signatures in single frames")
        elif model_type == "DGMR":
            st.markdown("**Strengths**: Sensitive to temporal evolution")
            st.markdown("**Best for**: Tracking storm growth over time")
        elif model_type == "Ensemble":
            st.markdown("**Strengths**: Combines multiple models for reliability")
            st.markdown("**Best for**: Reducing false positives and false negatives")
        else:
            st.markdown("**Strengths**: Balanced performance")
            st.markdown("**Best for**: General-purpose hail detection")
        
        # System resources
        try:
            import psutil
            cpu_percent = psutil.cpu_percent()
            memory_info = psutil.virtual_memory()
            
            st.markdown(f"CPU Usage: {cpu_percent}%")
            st.markdown(f"Memory Usage: {memory_info.percent}%")
        except:
            st.markdown("System resource info not available")
        
        # Device info
        import platform
        st.markdown(f"System: {platform.system()} {platform.version()}")
        st.markdown(f"Python: {platform.python_version()}")
        st.markdown(f"Torch: {torch.__version__}")
        
        st.markdown("</div>", unsafe_allow_html=True)

with tabs[3]:
    show_roi_calculator()

# Footer
st.markdown("---")
st.markdown(
    "Hail Detection System - A project combining DGMR and SteamCast for reliable hail prediction"
)

if __name__ == "__main__":
    pass