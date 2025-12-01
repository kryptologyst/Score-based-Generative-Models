"""Streamlit demo for score-based generative models."""

import streamlit as st
import torch
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import io
from PIL import Image

from src.data import get_device, set_seed, create_sample_grid
from src.models.score_network import ScoreNetwork, LangevinSampler


@st.cache_resource
def load_model(checkpoint_path: str):
    """Load model with caching."""
    device = get_device()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    config = checkpoint["config"]
    
    model = ScoreNetwork(
        in_channels=config["model"]["in_channels"],
        base_channels=config["model"]["base_channels"],
        num_layers=config["model"]["num_layers"],
        time_embed_dim=config["model"]["time_embed_dim"],
        use_attention=config["model"]["use_attention"],
    ).to(device)
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    
    return model, device


def generate_samples(
    model,
    device,
    num_samples,
    num_steps,
    step_size,
    sigma_min,
    sigma_max,
):
    """Generate samples."""
    sampler = LangevinSampler(
        model,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_steps=num_steps,
        step_size=step_size,
        device=device,
    )
    
    samples = sampler.sample(
        shape=(3, 32, 32),
        batch_size=num_samples,
    )
    
    return samples


def tensor_to_image(tensor):
    """Convert tensor to PIL Image."""
    # Denormalize
    tensor = (tensor + 1) / 2
    tensor = torch.clamp(tensor, 0, 1)
    
    # Convert to PIL
    if tensor.dim() == 4:
        # Batch of images
        images = []
        for i in range(tensor.shape[0]):
            img = tensor[i].permute(1, 2, 0).cpu().numpy()
            img = (img * 255).astype(np.uint8)
            images.append(Image.fromarray(img))
        return images
    else:
        # Single image
        img = tensor.permute(1, 2, 0).cpu().numpy()
        img = (img * 255).astype(np.uint8)
        return Image.fromarray(img)


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Score-based Generative Models",
        page_icon="🎨",
        layout="wide",
    )
    
    st.title("🎨 Score-based Generative Models")
    st.markdown("Generate images using score-based generative models with Langevin dynamics sampling.")
    
    # Sidebar for controls
    st.sidebar.header("Model Configuration")
    
    # Model selection
    checkpoint_path = st.sidebar.selectbox(
        "Select Model",
        options=["outputs/checkpoint_best.pth", "outputs/checkpoint_latest.pth"],
        help="Choose a trained model checkpoint"
    )
    
    if not Path(checkpoint_path).exists():
        st.error(f"Model checkpoint not found: {checkpoint_path}")
        st.info("Please train a model first using the training script.")
        return
    
    # Load model
    try:
        model, device = load_model(checkpoint_path)
        st.sidebar.success("Model loaded successfully!")
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return
    
    # Sampling parameters
    st.sidebar.header("Sampling Parameters")
    
    num_samples = st.sidebar.slider(
        "Number of Samples",
        min_value=1,
        max_value=64,
        value=16,
        help="Number of images to generate"
    )
    
    num_steps = st.sidebar.slider(
        "Langevin Steps",
        min_value=100,
        max_value=2000,
        value=1000,
        step=100,
        help="Number of Langevin dynamics steps"
    )
    
    step_size = st.sidebar.slider(
        "Step Size",
        min_value=0.00001,
        max_value=0.0001,
        value=0.00002,
        step=0.00001,
        format="%.5f",
        help="Step size for Langevin dynamics"
    )
    
    sigma_min = st.sidebar.slider(
        "Min Noise Level",
        min_value=0.001,
        max_value=0.1,
        value=0.01,
        step=0.001,
        format="%.3f",
        help="Minimum noise level"
    )
    
    sigma_max = st.sidebar.slider(
        "Max Noise Level",
        min_value=0.5,
        max_value=2.0,
        value=1.0,
        step=0.1,
        help="Maximum noise level"
    )
    
    # Random seed
    seed = st.sidebar.number_input(
        "Random Seed",
        min_value=0,
        max_value=1000000,
        value=42,
        help="Random seed for reproducibility"
    )
    
    # Generate button
    if st.sidebar.button("🎨 Generate Samples", type="primary"):
        with st.spinner("Generating samples..."):
            # Set seed
            set_seed(seed)
            
            # Generate samples
            samples = generate_samples(
                model,
                device,
                num_samples,
                num_steps,
                step_size,
                sigma_min,
                sigma_max,
            )
            
            # Convert to images
            images = tensor_to_image(samples)
            
            # Display samples
            st.header("Generated Samples")
            
            # Create grid layout
            cols = st.columns(4)
            for i, img in enumerate(images):
                with cols[i % 4]:
                    st.image(img, caption=f"Sample {i+1}")
            
            # Download button
            st.header("Download")
            
            # Create a grid image for download
            grid = create_sample_grid(samples, nrow=4)
            grid_img = tensor_to_image(grid)
            
            # Convert to bytes
            img_buffer = io.BytesIO()
            grid_img.save(img_buffer, format="PNG")
            img_buffer.seek(0)
            
            st.download_button(
                label="📥 Download Grid",
                data=img_buffer.getvalue(),
                file_name=f"samples_seed_{seed}.png",
                mime="image/png",
            )
    
    # Information section
    st.sidebar.header("About")
    st.sidebar.info("""
    This demo generates images using score-based generative models.
    
    **How it works:**
    1. The model learns the score (gradient) of the data distribution
    2. Samples are generated using Langevin dynamics
    3. The process starts from random noise and iteratively refines it
    
    **Parameters:**
    - **Langevin Steps**: More steps = better quality but slower
    - **Step Size**: Controls the size of each refinement step
    - **Noise Levels**: Control the noise schedule during sampling
    """)
    
    # Model info
    if st.sidebar.checkbox("Show Model Info"):
        st.sidebar.json({
            "Device": str(device),
            "Model Type": "Score Network",
            "Architecture": "U-Net with Attention",
            "Input Size": "32x32x3",
            "Parameters": f"{sum(p.numel() for p in model.parameters()):,}",
        })


if __name__ == "__main__":
    main()
