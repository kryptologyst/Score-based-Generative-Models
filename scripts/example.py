#!/usr/bin/env python3
"""Simple example demonstrating score-based generative models."""

import torch
import matplotlib.pyplot as plt
from pathlib import Path

from src.models.score_network import ScoreNetwork, ScoreMatchingLoss, LangevinSampler
from src.data import get_device, set_seed, save_image_grid


def main():
    """Run a simple example."""
    print("🎨 Score-based Generative Models Example")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create model
    print("Creating score network...")
    model = ScoreNetwork(
        in_channels=3,
        base_channels=32,  # Smaller for demo
        num_layers=3,
        time_embed_dim=64,
        use_attention=False,  # Disable for faster demo
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create loss function
    loss_fn = ScoreMatchingLoss(sigma_min=0.01, sigma_max=1.0)
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Generate some synthetic data (2D Gaussians)
    print("Generating synthetic data...")
    batch_size = 64
    image_size = 32
    
    # Create 2D Gaussian data
    def generate_gaussian_data(n_samples):
        """Generate 2D Gaussian data."""
        # Create 2D coordinates
        x = torch.linspace(-2, 2, image_size)
        y = torch.linspace(-2, 2, image_size)
        X, Y = torch.meshgrid(x, y, indexing='ij')
        
        # Create Gaussian
        sigma = 0.5
        gaussian = torch.exp(-(X**2 + Y**2) / (2 * sigma**2))
        
        # Repeat for RGB channels
        data = gaussian.unsqueeze(0).repeat(3, 1, 1)
        
        # Add some noise
        noise = torch.randn_like(data) * 0.1
        
        return data + noise
    
    # Generate training data
    train_data = []
    for _ in range(batch_size):
        sample = generate_gaussian_data(1)
        train_data.append(sample)
    
    train_data = torch.stack(train_data).to(device)
    print(f"Training data shape: {train_data.shape}")
    
    # Training loop
    print("Training model...")
    num_epochs = 50
    
    for epoch in range(num_epochs):
        # Compute loss
        loss, metrics = loss_fn(model, train_data, device)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Print progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {loss.item():.4f}")
    
    # Generate samples
    print("Generating samples...")
    sampler = LangevinSampler(
        model,
        sigma_min=0.01,
        sigma_max=1.0,
        num_steps=500,  # Fewer steps for demo
        step_size=0.00002,
        device=device,
    )
    
    samples = sampler.sample(
        shape=(3, image_size, image_size),
        batch_size=16,
    )
    
    # Save samples
    output_dir = Path("assets")
    output_dir.mkdir(exist_ok=True)
    
    save_path = output_dir / "example_samples.png"
    save_image_grid(samples, save_path, nrow=4)
    
    print(f"Samples saved to {save_path}")
    
    # Display some statistics
    print("\nSample Statistics:")
    print(f"Mean: {samples.mean().item():.4f}")
    print(f"Std: {samples.std().item():.4f}")
    print(f"Min: {samples.min().item():.4f}")
    print(f"Max: {samples.max().item():.4f}")
    
    print("\n✅ Example completed successfully!")
    print("Check the 'assets' directory for generated samples.")


if __name__ == "__main__":
    main()
