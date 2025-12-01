"""Sampling script for score-based generative models."""

import argparse
import os
from pathlib import Path
from typing import Optional

import torch
import matplotlib.pyplot as plt
import numpy as np

from src.data import get_device, set_seed, save_image_grid, create_sample_grid
from src.models.score_network import ScoreNetwork, LangevinSampler


def load_model(checkpoint_path: Path, device: torch.device) -> ScoreNetwork:
    """Load a trained model from checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        Loaded model
    """
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
    
    return model


def generate_samples(
    model: ScoreNetwork,
    device: torch.device,
    num_samples: int = 64,
    image_size: int = 32,
    num_steps: int = 1000,
    step_size: float = 0.00002,
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
) -> torch.Tensor:
    """Generate samples using Langevin dynamics.
    
    Args:
        model: Trained score network
        device: Device to run on
        num_samples: Number of samples to generate
        image_size: Size of generated images
        num_steps: Number of Langevin steps
        step_size: Step size for Langevin dynamics
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        
    Returns:
        Generated samples tensor
    """
    sampler = LangevinSampler(
        model,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_steps=num_steps,
        step_size=step_size,
        device=device,
    )
    
    samples = sampler.sample(
        shape=(3, image_size, image_size),
        batch_size=num_samples,
    )
    
    return samples


def generate_trajectory(
    model: ScoreNetwork,
    device: torch.device,
    num_samples: int = 4,
    image_size: int = 32,
    num_steps: int = 1000,
    step_size: float = 0.00002,
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
) -> list:
    """Generate sampling trajectory.
    
    Args:
        model: Trained score network
        device: Device to run on
        num_samples: Number of samples to generate
        image_size: Size of generated images
        num_steps: Number of Langevin steps
        step_size: Step size for Langevin dynamics
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        
    Returns:
        List of trajectory steps
    """
    sampler = LangevinSampler(
        model,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_steps=num_steps,
        step_size=step_size,
        device=device,
    )
    
    trajectory = sampler.sample(
        shape=(3, image_size, image_size),
        batch_size=num_samples,
        return_trajectory=True,
    )
    
    return trajectory


def visualize_trajectory(
    trajectory: list,
    save_path: Path,
    nrow: int = 4,
) -> None:
    """Visualize sampling trajectory.
    
    Args:
        trajectory: List of trajectory steps
        save_path: Path to save visualization
        nrow: Number of images per row
    """
    fig, axes = plt.subplots(
        len(trajectory),
        nrow,
        figsize=(nrow * 2, len(trajectory) * 2),
    )
    
    if len(trajectory) == 1:
        axes = axes.reshape(1, -1)
    
    for i, step in enumerate(trajectory):
        for j in range(nrow):
            if j < step.shape[0]:
                img = step[j].cpu().permute(1, 2, 0)
                img = (img + 1) / 2  # Denormalize
                img = torch.clamp(img, 0, 1)
                
                axes[i, j].imshow(img)
                axes[i, j].set_title(f"Step {i * len(trajectory) + j}")
                axes[i, j].axis("off")
            else:
                axes[i, j].axis("off")
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def interpolate_samples(
    model: ScoreNetwork,
    device: torch.device,
    num_interpolations: int = 8,
    image_size: int = 32,
    num_steps: int = 1000,
    step_size: float = 0.00002,
    sigma_min: float = 0.01,
    sigma_max: float = 1.0,
) -> torch.Tensor:
    """Generate interpolated samples.
    
    Args:
        model: Trained score network
        device: Device to run on
        num_interpolations: Number of interpolation steps
        image_size: Size of generated images
        num_steps: Number of Langevin steps
        step_size: Step size for Langevin dynamics
        sigma_min: Minimum noise level
        sigma_max: Maximum noise level
        
    Returns:
        Interpolated samples tensor
    """
    # Generate two random samples
    sampler = LangevinSampler(
        model,
        sigma_min=sigma_min,
        sigma_max=sigma_max,
        num_steps=num_steps,
        step_size=step_size,
        device=device,
    )
    
    sample1 = sampler.sample(
        shape=(3, image_size, image_size),
        batch_size=1,
    )
    
    sample2 = sampler.sample(
        shape=(3, image_size, image_size),
        batch_size=1,
    )
    
    # Interpolate in noise space
    interpolations = []
    for i in range(num_interpolations):
        alpha = i / (num_interpolations - 1)
        
        # Start with interpolated noise
        noise = (1 - alpha) * torch.randn_like(sample1) + alpha * torch.randn_like(sample2)
        
        # Run Langevin dynamics from interpolated noise
        x = noise
        with torch.no_grad():
            for step in range(num_steps):
                sigma = sigma_max * (sigma_min / sigma_max) ** (step / num_steps)
                t = torch.full((1, 1), sigma, device=device)
                
                score = model(x, t)
                noise_step = torch.randn_like(x)
                x = x + step_size * score + np.sqrt(2 * step_size) * noise_step
        
        interpolations.append(x)
    
    return torch.cat(interpolations, dim=0)


def main():
    """Main sampling function."""
    parser = argparse.ArgumentParser(description="Sample from score-based generative model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="samples", help="Output directory")
    parser.add_argument("--num_samples", type=int, default=64, help="Number of samples to generate")
    parser.add_argument("--num_steps", type=int, default=1000, help="Number of Langevin steps")
    parser.add_argument("--step_size", type=float, default=0.00002, help="Langevin step size")
    parser.add_argument("--sigma_min", type=float, default=0.01, help="Minimum noise level")
    parser.add_argument("--sigma_max", type=float, default=1.0, help="Maximum noise level")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--trajectory", action="store_true", help="Generate trajectory visualization")
    parser.add_argument("--interpolation", action="store_true", help="Generate interpolation samples")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("Loading model...")
    model = load_model(Path(args.checkpoint), device)
    print("Model loaded successfully!")
    
    # Generate samples
    print(f"Generating {args.num_samples} samples...")
    samples = generate_samples(
        model,
        device,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        step_size=args.step_size,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )
    
    # Save samples
    save_path = output_dir / f"samples_seed_{args.seed}.png"
    save_image_grid(samples, save_path, nrow=8)
    print(f"Samples saved to {save_path}")
    
    # Generate trajectory if requested
    if args.trajectory:
        print("Generating trajectory visualization...")
        trajectory = generate_trajectory(
            model,
            device,
            num_samples=4,
            num_steps=args.num_steps,
            step_size=args.step_size,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        )
        
        trajectory_path = output_dir / f"trajectory_seed_{args.seed}.png"
        visualize_trajectory(trajectory, trajectory_path)
        print(f"Trajectory saved to {trajectory_path}")
    
    # Generate interpolation if requested
    if args.interpolation:
        print("Generating interpolation samples...")
        interpolations = interpolate_samples(
            model,
            device,
            num_interpolations=8,
            num_steps=args.num_steps,
            step_size=args.step_size,
            sigma_min=args.sigma_min,
            sigma_max=args.sigma_max,
        )
        
        interpolation_path = output_dir / f"interpolation_seed_{args.seed}.png"
        save_image_grid(interpolations, interpolation_path, nrow=8)
        print(f"Interpolation saved to {interpolation_path}")
    
    print("Sampling completed!")


if __name__ == "__main__":
    main()
