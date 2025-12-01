"""Score-based generative model implementation with proper score matching."""

from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class ScoreNetwork(nn.Module):
    """Score network that learns the gradient of the data distribution.
    
    This network implements a U-Net architecture suitable for score-based
    generative modeling on images.
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        base_channels: int = 64,
        num_layers: int = 4,
        time_embed_dim: int = 128,
        use_attention: bool = True,
    ) -> None:
        """Initialize the score network.
        
        Args:
            in_channels: Number of input channels (3 for RGB images)
            base_channels: Base number of channels in the network
            num_layers: Number of downsampling/upsampling layers
            time_embed_dim: Dimension of time embedding
            use_attention: Whether to use attention layers
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.base_channels = base_channels
        self.num_layers = num_layers
        self.time_embed_dim = time_embed_dim
        self.use_attention = use_attention
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Input projection
        self.input_proj = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        
        # Downsampling path
        self.down_layers = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        self.down_attns = nn.ModuleList()
        
        ch = base_channels
        for i in range(num_layers):
            self.down_layers.append(
                nn.Sequential(
                    nn.Conv2d(ch, ch, 3, padding=1),
                    nn.GroupNorm(8, ch),
                    nn.SiLU(),
                    nn.Conv2d(ch, ch, 3, padding=1),
                    nn.GroupNorm(8, ch),
                    nn.SiLU(),
                )
            )
            
            if i < num_layers - 1:
                self.down_samples.append(nn.Conv2d(ch, ch * 2, 2, stride=2))
                ch *= 2
            else:
                self.down_samples.append(nn.Identity())
                
            if use_attention and i >= num_layers // 2:
                self.down_attns.append(
                    nn.MultiheadAttention(ch, num_heads=8, batch_first=True)
                )
            else:
                self.down_attns.append(nn.Identity())
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
            nn.Conv2d(ch, ch, 3, padding=1),
            nn.GroupNorm(8, ch),
            nn.SiLU(),
        )
        
        # Upsampling path
        self.up_layers = nn.ModuleList()
        self.up_samples = nn.ModuleList()
        self.up_attns = nn.ModuleList()
        
        for i in range(num_layers):
            if i == 0:
                self.up_layers.append(
                    nn.Sequential(
                        nn.Conv2d(ch, ch, 3, padding=1),
                        nn.GroupNorm(8, ch),
                        nn.SiLU(),
                        nn.Conv2d(ch, ch, 3, padding=1),
                        nn.GroupNorm(8, ch),
                        nn.SiLU(),
                    )
                )
            else:
                self.up_layers.append(
                    nn.Sequential(
                        nn.Conv2d(ch * 2, ch, 3, padding=1),
                        nn.GroupNorm(8, ch),
                        nn.SiLU(),
                        nn.Conv2d(ch, ch, 3, padding=1),
                        nn.GroupNorm(8, ch),
                        nn.SiLU(),
                    )
                )
            
            if i < num_layers - 1:
                self.up_samples.append(nn.ConvTranspose2d(ch, ch // 2, 2, stride=2))
                ch //= 2
            else:
                self.up_samples.append(nn.Identity())
                
            if use_attention and i >= num_layers // 2:
                self.up_attns.append(
                    nn.MultiheadAttention(ch, num_heads=8, batch_first=True)
                )
            else:
                self.up_attns.append(nn.Identity())
        
        # Output projection
        self.output_proj = nn.Conv2d(base_channels, in_channels, 3, padding=1)
        
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Forward pass of the score network.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            t: Time tensor of shape (B, 1)
            
        Returns:
            Score tensor of same shape as input
        """
        B, C, H, W = x.shape
        
        # Time embedding
        t_emb = self.time_mlp(t)  # (B, time_embed_dim)
        
        # Input projection
        h = self.input_proj(x)
        
        # Downsampling path
        skip_connections = []
        ch = self.base_channels
        
        for i in range(self.num_layers):
            # Residual connection
            residual = h
            h = self.down_layers[i](h)
            h = h + residual
            
            # Attention (if enabled)
            if self.use_attention and i >= self.num_layers // 2:
                B, C_h, H_h, W_h = h.shape
                h_flat = h.view(B, C_h, H_h * W_h).transpose(1, 2)  # (B, H*W, C)
                h_attn, _ = self.down_attns[i](h_flat, h_flat, h_flat)
                h = h_attn.transpose(1, 2).view(B, C_h, H_h, W_h)
            
            skip_connections.append(h)
            
            # Downsample
            h = self.down_samples[i](h)
            if i < self.num_layers - 1:
                ch *= 2
        
        # Bottleneck
        residual = h
        h = self.bottleneck(h)
        h = h + residual
        
        # Upsampling path
        for i in range(self.num_layers):
            # Upsample
            h = self.up_samples[i](h)
            if i < self.num_layers - 1:
                ch //= 2
            
            # Skip connection
            skip = skip_connections[self.num_layers - 1 - i]
            h = torch.cat([h, skip], dim=1)
            
            # Residual connection
            residual = h
            h = self.up_layers[i](h)
            h = h + residual
            
            # Attention (if enabled)
            if self.use_attention and i >= self.num_layers // 2:
                B, C_h, H_h, W_h = h.shape
                h_flat = h.view(B, C_h, H_h * W_h).transpose(1, 2)  # (B, H*W, C)
                h_attn, _ = self.up_attns[i](h_flat, h_flat, h_flat)
                h = h_attn.transpose(1, 2).view(B, C_h, H_h, W_h)
        
        # Output projection
        score = self.output_proj(h)
        
        return score


class ScoreMatchingLoss(nn.Module):
    """Score matching loss for training score-based generative models."""
    
    def __init__(self, sigma_min: float = 0.01, sigma_max: float = 1.0) -> None:
        """Initialize the score matching loss.
        
        Args:
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
        """
        super().__init__()
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        
    def forward(
        self, 
        score_net: nn.Module, 
        x: Tensor, 
        device: torch.device
    ) -> Tuple[Tensor, Dict[str, float]]:
        """Compute the score matching loss.
        
        Args:
            score_net: The score network
            x: Real data samples of shape (B, C, H, W)
            device: Device to run computation on
            
        Returns:
            Loss tensor and metrics dictionary
        """
        B = x.shape[0]
        
        # Sample random noise levels
        sigma = torch.rand(B, 1, device=device) * (self.sigma_max - self.sigma_min) + self.sigma_min
        
        # Add noise to data
        noise = torch.randn_like(x)
        x_noisy = x + sigma.view(-1, 1, 1, 1) * noise
        
        # Predict score
        t = sigma  # Use sigma as time
        predicted_score = score_net(x_noisy, t)
        
        # True score is -noise / sigma
        true_score = -noise / sigma.view(-1, 1, 1, 1)
        
        # Score matching loss
        loss = F.mse_loss(predicted_score, true_score)
        
        # Additional metrics
        metrics = {
            "loss": loss.item(),
            "sigma_mean": sigma.mean().item(),
            "sigma_std": sigma.std().item(),
            "score_norm": predicted_score.norm(dim=(1, 2, 3)).mean().item(),
        }
        
        return loss, metrics


class LangevinSampler:
    """Langevin dynamics sampler for score-based generative models."""
    
    def __init__(
        self,
        score_net: nn.Module,
        sigma_min: float = 0.01,
        sigma_max: float = 1.0,
        num_steps: int = 1000,
        step_size: float = 0.00002,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """Initialize the Langevin sampler.
        
        Args:
            score_net: Trained score network
            sigma_min: Minimum noise level
            sigma_max: Maximum noise level
            num_steps: Number of Langevin steps
            step_size: Step size for Langevin dynamics
            device: Device to run sampling on
        """
        self.score_net = score_net
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
        self.num_steps = num_steps
        self.step_size = step_size
        self.device = device
        
    def sample(
        self, 
        shape: Tuple[int, ...], 
        batch_size: int = 1,
        return_trajectory: bool = False
    ) -> Union[Tensor, List[Tensor]]:
        """Sample from the learned distribution using Langevin dynamics.
        
        Args:
            shape: Shape of samples to generate (excluding batch dimension)
            batch_size: Number of samples to generate
            return_trajectory: Whether to return the sampling trajectory
            
        Returns:
            Generated samples or list of trajectory steps
        """
        self.score_net.eval()
        
        # Initialize with random noise
        x = torch.randn(batch_size, *shape, device=self.device)
        
        if return_trajectory:
            trajectory = [x.clone()]
        
        # Annealed Langevin dynamics
        with torch.no_grad():
            for i in range(self.num_steps):
                # Current noise level (annealing schedule)
                sigma = self.sigma_max * (self.sigma_min / self.sigma_max) ** (i / self.num_steps)
                t = torch.full((batch_size, 1), sigma, device=self.device)
                
                # Get score
                score = self.score_net(x, t)
                
                # Langevin step
                noise = torch.randn_like(x)
                x = x + self.step_size * score + np.sqrt(2 * self.step_size) * noise
                
                if return_trajectory and i % (self.num_steps // 10) == 0:
                    trajectory.append(x.clone())
        
        if return_trajectory:
            return trajectory
        else:
            return x
