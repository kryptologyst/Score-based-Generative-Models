"""Unit tests for score-based generative models."""

import pytest
import torch
import numpy as np
from pathlib import Path

from src.models.score_network import ScoreNetwork, ScoreMatchingLoss, LangevinSampler
from src.data import get_device, set_seed
from src.utils.metrics import GenerativeMetrics


class TestScoreNetwork:
    """Test the ScoreNetwork class."""
    
    def test_init(self):
        """Test network initialization."""
        model = ScoreNetwork(
            in_channels=3,
            base_channels=64,
            num_layers=4,
            time_embed_dim=128,
            use_attention=True
        )
        
        assert model.in_channels == 3
        assert model.base_channels == 64
        assert model.num_layers == 4
        assert model.time_embed_dim == 128
        assert model.use_attention == True
    
    def test_forward(self):
        """Test forward pass."""
        model = ScoreNetwork(in_channels=3, base_channels=32, num_layers=2)
        
        # Test input
        x = torch.randn(2, 3, 32, 32)
        t = torch.randn(2, 1)
        
        # Forward pass
        output = model(x, t)
        
        # Check output shape
        assert output.shape == x.shape
        assert output.dtype == x.dtype
    
    def test_gradient_flow(self):
        """Test gradient flow."""
        model = ScoreNetwork(in_channels=3, base_channels=32, num_layers=2)
        
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        t = torch.randn(2, 1)
        
        output = model(x, t)
        loss = output.sum()
        loss.backward()
        
        # Check gradients exist
        assert x.grad is not None
        assert x.grad.requires_grad == False


class TestScoreMatchingLoss:
    """Test the ScoreMatchingLoss class."""
    
    def test_init(self):
        """Test loss initialization."""
        loss_fn = ScoreMatchingLoss(sigma_min=0.01, sigma_max=1.0)
        
        assert loss_fn.sigma_min == 0.01
        assert loss_fn.sigma_max == 1.0
    
    def test_forward(self):
        """Test loss computation."""
        model = ScoreNetwork(in_channels=3, base_channels=32, num_layers=2)
        loss_fn = ScoreMatchingLoss(sigma_min=0.01, sigma_max=1.0)
        
        device = get_device()
        model = model.to(device)
        
        # Test data
        x = torch.randn(4, 3, 32, 32).to(device)
        
        # Compute loss
        loss, metrics = loss_fn(model, x, device)
        
        # Check loss is scalar
        assert loss.dim() == 0
        assert loss.item() >= 0
        
        # Check metrics
        assert "loss" in metrics
        assert "sigma_mean" in metrics
        assert "sigma_std" in metrics
        assert "score_norm" in metrics


class TestLangevinSampler:
    """Test the LangevinSampler class."""
    
    def test_init(self):
        """Test sampler initialization."""
        model = ScoreNetwork(in_channels=3, base_channels=32, num_layers=2)
        device = get_device()
        
        sampler = LangevinSampler(
            model,
            sigma_min=0.01,
            sigma_max=1.0,
            num_steps=100,
            step_size=0.00002,
            device=device
        )
        
        assert sampler.sigma_min == 0.01
        assert sampler.sigma_max == 1.0
        assert sampler.num_steps == 100
        assert sampler.step_size == 0.00002
    
    def test_sample(self):
        """Test sampling."""
        model = ScoreNetwork(in_channels=3, base_channels=32, num_layers=2)
        device = get_device()
        model = model.to(device)
        
        sampler = LangevinSampler(
            model,
            sigma_min=0.01,
            sigma_max=1.0,
            num_steps=10,  # Small number for testing
            step_size=0.00002,
            device=device
        )
        
        # Generate samples
        samples = sampler.sample(
            shape=(3, 32, 32),
            batch_size=4
        )
        
        # Check output shape
        assert samples.shape == (4, 3, 32, 32)
        assert samples.device == device
    
    def test_trajectory(self):
        """Test trajectory generation."""
        model = ScoreNetwork(in_channels=3, base_channels=32, num_layers=2)
        device = get_device()
        model = model.to(device)
        
        sampler = LangevinSampler(
            model,
            sigma_min=0.01,
            sigma_max=1.0,
            num_steps=10,
            step_size=0.00002,
            device=device
        )
        
        # Generate trajectory
        trajectory = sampler.sample(
            shape=(3, 32, 32),
            batch_size=2,
            return_trajectory=True
        )
        
        # Check trajectory
        assert isinstance(trajectory, list)
        assert len(trajectory) > 1
        
        for step in trajectory:
            assert step.shape == (2, 3, 32, 32)
            assert step.device == device


class TestGenerativeMetrics:
    """Test the GenerativeMetrics class."""
    
    def test_init(self):
        """Test metrics initialization."""
        device = get_device()
        metrics = GenerativeMetrics(device, num_samples=100, batch_size=16)
        
        assert metrics.device == device
        assert metrics.num_samples == 100
        assert metrics.batch_size == 16
    
    def test_evaluate_all(self):
        """Test comprehensive evaluation."""
        device = get_device()
        metrics = GenerativeMetrics(device, num_samples=10, batch_size=4)
        
        # Generate test data
        real_images = torch.randn(10, 3, 32, 32).to(device)
        fake_images = torch.randn(10, 3, 32, 32).to(device)
        
        # Compute metrics
        results = metrics.evaluate_all(real_images, fake_images)
        
        # Check all metrics are present
        expected_metrics = ["fid", "is_mean", "is_std", "lpips", "precision", "recall"]
        for metric in expected_metrics:
            assert metric in results
            assert isinstance(results[metric], (int, float))


class TestDataUtils:
    """Test data utilities."""
    
    def test_set_seed(self):
        """Test seed setting."""
        set_seed(42)
        
        # Generate random numbers
        torch_rand = torch.rand(1)
        np_rand = np.random.rand(1)
        
        # Set seed again
        set_seed(42)
        
        # Generate again
        torch_rand2 = torch.rand(1)
        np_rand2 = np.random.rand(1)
        
        # Should be the same
        assert torch.allclose(torch_rand, torch_rand2)
        assert np.allclose(np_rand, np_rand2)
    
    def test_get_device(self):
        """Test device detection."""
        device = get_device()
        
        assert isinstance(device, torch.device)
        assert device.type in ["cpu", "cuda", "mps"]


if __name__ == "__main__":
    pytest.main([__file__])
