"""Evaluation metrics for score-based generative models."""

import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.image import FrechetInceptionDistance, InceptionScore
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


class GenerativeMetrics:
    """Comprehensive evaluation metrics for generative models."""
    
    def __init__(
        self,
        device: torch.device,
        num_samples: int = 10000,
        batch_size: int = 64,
    ) -> None:
        """Initialize the metrics evaluator.
        
        Args:
            device: Device to run evaluation on
            num_samples: Number of samples to use for evaluation
            batch_size: Batch size for evaluation
        """
        self.device = device
        self.num_samples = num_samples
        self.batch_size = batch_size
        
        # Initialize metrics
        self.fid = FrechetInceptionDistance(normalize=True).to(device)
        self.is_score = InceptionScore(normalize=True).to(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type="alex", normalize=True
        ).to(device)
        
    def compute_fid(
        self,
        real_images: Tensor,
        fake_images: Tensor,
    ) -> float:
        """Compute Fréchet Inception Distance (FID).
        
        Args:
            real_images: Real images tensor (B, C, H, W)
            fake_images: Generated images tensor (B, C, H, W)
            
        Returns:
            FID score
        """
        self.fid.reset()
        
        # Process in batches
        for i in range(0, len(real_images), self.batch_size):
            real_batch = real_images[i:i + self.batch_size].to(self.device)
            fake_batch = fake_images[i:i + self.batch_size].to(self.device)
            
            self.fid.update(real_batch, real=True)
            self.fid.update(fake_batch, real=False)
        
        return self.fid.compute().item()
    
    def compute_is(
        self,
        fake_images: Tensor,
    ) -> Tuple[float, float]:
        """Compute Inception Score (IS).
        
        Args:
            fake_images: Generated images tensor (B, C, H, W)
            
        Returns:
            Tuple of (IS mean, IS std)
        """
        self.is_score.reset()
        
        # Process in batches
        for i in range(0, len(fake_images), self.batch_size):
            fake_batch = fake_images[i:i + self.batch_size].to(self.device)
            self.is_score.update(fake_batch)
        
        is_mean, is_std = self.is_score.compute()
        return is_mean.item(), is_std.item()
    
    def compute_lpips(
        self,
        real_images: Tensor,
        fake_images: Tensor,
    ) -> float:
        """Compute LPIPS (Learned Perceptual Image Patch Similarity).
        
        Args:
            real_images: Real images tensor (B, C, H, W)
            fake_images: Generated images tensor (B, C, H, W)
            
        Returns:
            LPIPS score
        """
        self.lpips.reset()
        
        # Ensure same number of images
        min_len = min(len(real_images), len(fake_images))
        real_images = real_images[:min_len]
        fake_images = fake_images[:min_len]
        
        # Process in batches
        lpips_scores = []
        for i in range(0, min_len, self.batch_size):
            real_batch = real_images[i:i + self.batch_size].to(self.device)
            fake_batch = fake_images[i:i + self.batch_size].to(self.device)
            
            batch_lpips = self.lpips(real_batch, fake_batch)
            lpips_scores.append(batch_lpips)
        
        return torch.cat(lpips_scores).mean().item()
    
    def compute_precision_recall(
        self,
        real_images: Tensor,
        fake_images: Tensor,
        k: int = 3,
    ) -> Tuple[float, float]:
        """Compute Precision and Recall metrics.
        
        Args:
            real_images: Real images tensor (B, C, H, W)
            fake_images: Generated images tensor (B, C, H, W)
            k: Number of nearest neighbors for precision/recall
            
        Returns:
            Tuple of (precision, recall)
        """
        # Extract features using a simple CNN
        features_real = self._extract_features(real_images)
        features_fake = self._extract_features(fake_images)
        
        # Compute precision
        precision = self._compute_precision(features_fake, features_real, k)
        
        # Compute recall
        recall = self._compute_precision(features_real, features_fake, k)
        
        return precision, recall
    
    def _extract_features(self, images: Tensor) -> Tensor:
        """Extract features from images using a simple CNN.
        
        Args:
            images: Input images tensor (B, C, H, W)
            
        Returns:
            Feature tensor (B, feature_dim)
        """
        # Simple feature extractor
        features = F.adaptive_avg_pool2d(images, (1, 1))
        features = features.view(features.size(0), -1)
        return features
    
    def _compute_precision(
        self,
        features1: Tensor,
        features2: Tensor,
        k: int,
    ) -> float:
        """Compute precision between two sets of features.
        
        Args:
            features1: First set of features
            features2: Second set of features
            k: Number of nearest neighbors
            
        Returns:
            Precision score
        """
        # Compute pairwise distances
        distances = torch.cdist(features1, features2)
        
        # Find k nearest neighbors
        _, indices = torch.topk(distances, k, dim=1, largest=False)
        
        # Compute precision
        precision = 0.0
        for i in range(len(features1)):
            # Count how many of the k nearest neighbors are from the same set
            neighbor_indices = indices[i]
            precision += (neighbor_indices < len(features1)).float().mean()
        
        return precision / len(features1)
    
    def evaluate_all(
        self,
        real_images: Tensor,
        fake_images: Tensor,
    ) -> Dict[str, float]:
        """Compute all evaluation metrics.
        
        Args:
            real_images: Real images tensor (B, C, H, W)
            fake_images: Generated images tensor (B, C, H, W)
            
        Returns:
            Dictionary of metric scores
        """
        metrics = {}
        
        # FID
        metrics["fid"] = self.compute_fid(real_images, fake_images)
        
        # Inception Score
        is_mean, is_std = self.compute_is(fake_images)
        metrics["is_mean"] = is_mean
        metrics["is_std"] = is_std
        
        # LPIPS
        metrics["lpips"] = self.compute_lpips(real_images, fake_images)
        
        # Precision and Recall
        precision, recall = self.compute_precision_recall(real_images, fake_images)
        metrics["precision"] = precision
        metrics["recall"] = recall
        
        return metrics


class ModelLeaderboard:
    """Leaderboard for tracking model performance."""
    
    def __init__(self, save_path: Union[str, Path]) -> None:
        """Initialize the leaderboard.
        
        Args:
            save_path: Path to save the leaderboard
        """
        self.save_path = Path(save_path)
        self.results = []
        
        # Create directory if it doesn't exist
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Load existing results
        if self.save_path.exists():
            self.load_results()
    
    def add_result(
        self,
        model_name: str,
        metrics: Dict[str, float],
        config: Optional[Dict] = None,
    ) -> None:
        """Add a new result to the leaderboard.
        
        Args:
            model_name: Name of the model
            metrics: Dictionary of metric scores
            config: Model configuration (optional)
        """
        result = {
            "model_name": model_name,
            "metrics": metrics,
            "config": config,
            "timestamp": torch.datetime.now().isoformat(),
        }
        
        self.results.append(result)
        self.save_results()
    
    def get_best_model(self, metric: str = "fid") -> Optional[Dict]:
        """Get the best model for a given metric.
        
        Args:
            metric: Metric to optimize (lower is better for FID, higher for others)
            
        Returns:
            Best model result or None
        """
        if not self.results:
            return None
        
        # Determine if lower is better
        lower_is_better = metric in ["fid", "lpips"]
        
        if lower_is_better:
            best_result = min(self.results, key=lambda x: x["metrics"].get(metric, float("inf")))
        else:
            best_result = max(self.results, key=lambda x: x["metrics"].get(metric, float("-inf")))
        
        return best_result
    
    def get_leaderboard(self, metric: str = "fid") -> List[Dict]:
        """Get the leaderboard sorted by a metric.
        
        Args:
            metric: Metric to sort by
            
        Returns:
            List of results sorted by the metric
        """
        if not self.results:
            return []
        
        # Determine if lower is better
        lower_is_better = metric in ["fid", "lpips"]
        
        if lower_is_better:
            sorted_results = sorted(
                self.results,
                key=lambda x: x["metrics"].get(metric, float("inf"))
            )
        else:
            sorted_results = sorted(
                self.results,
                key=lambda x: x["metrics"].get(metric, float("-inf")),
                reverse=True
            )
        
        return sorted_results
    
    def save_results(self) -> None:
        """Save results to file."""
        import json
        
        with open(self.save_path, "w") as f:
            json.dump(self.results, f, indent=2)
    
    def load_results(self) -> None:
        """Load results from file."""
        import json
        
        with open(self.save_path, "r") as f:
            self.results = json.load(f)
    
    def print_leaderboard(self, metric: str = "fid") -> None:
        """Print the leaderboard to console.
        
        Args:
            metric: Metric to display
        """
        leaderboard = self.get_leaderboard(metric)
        
        if not leaderboard:
            print("No results available.")
            return
        
        print(f"\nLeaderboard (sorted by {metric}):")
        print("-" * 80)
        print(f"{'Rank':<4} {'Model':<20} {metric.upper():<10} {'FID':<8} {'IS':<8} {'LPIPS':<8}")
        print("-" * 80)
        
        for i, result in enumerate(leaderboard[:10]):  # Top 10
            metrics = result["metrics"]
            print(
                f"{i+1:<4} {result['model_name']:<20} "
                f"{metrics.get(metric, 'N/A'):<10.4f} "
                f"{metrics.get('fid', 'N/A'):<8.4f} "
                f"{metrics.get('is_mean', 'N/A'):<8.4f} "
                f"{metrics.get('lpips', 'N/A'):<8.4f}"
            )
