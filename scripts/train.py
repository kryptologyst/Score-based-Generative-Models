"""Training script for score-based generative models."""

import argparse
import os
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.data import (
    ImageDataset,
    create_dataloader,
    get_transforms,
    get_device,
    set_seed,
    save_image_grid,
)
from src.models.score_network import ScoreNetwork, ScoreMatchingLoss, LangevinSampler
from src.utils.metrics import GenerativeMetrics, ModelLeaderboard


class ScoreBasedTrainer:
    """Trainer for score-based generative models."""
    
    def __init__(
        self,
        config: Dict,
        device: torch.device,
        save_dir: Path,
    ) -> None:
        """Initialize the trainer.
        
        Args:
            config: Training configuration
            device: Device to train on
            save_dir: Directory to save checkpoints and results
        """
        self.config = config
        self.device = device
        self.save_dir = save_dir
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize model
        self.model = ScoreNetwork(
            in_channels=config["model"]["in_channels"],
            base_channels=config["model"]["base_channels"],
            num_layers=config["model"]["num_layers"],
            time_embed_dim=config["model"]["time_embed_dim"],
            use_attention=config["model"]["use_attention"],
        ).to(device)
        
        # Initialize loss function
        self.loss_fn = ScoreMatchingLoss(
            sigma_min=config["training"]["sigma_min"],
            sigma_max=config["training"]["sigma_max"],
        )
        
        # Initialize optimizer
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=config["training"]["learning_rate"],
            betas=config["training"]["betas"],
            weight_decay=config["training"]["weight_decay"],
        )
        
        # Initialize scheduler
        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer,
            T_max=config["training"]["num_epochs"],
        )
        
        # Initialize metrics
        self.metrics = GenerativeMetrics(device)
        self.leaderboard = ModelLeaderboard(save_dir / "leaderboard.json")
        
        # Initialize sampler
        self.sampler = LangevinSampler(
            self.model,
            sigma_min=config["sampling"]["sigma_min"],
            sigma_max=config["sampling"]["sigma_max"],
            num_steps=config["sampling"]["num_steps"],
            step_size=config["sampling"]["step_size"],
            device=device,
        )
        
        # Training state
        self.current_epoch = 0
        self.best_fid = float("inf")
        
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch.
        
        Args:
            dataloader: Training data loader
            
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {self.current_epoch + 1}")
        
        for batch_idx, real_images in enumerate(pbar):
            real_images = real_images.to(self.device)
            
            # Compute loss
            loss, metrics = self.loss_fn(self.model, real_images, self.device)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            if self.config["training"]["grad_clip"] > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config["training"]["grad_clip"]
                )
            
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            num_batches += 1
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "sigma_mean": f"{metrics['sigma_mean']:.4f}",
            })
            
            # Log to wandb if available
            if self.config["logging"]["use_wandb"]:
                import wandb
                wandb.log({
                    "train/loss": loss.item(),
                    "train/sigma_mean": metrics["sigma_mean"],
                    "train/sigma_std": metrics["sigma_std"],
                    "train/score_norm": metrics["score_norm"],
                })
        
        return {
            "loss": total_loss / num_batches,
            "lr": self.optimizer.param_groups[0]["lr"],
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate the model.
        
        Args:
            dataloader: Evaluation data loader
            
        Returns:
            Dictionary of evaluation metrics
        """
        self.model.eval()
        
        # Generate samples
        with torch.no_grad():
            # Generate samples
            fake_images = self.sampler.sample(
                shape=(3, 32, 32),  # CIFAR-10 shape
                batch_size=len(dataloader.dataset),
            )
            
            # Get real images
            real_images = []
            for batch in dataloader:
                real_images.append(batch)
            real_images = torch.cat(real_images, dim=0)
            
            # Compute metrics
            metrics = self.metrics.evaluate_all(real_images, fake_images)
        
        return metrics
    
    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint.
        
        Args:
            is_best: Whether this is the best model so far
        """
        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "best_fid": self.best_fid,
            "config": self.config,
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / "checkpoint_latest.pth")
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / "checkpoint_best.pth")
    
    def load_checkpoint(self, checkpoint_path: Path) -> None:
        """Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        """
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        self.current_epoch = checkpoint["epoch"]
        self.best_fid = checkpoint["best_fid"]
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> None:
        """Train the model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
        """
        print(f"Starting training for {self.config['training']['num_epochs']} epochs...")
        
        for epoch in range(self.current_epoch, self.config["training"]["num_epochs"]):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch(train_loader)
            
            # Update scheduler
            self.scheduler.step()
            
            # Evaluate
            if (epoch + 1) % self.config["training"]["eval_interval"] == 0:
                val_metrics = self.evaluate(val_loader)
                
                # Update best FID
                is_best = val_metrics["fid"] < self.best_fid
                if is_best:
                    self.best_fid = val_metrics["fid"]
                
                # Save checkpoint
                self.save_checkpoint(is_best)
                
                # Add to leaderboard
                model_name = f"score_model_epoch_{epoch + 1}"
                self.leaderboard.add_result(model_name, val_metrics, self.config)
                
                # Print metrics
                print(f"\nEpoch {epoch + 1} Metrics:")
                print(f"  Train Loss: {train_metrics['loss']:.4f}")
                print(f"  FID: {val_metrics['fid']:.4f}")
                print(f"  IS: {val_metrics['is_mean']:.4f} ± {val_metrics['is_std']:.4f}")
                print(f"  LPIPS: {val_metrics['lpips']:.4f}")
                print(f"  Precision: {val_metrics['precision']:.4f}")
                print(f"  Recall: {val_metrics['recall']:.4f}")
                
                # Generate sample images
                if (epoch + 1) % self.config["training"]["sample_interval"] == 0:
                    self.generate_samples(epoch + 1)
            
            # Log to wandb if available
            if self.config["logging"]["use_wandb"]:
                import wandb
                wandb.log({
                    "epoch": epoch + 1,
                    "train/loss": train_metrics["loss"],
                    "train/lr": train_metrics["lr"],
                })
        
        print("Training completed!")
        self.leaderboard.print_leaderboard()
    
    def generate_samples(self, epoch: int) -> None:
        """Generate and save sample images.
        
        Args:
            epoch: Current epoch number
        """
        self.model.eval()
        
        with torch.no_grad():
            # Generate samples
            samples = self.sampler.sample(
                shape=(3, 32, 32),
                batch_size=64,
            )
            
            # Save grid
            save_path = self.save_dir / f"samples_epoch_{epoch}.png"
            save_image_grid(samples, save_path, nrow=8)


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train score-based generative model")
    parser.add_argument("--config", type=str, default="configs/train.yaml", help="Config file path")
    parser.add_argument("--data_dir", type=str, default="data/cifar10", help="Data directory")
    parser.add_argument("--save_dir", type=str, default="outputs", help="Output directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Load config
    import yaml
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Create data loaders
    train_transform = get_transforms(
        image_size=config["data"]["image_size"],
        normalize=True,
        augment=True,
    )
    
    val_transform = get_transforms(
        image_size=config["data"]["image_size"],
        normalize=True,
        augment=False,
    )
    
    train_dataset = ImageDataset(
        data_dir=args.data_dir,
        transform=train_transform,
        split="train",
        max_samples=config["data"]["max_train_samples"],
    )
    
    val_dataset = ImageDataset(
        data_dir=args.data_dir,
        transform=val_transform,
        split="val",
        max_samples=config["data"]["max_val_samples"],
    )
    
    train_loader = create_dataloader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
    )
    
    val_loader = create_dataloader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
    )
    
    # Create trainer
    save_dir = Path(args.save_dir)
    trainer = ScoreBasedTrainer(config, device, save_dir)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(Path(args.resume))
    
    # Initialize wandb if enabled
    if config["logging"]["use_wandb"]:
        import wandb
        wandb.init(
            project="score-based-generative-models",
            config=config,
            name=f"score_model_{args.seed}",
        )
    
    # Train
    trainer.train(train_loader, val_loader)
    
    # Close wandb
    if config["logging"]["use_wandb"]:
        wandb.finish()


if __name__ == "__main__":
    main()
