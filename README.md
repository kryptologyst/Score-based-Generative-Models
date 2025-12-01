# Score-based Generative Models

A production-ready implementation of score-based generative models with proper score matching and Langevin dynamics sampling.

## Overview

Score-based generative models learn the gradient (score) of the data distribution and generate samples by iteratively refining random noise using Langevin dynamics. This approach offers several advantages:

- **Direct sampling**: No adversarial training required
- **Flexible architecture**: Can use any neural network architecture
- **Theoretical foundation**: Based on score matching theory
- **High quality**: Can generate high-quality samples with proper training

## Features

- **Modern Architecture**: U-Net with attention mechanisms
- **Proper Score Matching**: Implements noise perturbation and score matching loss
- **Langevin Sampling**: Annealed Langevin dynamics for high-quality generation
- **Comprehensive Evaluation**: FID, IS, LPIPS, Precision/Recall metrics
- **Interactive Demo**: Streamlit web interface for sampling
- **Production Ready**: Type hints, documentation, testing, CI/CD

## Installation

### Prerequisites

- Python 3.10+
- PyTorch 2.0+
- CUDA (optional, for GPU acceleration)

### Install Dependencies

```bash
# Clone the repository
git clone https://github.com/kryptologyst/Score-based-Generative-Models.git
cd Score-based-Generative-Models

# Install dependencies
pip install -r requirements.txt

# For development
pip install -e ".[dev]"
```

### Optional GPU Acceleration

```bash
# Install xformers for memory-efficient attention (Linux/Windows only)
pip install xformers
```

## Quick Start

### 1. Train a Model

```bash
# Train on CIFAR-10
python scripts/train.py \
    --config configs/train.yaml \
    --data_dir data/cifar10 \
    --save_dir outputs \
    --seed 42
```

### 2. Generate Samples

```bash
# Generate samples from trained model
python scripts/sample.py \
    --checkpoint outputs/checkpoint_best.pth \
    --output_dir samples \
    --num_samples 64 \
    --seed 42
```

### 3. Interactive Demo

```bash
# Launch Streamlit demo
streamlit run demo/app.py
```

## Project Structure

```
score-based-generative-models/
├── src/
│   ├── models/
│   │   └── score_network.py      # Score network implementation
│   ├── data/
│   │   └── __init__.py           # Data pipeline and utilities
│   └── utils/
│       └── metrics.py            # Evaluation metrics
├── scripts/
│   ├── train.py                  # Training script
│   └── sample.py                 # Sampling script
├── configs/
│   └── train.yaml               # Training configuration
├── demo/
│   └── app.py                   # Streamlit demo
├── tests/                       # Unit tests
├── assets/                      # Generated samples and visualizations
├── requirements.txt             # Dependencies
├── pyproject.toml              # Project configuration
└── README.md                   # This file
```

## Configuration

The training configuration is defined in `configs/train.yaml`:

```yaml
# Model configuration
model:
  in_channels: 3
  base_channels: 64
  num_layers: 4
  time_embed_dim: 128
  use_attention: true

# Training configuration
training:
  num_epochs: 100
  batch_size: 64
  learning_rate: 0.0002
  sigma_min: 0.01
  sigma_max: 1.0
  eval_interval: 5
  sample_interval: 10

# Sampling configuration
sampling:
  num_steps: 1000
  step_size: 0.00002
  sigma_min: 0.01
  sigma_max: 1.0
```

## Usage

### Training

```python
from src.models.score_network import ScoreNetwork, ScoreMatchingLoss
from src.data import ImageDataset, get_transforms

# Create model
model = ScoreNetwork(
    in_channels=3,
    base_channels=64,
    num_layers=4,
    use_attention=True
)

# Create loss function
loss_fn = ScoreMatchingLoss(sigma_min=0.01, sigma_max=1.0)

# Create dataset
transform = get_transforms(image_size=32, augment=True)
dataset = ImageDataset("data/cifar10", transform=transform)
```

### Sampling

```python
from src.models.score_network import LangevinSampler

# Create sampler
sampler = LangevinSampler(
    model,
    sigma_min=0.01,
    sigma_max=1.0,
    num_steps=1000,
    step_size=0.00002
)

# Generate samples
samples = sampler.sample(
    shape=(3, 32, 32),
    batch_size=64
)
```

### Evaluation

```python
from src.utils.metrics import GenerativeMetrics

# Create metrics evaluator
metrics = GenerativeMetrics(device)

# Compute metrics
results = metrics.evaluate_all(real_images, fake_images)
print(f"FID: {results['fid']:.4f}")
print(f"IS: {results['is_mean']:.4f}")
```

## Evaluation Metrics

The implementation includes comprehensive evaluation metrics:

- **FID (Fréchet Inception Distance)**: Measures distribution similarity
- **IS (Inception Score)**: Measures quality and diversity
- **LPIPS**: Perceptual similarity metric
- **Precision/Recall**: Measures quality and coverage

### Model Leaderboard

Results are automatically tracked in a leaderboard:

```python
from src.utils.metrics import ModelLeaderboard

leaderboard = ModelLeaderboard("outputs/leaderboard.json")
leaderboard.add_result("my_model", metrics, config)
leaderboard.print_leaderboard()
```

## Advanced Features

### Custom Datasets

```python
from src.data import ImageDataset

# Custom dataset
dataset = ImageDataset(
    data_dir="path/to/images",
    transform=get_transforms(image_size=64),
    max_samples=10000
)
```

### Model Checkpointing

```python
# Save checkpoint
torch.save({
    'model_state_dict': model.state_dict(),
    'optimizer_state_dict': optimizer.state_dict(),
    'epoch': epoch,
    'config': config
}, 'checkpoint.pth')

# Load checkpoint
checkpoint = torch.load('checkpoint.pth')
model.load_state_dict(checkpoint['model_state_dict'])
```

### Wandb Integration

Enable Wandb logging in the config:

```yaml
logging:
  use_wandb: true
```

## Performance Tips

1. **Batch Size**: Use larger batch sizes for better gradient estimates
2. **Learning Rate**: Start with 0.0002 and adjust based on training stability
3. **Noise Schedule**: Use sigma_min=0.01 and sigma_max=1.0 for most datasets
4. **Sampling Steps**: More steps (1000+) for better quality
5. **GPU Memory**: Use gradient accumulation for large models

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Reduce batch size or use gradient accumulation
2. **Poor Sample Quality**: Increase sampling steps or adjust noise schedule
3. **Training Instability**: Reduce learning rate or increase gradient clipping
4. **Slow Sampling**: Reduce sampling steps or use faster schedulers

### Debug Mode

Enable debug logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

### Development Setup

```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Format code
black src/ scripts/ tests/
ruff check src/ scripts/ tests/

# Run pre-commit hooks
pre-commit install
pre-commit run --all-files
```

## License

MIT License - see LICENSE file for details.

## Citation

If you use this code in your research, please cite:

```bibtex
@software{score_based_generative_models,
  title={Score-based Generative Models},
  author={Kryptologyst},
  year={2025},
  url={https://github.com/kryptologyst/Score-based-Generative-Models}
}
```

## Acknowledgments

- Original score matching paper by Hyvärinen
- Langevin dynamics sampling methods
- U-Net architecture for generative modeling
- PyTorch and the open-source ML community
# Score-based-Generative-Models
