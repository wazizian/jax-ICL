# GPT2 Linear Regression Training

A well-organized project template for training a GPT2 model on linear regression data using Flax, JAX, and Hydra.

## Project Structure

```
gpt2_linear_regression/
├── configs/                    # Hydra configuration files
│   ├── config.yaml            # Main configuration
│   ├── data/
│   │   └── linear_regression.yaml
│   ├── model/
│   │   └── gpt2_small.yaml
│   └── training/
│       └── default.yaml
├── src/                       # Source code
│   ├── data/
│   │   └── dataset.py         # Linear regression data generation
│   ├── models/
│   │   └── gpt2_model.py      # GPT2 model implementation in Flax
│   └── training/
│       └── trainer.py         # Training loop with JAX compilation
├── scripts/
│   └── run_training.sh        # Training script
├── train.py                   # Main training entry point
├── requirements.txt           # Python dependencies
└── setup.py                  # Package setup
```

## Features

- **GPT2 Model**: Full GPT2 implementation using Flax/Linen
- **Linear Regression Data**: Synthetic data generation with Gaussian priors
  - Sequences of (x_i, y_i) pairs where y_i = w^T x_i + noise
  - Weight vectors w sampled from Gaussian prior for each sequence
- **Hydra Configuration**: Modular configuration management
- **JAX Compilation**: Fully compiled training and evaluation steps
- **Checkpointing**: Model checkpoint saving and loading
- **Logging**: Comprehensive training logging

## Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd gpt2_linear_regression
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

Or install in development mode:
```bash
pip install -e .
```

## Usage

### Basic Training

Run training with default configuration:
```bash
python train.py
```

Or use the convenience script:
```bash
./scripts/run_training.sh
```

### Configuration Override

Override specific parameters:
```bash
python train.py training.learning_rate=1e-3 training.num_epochs=20
```

Override configuration files:
```bash
python train.py model=gpt2_small data=linear_regression training=default
```

### Custom Configuration

Create custom configuration files in the `configs/` directory and reference them:
```bash
python train.py --config-name=my_custom_config
```

## Configuration

### Model Configuration (`configs/model/gpt2_small.yaml`)
- Vocabulary size, embedding dimension, number of layers
- Attention heads, dropout rates
- Initialization parameters

### Data Configuration (`configs/data/linear_regression.yaml`)
- Input dimension, sequence length
- Gaussian prior parameters for weight vectors
- Noise parameters, tokenization settings
- Batch size and data generation parameters

### Training Configuration (`configs/training/default.yaml`)
- Learning rate, optimizer parameters
- Training schedule, evaluation frequency
- Checkpointing and logging settings

## Data Format

The model receives sequences of tokenized (x_i, y_i) pairs:
- Each sequence represents a linear regression problem
- Format: [x0_0, x0_1, ..., x0_d, y0, x1_0, x1_1, ..., x1_d, y1, ...]
- Values are discretized into tokens for language modeling

## Model Architecture

- Standard GPT2 architecture implemented in Flax
- Causal self-attention with multiple heads
- Layer normalization and residual connections
- Language modeling head for next-token prediction

## Training Features

- **Full JAX Compilation**: Both training and evaluation steps are JIT-compiled
- **Gradient Clipping**: Prevents exploding gradients
- **Learning Rate Scheduling**: Linear warmup followed by constant rate
- **Weight Decay**: AdamW optimizer with configurable weight decay
- **Checkpointing**: Automatic model checkpoint saving
- **Evaluation**: Regular evaluation on test set during training

## Output

Training outputs are saved in `outputs/` directory with timestamp:
- Model checkpoints
- Training logs
- Hydra configuration snapshots

## Requirements

- Python 3.8+
- JAX/JAXLib
- Flax
- Optax
- Transformers (for reference, not directly used)
- Hydra-core
- NumPy

## License

MIT License