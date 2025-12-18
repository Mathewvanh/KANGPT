# KANGPT

**Kolmogorov-Arnold Networks for GPT** - A transformer language model that replaces MLP layers with learnable polynomial transformations.

## Overview

KANGPT implements [Kolmogorov-Arnold Networks (KAN)](https://arxiv.org/abs/2404.19756) as a drop-in replacement for the MLP layers in GPT-style transformers. Instead of the standard `Linear -> GELU -> Linear` pattern, KANGPT uses learnable Chebyshev polynomial basis functions.

**Key idea**: Each output neuron computes a weighted sum of polynomial basis functions applied to inputs:

```
output_j = Σᵢ Σₖ C[j,i,k] · Tₖ(xᵢ)
```

where `Tₖ` is the Chebyshev polynomial of degree `k`.

## Installation

```bash
git clone https://github.com/Mathewvanh/KANGPT.git
cd KANGPT
pip install -r requirements.txt
```

## Quick Start

### 1. Download Data

Download and tokenize FineWeb-Edu dataset:

```bash
# Download 100M tokens for testing
python download_data.py --output_dir data --num_tokens 100M

# Download 10B tokens for full training
python download_data.py --output_dir data --num_tokens 10B
```

### 2. Train

```bash
python train.py \
    --train_data data/train.bin \
    --val_data data/val.bin \
    --num_layers 12 \
    --hidden_size 768 \
    --n_head 12 \
    --kan_degree 5 \
    --batch_size 32 \
    --learning_rate 6e-4 \
    --max_epochs 1
```

### 3. Generate Text

```bash
python generate.py \
    --checkpoint checkpoints/last.ckpt \
    --prompt "The quick brown fox" \
    --max_tokens 100
```

## Model Architecture

KANGPT follows the GPT-2 architecture with KAN-based MLPs:

| Component | Standard GPT | KANGPT |
|-----------|-------------|--------|
| Attention | Multi-head causal self-attention | Same |
| MLP | Linear → GELU → Linear | KAN → LayerNorm → tanh → KAN |
| Normalization | LayerNorm | Same |

The KAN layer uses Chebyshev polynomials as basis functions, which are orthogonal on [-1, 1]. The `tanh` activation squashes inputs to this range.

## Configuration

Key hyperparameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `n_layer` | 12 | Number of transformer layers |
| `n_head` | 12 | Number of attention heads |
| `n_embd` | 768 | Embedding dimension |
| `kan_degree` | 5 | Maximum polynomial degree |
| `kan_hidden_multiplier` | 4.0 | Hidden dim multiplier (like standard MLP) |

## API Usage

```python
from kangpt import KANGPT, KANGPTConfig, KANGPTLM

# Create model directly
config = KANGPTConfig(
    n_layer=12,
    n_head=12,
    n_embd=768,
    vocab_size=50257,
    kan_degree=5,
)
model = KANGPT(config)

# Or use Lightning wrapper for training
model = KANGPTLM(
    vocab_size=50257,
    hidden_size=768,
    num_layers=12,
    kan_degree=5,
    learning_rate=6e-4,
)
```

## Project Structure

```
KANGPT/
├── kangpt/
│   ├── __init__.py
│   ├── config.py        # KANGPTConfig dataclass
│   ├── kan_layer.py     # KANLayer with Chebyshev polynomials
│   ├── model.py         # KANGPT model and components
│   └── lightning.py     # PyTorch Lightning wrapper
├── train.py             # Training script
├── generate.py          # Text generation
├── download_data.py     # Dataset preparation
├── requirements.txt
└── LICENSE
```

## Citation

If you use this code, please cite:

```bibtex
@software{kangpt2025,
  author = {Vanherreweghe, Mathew},
  organization = {Pebblebed},
  title = {KANGPT: Kolmogorov-Arnold Networks for GPT},
  year = {2025},
  url = {https://github.com/Mathewvanh/KANGPT}
}
```

## License

This project is licensed under CC BY-NC-SA 4.0. See [LICENSE](LICENSE) for details.

## Acknowledgments

- [KAN: Kolmogorov-Arnold Networks](https://arxiv.org/abs/2404.19756) - Liu et al.
- [nanoGPT](https://github.com/karpathy/nanoGPT) - Andrej Karpathy
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) - HuggingFace
