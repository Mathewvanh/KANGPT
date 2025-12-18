"""
Configuration for KANGPT models.
"""

from dataclasses import dataclass


@dataclass
class KANGPTConfig:
    """Configuration for KANGPT model.

    Args:
        n_layer: Number of transformer layers
        n_head: Number of attention heads
        n_embd: Embedding dimension (hidden size)
        vocab_size: Vocabulary size
        block_size: Maximum sequence length
        kan_degree: Polynomial degree for KAN layers
        kan_hidden_multiplier: Multiplier for KAN hidden dimension (default 4.0 like standard MLP)
        dropout: Dropout rate
        bias: Whether to use bias in linear layers
    """
    n_layer: int = 12
    n_head: int = 12
    n_embd: int = 768
    vocab_size: int = 50257
    block_size: int = 1024
    kan_degree: int = 5
    kan_hidden_multiplier: float = 4.0
    dropout: float = 0.0
    bias: bool = True
