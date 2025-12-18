"""
KANGPT Model - GPT with KAN-based MLP layers.

Replaces the standard MLP (Linear -> GELU -> Linear) with KAN layers
that use learnable Chebyshev polynomial transformations.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from .config import KANGPTConfig
from .kan_layer import KANLayer


class LayerNorm(nn.Module):
    """LayerNorm with optional bias."""

    def __init__(self, ndim: int, bias: bool = True):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(ndim))
        self.bias = nn.Parameter(torch.zeros(ndim)) if bias else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(x, self.weight.shape, self.weight, self.bias, 1e-5)


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention with Flash Attention support."""

    def __init__(self, config: KANGPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0

        # Key, query, value projections for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd, bias=config.bias)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd, bias=config.bias)

        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.dropout = config.dropout

        # Flash attention if available
        self.flash = hasattr(F, 'scaled_dot_product_attention')
        if not self.flash:
            self.register_buffer(
                "bias",
                torch.tril(torch.ones(config.block_size, config.block_size))
                .view(1, 1, config.block_size, config.block_size)
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.size()

        # Calculate q, k, v for all heads
        q, k, v = self.c_attn(x).split(self.n_embd, dim=2)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2)

        if self.flash:
            y = F.scaled_dot_product_attention(
                q, k, v,
                attn_mask=None,
                dropout_p=self.dropout if self.training else 0,
                is_causal=True
            )
        else:
            att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
            att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
            att = F.softmax(att, dim=-1)
            att = self.attn_dropout(att)
            y = att @ v

        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.resid_dropout(self.c_proj(y))


class KANMLP(nn.Module):
    """KAN-based MLP that replaces standard MLP in transformer blocks.

    Standard MLP: Linear(d -> 4d) -> GELU -> Linear(4d -> d)
    KANMLP: KAN(d -> 4d) -> LayerNorm -> tanh -> KAN(4d -> d)

    The tanh is necessary because Chebyshev polynomials are defined on [-1, 1].
    """

    def __init__(self, config: KANGPTConfig):
        super().__init__()

        hidden_size = int(config.kan_hidden_multiplier * config.n_embd)

        # KAN layers replace linear layers
        self.kan_fc = KANLayer(
            in_dim=config.n_embd,
            out_dim=hidden_size,
            degree=config.kan_degree
        )
        self.kan_proj = KANLayer(
            in_dim=hidden_size,
            out_dim=config.n_embd,
            degree=config.kan_degree
        )

        self.ln = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, n_embd = x.shape

        # Flatten for KAN processing
        x_flat = x.view(-1, n_embd)

        # Apply tanh to keep inputs in [-1, 1] for Chebyshev
        x_scaled = torch.tanh(x_flat)

        # First KAN layer
        hidden = self.kan_fc(x_scaled)

        # LayerNorm + tanh for second KAN layer
        hidden = self.ln(hidden)
        hidden = torch.tanh(hidden)

        # Second KAN layer
        output = self.kan_proj(hidden)

        # Reshape and dropout
        output = output.view(batch_size, seq_len, n_embd)
        return self.dropout(output)


class KANBlock(nn.Module):
    """Transformer block with KAN-based MLP."""

    def __init__(self, config: KANGPTConfig):
        super().__init__()
        self.ln_1 = LayerNorm(config.n_embd, bias=config.bias)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = LayerNorm(config.n_embd, bias=config.bias)
        self.mlp = KANMLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x


class KANGPT(nn.Module):
    """GPT model with KAN-based MLP layers.

    Architecture follows GPT-2 but replaces standard MLPs with KAN layers
    that use Chebyshev polynomial basis functions.
    """

    def __init__(self, config: KANGPTConfig):
        super().__init__()
        assert config.vocab_size is not None
        assert config.block_size is not None
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte=nn.Embedding(config.vocab_size, config.n_embd),
            wpe=nn.Embedding(config.block_size, config.n_embd),
            drop=nn.Dropout(config.dropout),
            h=nn.ModuleList([KANBlock(config) for _ in range(config.n_layer)]),
            ln_f=LayerNorm(config.n_embd, bias=config.bias),
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight tying
        self.transformer.wte.weight = self.lm_head.weight

        # Initialize weights
        self.apply(self._init_weights)

        # Scale KAN projection layers like GPT-2 scales c_proj
        for pn, p in self.named_parameters():
            if pn.endswith('kan_proj.C'):
                with torch.no_grad():
                    p.data *= 1.0 / np.sqrt(2 * config.n_layer)

        print(f"KANGPT initialized with {self.get_num_params()/1e6:.2f}M parameters")

    def get_num_params(self, non_embedding: bool = True) -> int:
        """Return number of parameters (excluding position embeddings by default)."""
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.transformer.wpe.weight.numel()
        return n_params

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
        """Forward pass.

        Args:
            idx: Input token indices of shape (batch, seq_len)
            targets: Target token indices of shape (batch, seq_len) for loss calculation

        Returns:
            logits: Output logits of shape (batch, seq_len, vocab_size) or (batch, 1, vocab_size)
            loss: Cross-entropy loss if targets provided, else None
        """
        device = idx.device
        b, t = idx.size()
        assert t <= self.config.block_size, \
            f"Sequence length {t} exceeds block size {self.config.block_size}"

        pos = torch.arange(0, t, dtype=torch.long, device=device)

        # Forward through transformer
        tok_emb = self.transformer.wte(idx)
        pos_emb = self.transformer.wpe(pos)
        x = self.transformer.drop(tok_emb + pos_emb)

        for block in self.transformer.h:
            x = block(x)

        x = self.transformer.ln_f(x)

        if targets is not None:
            logits = self.lm_head(x)
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
        else:
            # Inference optimization: only compute last position
            logits = self.lm_head(x[:, [-1], :])
            loss = None

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int = None
    ) -> torch.Tensor:
        """Generate tokens autoregressively.

        Args:
            idx: Conditioning sequence of shape (batch, seq_len)
            max_new_tokens: Number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_k: If set, only sample from top k most likely tokens

        Returns:
            Generated sequence of shape (batch, seq_len + max_new_tokens)
        """
        for _ in range(max_new_tokens):
            # Crop to block size if needed
            idx_cond = idx if idx.size(1) <= self.config.block_size else idx[:, -self.config.block_size:]

            # Get logits
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature

            # Optional top-k filtering
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float('-inf')

            # Sample
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    def configure_optimizers(
        self,
        weight_decay: float,
        learning_rate: float,
        betas: tuple = (0.9, 0.95),
        device_type: str = 'cuda'
    ):
        """Configure AdamW optimizer with weight decay.

        2D parameters (weights) get weight decay, 1D parameters (biases, norms) don't.
        """
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}

        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"Optimizer: {len(decay_params)} decay tensors ({num_decay:,} params), "
              f"{len(nodecay_params)} no-decay tensors ({num_nodecay:,} params)")

        # Use fused AdamW if available
        import inspect
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, **extra_args)
        print(f"Using fused AdamW: {use_fused}")

        return optimizer
