"""
PyTorch Lightning wrapper for KANGPT models.
"""

import math
import torch
import torch.nn.functional as F
import lightning as L
from torch import Tensor
from typing import Optional, Any

from .config import KANGPTConfig
from .model import KANGPT


class KANGPTLM(L.LightningModule):
    """PyTorch Lightning wrapper for KANGPT model."""

    def __init__(
        self,
        vocab_size: int,
        hidden_size: int = 768,
        num_layers: int = 12,
        n_head: int = 12,
        block_size: int = 1024,
        dropout: float = 0.0,
        learning_rate: float = 6e-4,
        weight_decay: float = 0.1,
        kan_degree: int = 5,
        kan_hidden_multiplier: float = 4.0,
        pad_token_id: int = 0,
        betas: tuple = (0.9, 0.95),
        # Warmup + Cosine scheduler
        use_warmup_cosine_scheduler: bool = True,
        warmup_steps: int = 350,
        max_training_steps: Optional[int] = None,
        warmup_start_lr_ratio: float = 0.1,
        min_lr_ratio: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.n_head = n_head
        self.block_size = block_size
        self.dropout = dropout
        self.pad_token_id = pad_token_id
        self.betas = betas

        # Create config and model
        config = KANGPTConfig(
            n_layer=num_layers,
            n_head=n_head,
            n_embd=hidden_size,
            vocab_size=vocab_size,
            block_size=block_size,
            kan_degree=kan_degree,
            kan_hidden_multiplier=kan_hidden_multiplier,
            dropout=dropout,
        )

        self.kan_gpt = KANGPT(config)

    def forward(self, x: Tensor, targets: Optional[Tensor] = None):
        """Forward pass."""
        return self.kan_gpt(x, targets)

    def training_step(self, batch: tuple, batch_idx: int) -> Tensor:
        x, y = batch

        # Forward pass
        logits, loss = self.kan_gpt(x, y)

        # Log metrics
        self.log('train/loss', loss, prog_bar=True)
        self.log('train/ppl', torch.exp(loss), prog_bar=True)

        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> Tensor:
        x, y = batch

        logits, loss = self.kan_gpt(x, y)

        # Calculate accuracy
        preds = logits.argmax(dim=-1)
        mask = (y != self.pad_token_id)
        correct = (preds == y) & mask
        acc = correct.sum().float() / mask.sum().float()

        self.log('val/loss', loss, prog_bar=True, sync_dist=True)
        self.log('val/ppl', torch.exp(loss), prog_bar=True, sync_dist=True)
        self.log('val/acc', acc, prog_bar=True, sync_dist=True)

        return loss

    def configure_optimizers(self):
        """Configure AdamW optimizer with warmup + cosine schedule."""
        # Get param groups (2D params get weight decay, 1D params don't)
        param_dict = {pn: p for pn, p in self.named_parameters() if p.requires_grad}
        decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
        nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]

        optim_groups = [
            {'params': decay_params, 'weight_decay': self.hparams.weight_decay},
            {'params': nodecay_params, 'weight_decay': 0.0}
        ]

        num_decay = sum(p.numel() for p in decay_params)
        num_nodecay = sum(p.numel() for p in nodecay_params)
        print(f"Optimizer: {len(decay_params)} decay tensors ({num_decay:,} params), "
              f"{len(nodecay_params)} no-decay tensors ({num_nodecay:,} params)")

        # Use fused AdamW if available
        import inspect
        device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
        fused_available = 'fused' in inspect.signature(torch.optim.AdamW).parameters
        use_fused = fused_available and device_type == 'cuda'
        extra_args = dict(fused=True) if use_fused else dict()

        optimizer = torch.optim.AdamW(
            optim_groups,
            lr=self.hparams.learning_rate,
            betas=self.betas,
            **extra_args
        )
        print(f"Using fused AdamW: {use_fused}")

        # Setup scheduler
        if self.hparams.use_warmup_cosine_scheduler:
            if self.hparams.max_training_steps is None:
                raise ValueError("max_training_steps must be set for warmup+cosine scheduler")

            warmup_steps = self.hparams.warmup_steps
            max_steps = self.hparams.max_training_steps

            # Linear warmup
            warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=self.hparams.warmup_start_lr_ratio,
                end_factor=1.0,
                total_iters=warmup_steps
            )

            # Cosine decay
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max_steps - warmup_steps,
                eta_min=self.hparams.learning_rate * self.hparams.min_lr_ratio
            )

            # Combine
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer,
                schedulers=[warmup_scheduler, cosine_scheduler],
                milestones=[warmup_steps]
            )

            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1
                }
            }
        else:
            return optimizer

    @torch.no_grad()
    def generate(
        self,
        prompt_ids: list,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: Optional[int] = None,
    ) -> list:
        """Generate text from prompt."""
        self.eval()
        device = next(self.parameters()).device

        idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
        output = self.kan_gpt.generate(
            idx,
            max_new_tokens=max_length,
            temperature=temperature,
            top_k=top_k
        )
        return output[0].tolist()
