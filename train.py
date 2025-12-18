#!/usr/bin/env python3
"""
Training script for KANGPT using PyTorch Lightning.

Example:
    python train.py --train_data data/train.bin --val_data data/val.bin
"""

import os
import argparse
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from transformers import AutoTokenizer

from kangpt import KANGPTLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"


class TokenDataset(Dataset):
    """Dataset for loading pre-tokenized data from .bin files.

    Args:
        bin_path: Path to pre-tokenized .bin file (uint16 format)
        seq_length: Length of each sequence
        tokenizer_name: Tokenizer name for vocab size
    """

    def __init__(
        self,
        bin_path: str,
        seq_length: int,
        tokenizer_name: str = 'gpt2',
    ):
        self.seq_length = seq_length
        self.window_size = seq_length + 1  # +1 for target

        # Load tokenizer for vocab size
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.add_special_tokens({'pad_token': '<PAD>'})
            print(f"Added PAD token: '{self.tokenizer.pad_token}' (id: {self.tokenizer.pad_token_id})")

        self.vocab_size = len(self.tokenizer)
        self.pad_token_id = self.tokenizer.pad_token_id

        # Load data (memory-mapped)
        bin_path = Path(bin_path)
        if not bin_path.exists():
            raise FileNotFoundError(
                f"Data file not found: {bin_path}\n"
                f"Run: python download_data.py --output_dir {bin_path.parent}"
            )

        print(f"Loading pre-tokenized data from {bin_path}")
        self.tokens = np.memmap(str(bin_path), dtype=np.uint16, mode='r')

        self.num_sequences = max(0, (len(self.tokens) - self.window_size) // seq_length + 1)

        print(f"Dataset loaded:")
        print(f"  Total tokens: {len(self.tokens):,}")
        print(f"  Sequences: {self.num_sequences:,} (length={seq_length})")
        print(f"  Vocabulary size: {self.vocab_size}")

    def __len__(self) -> int:
        return self.num_sequences

    def __getitem__(self, idx: int):
        start = idx * self.seq_length
        end = start + self.window_size
        window = self.tokens[start:end]

        x = torch.from_numpy(window[:-1].astype(np.int64))
        y = torch.from_numpy(window[1:].astype(np.int64))
        return x, y


def main():
    parser = argparse.ArgumentParser(description="Train KANGPT with PyTorch Lightning")

    # Data
    parser.add_argument('--train_data', type=str, required=True,
                       help='Path to training .bin file')
    parser.add_argument('--val_data', type=str, required=True,
                       help='Path to validation .bin file')
    parser.add_argument('--tokenizer', type=str, default='gpt2',
                       help='Tokenizer name')

    # Model
    parser.add_argument('--num_layers', type=int, default=12,
                       help='Number of transformer layers')
    parser.add_argument('--hidden_size', type=int, default=768,
                       help='Hidden size / embedding dimension')
    parser.add_argument('--n_head', type=int, default=12,
                       help='Number of attention heads')
    parser.add_argument('--seq_length', type=int, default=1024,
                       help='Sequence length')
    parser.add_argument('--kan_degree', type=int, default=5,
                       help='Polynomial degree for KAN layers')
    parser.add_argument('--kan_hidden_multiplier', type=float, default=4.0,
                       help='Hidden dimension multiplier for KAN MLP')
    parser.add_argument('--dropout', type=float, default=0.0,
                       help='Dropout rate')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=6e-4,
                       help='Peak learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.1,
                       help='Weight decay')
    parser.add_argument('--max_epochs', type=int, default=1,
                       help='Maximum epochs')
    parser.add_argument('--warmup_steps', type=int, default=350,
                       help='Warmup steps')
    parser.add_argument('--min_lr_ratio', type=float, default=0.1,
                       help='Minimum LR as ratio of peak LR')
    parser.add_argument('--accumulate_grad_batches', type=int, default=16,
                       help='Gradient accumulation steps')
    parser.add_argument('--gradient_clip_val', type=float, default=1.0,
                       help='Gradient clipping value')
    parser.add_argument('--val_check_interval', type=float, default=1.0,
                       help='Validation check interval')

    # System
    parser.add_argument('--devices', type=int, default=1,
                       help='Number of GPUs')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='DataLoader workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Output directory for checkpoints')

    args = parser.parse_args()

    # Set seed
    L.seed_everything(args.seed, workers=True)

    print("=" * 80)
    print("Training KANGPT")
    print("=" * 80)

    # Load datasets
    print("\nLoading datasets...")
    train_dataset = TokenDataset(
        bin_path=args.train_data,
        seq_length=args.seq_length,
        tokenizer_name=args.tokenizer,
    )
    val_dataset = TokenDataset(
        bin_path=args.val_data,
        seq_length=args.seq_length,
        tokenizer_name=args.tokenizer,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Calculate max training steps
    optimizer_steps_per_epoch = len(train_loader) // args.accumulate_grad_batches
    max_training_steps = optimizer_steps_per_epoch * args.max_epochs

    print(f"\nScheduler configuration:")
    print(f"  Train batches per epoch: {len(train_loader):,}")
    print(f"  Gradient accumulation: {args.accumulate_grad_batches}")
    print(f"  Optimizer steps per epoch: {optimizer_steps_per_epoch:,}")
    print(f"  Total optimizer steps: {max_training_steps:,}")
    print(f"  Warmup steps: {args.warmup_steps}")
    print(f"  Peak LR: {args.learning_rate:.2e}")
    print(f"  Min LR: {args.learning_rate * args.min_lr_ratio:.2e}")

    # Create model
    print("\nCreating model...")
    model = KANGPTLM(
        vocab_size=train_dataset.vocab_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        n_head=args.n_head,
        block_size=args.seq_length,
        dropout=args.dropout,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        kan_degree=args.kan_degree,
        kan_hidden_multiplier=args.kan_hidden_multiplier,
        pad_token_id=train_dataset.pad_token_id,
        use_warmup_cosine_scheduler=True,
        warmup_steps=args.warmup_steps,
        max_training_steps=max_training_steps,
        min_lr_ratio=args.min_lr_ratio,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # Setup callbacks
    checkpoint_dir = Path(args.output_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    callbacks = [
        ModelCheckpoint(
            monitor='val/loss',
            mode='min',
            save_top_k=3,
            filename='{epoch}-{step}-{val/loss:.3f}',
            dirpath=str(checkpoint_dir),
        ),
        ModelCheckpoint(
            save_last=True,
            filename='last-{step}',
            dirpath=str(checkpoint_dir),
        ),
        LearningRateMonitor(logging_interval='step'),
    ]

    # Create trainer
    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator='auto',
        devices=args.devices,
        callbacks=callbacks,
        val_check_interval=args.val_check_interval,
        precision='16-mixed',
        accumulate_grad_batches=args.accumulate_grad_batches,
        gradient_clip_val=args.gradient_clip_val,
        enable_progress_bar=True,
    )

    # Train
    print("\nStarting training...")
    trainer.fit(model, train_loader, val_loader)

    print("\nTraining complete!")


if __name__ == '__main__':
    main()
