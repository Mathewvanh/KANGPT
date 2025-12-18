#!/usr/bin/env python3
"""
Download and prepare FineWeb-Edu dataset for KANGPT training.

This script downloads a subset of the FineWeb-Edu dataset from HuggingFace,
tokenizes it using GPT-2 tokenizer, and saves as memory-mapped .bin files.

Example:
    # Download 10B tokens (default)
    python download_data.py --output_dir data

    # Download smaller subset for testing
    python download_data.py --output_dir data --num_tokens 100M
"""

import argparse
from pathlib import Path

import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer


def parse_num_tokens(s: str) -> int:
    """Parse token count string like '10B', '100M'."""
    s = s.upper().strip()
    multipliers = {'K': 1_000, 'M': 1_000_000, 'B': 1_000_000_000, 'T': 1_000_000_000_000}

    for suffix, mult in multipliers.items():
        if s.endswith(suffix):
            return int(float(s[:-1]) * mult)
    return int(s)


def download_and_tokenize(output_dir: str, num_tokens: int, val_ratio: float = 0.01):
    """Download FineWeb-Edu and create tokenized .bin files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Target: {num_tokens:,} tokens")
    print(f"Output: {output_dir}")

    # Load tokenizer
    print("\nLoading GPT-2 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    # Load dataset
    print("\nLoading FineWeb-Edu dataset...")
    try:
        from datasets import load_dataset
    except ImportError:
        print("Error: 'datasets' package not installed.")
        print("Install with: pip install datasets")
        return

    # Choose subset based on size
    if num_tokens <= 10_000_000_000:
        subset = "sample-10BT"
    else:
        subset = "sample-100BT"

    print(f"Using HuggingFaceFW/fineweb-edu ({subset})")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        subset,
        split='train',
        streaming=True,
        trust_remote_code=True
    )

    # Tokenize and collect
    print("\nTokenizing...")
    all_tokens = []
    total_tokens = 0

    pbar = tqdm(total=num_tokens, unit='tok', unit_scale=True)
    for example in dataset:
        text = example.get('text', '')
        if not text:
            continue

        tokens = tokenizer.encode(text)
        all_tokens.extend(tokens)
        total_tokens += len(tokens)
        pbar.update(len(tokens))

        if total_tokens >= num_tokens:
            break
    pbar.close()

    all_tokens = all_tokens[:num_tokens]
    print(f"\nCollected {len(all_tokens):,} tokens")

    # Split train/val
    val_size = int(len(all_tokens) * val_ratio)
    train_tokens = all_tokens[:-val_size] if val_size > 0 else all_tokens
    val_tokens = all_tokens[-val_size:] if val_size > 0 else []

    print(f"Train: {len(train_tokens):,} tokens")
    print(f"Val: {len(val_tokens):,} tokens")

    # Save as .bin files
    def save_bin(tokens, path):
        arr = np.array(tokens, dtype=np.uint16)
        arr.tofile(path)
        print(f"Saved {path} ({arr.nbytes / 1e9:.2f} GB)")

    save_bin(train_tokens, output_dir / 'train.bin')
    if val_tokens:
        save_bin(val_tokens, output_dir / 'val.bin')

    print("\nDone! Train with:")
    print(f"  python train.py --train_data {output_dir}/train.bin --val_data {output_dir}/val.bin")


def main():
    parser = argparse.ArgumentParser(description="Download FineWeb-Edu dataset")

    parser.add_argument('--output_dir', type=str, default='data',
                       help='Output directory')
    parser.add_argument('--num_tokens', type=str, default='10B',
                       help='Number of tokens (e.g., 100M, 10B)')
    parser.add_argument('--val_ratio', type=float, default=0.01,
                       help='Validation split ratio')

    args = parser.parse_args()
    download_and_tokenize(args.output_dir, parse_num_tokens(args.num_tokens), args.val_ratio)


if __name__ == '__main__':
    main()
