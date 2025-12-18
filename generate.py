#!/usr/bin/env python3
"""
Text generation script for KANGPT models.

Example:
    python generate.py --checkpoint checkpoints/last.ckpt --prompt "The quick brown fox"
"""

import argparse
import torch
from transformers import AutoTokenizer

from kangpt import KANGPTLM


def main():
    parser = argparse.ArgumentParser(description="Generate text with KANGPT")

    parser.add_argument('--checkpoint', type=str, required=True,
                       help='Path to Lightning checkpoint (.ckpt)')
    parser.add_argument('--prompt', type=str, default="The",
                       help='Prompt text')
    parser.add_argument('--max_tokens', type=int, default=100,
                       help='Maximum tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.8,
                       help='Sampling temperature')
    parser.add_argument('--top_k', type=int, default=40,
                       help='Top-k filtering (0 to disable)')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device: auto, cuda, or cpu')

    args = parser.parse_args()

    # Setup device
    if args.device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
    else:
        device = args.device
    print(f"Using device: {device}")

    # Load model from Lightning checkpoint
    print(f"Loading model from {args.checkpoint}...")
    model = KANGPTLM.load_from_checkpoint(args.checkpoint, map_location=device)
    model.eval()
    model.to(device)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained('gpt2')
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({'pad_token': '<PAD>'})

    # Tokenize prompt
    prompt_ids = tokenizer.encode(args.prompt)
    print(f"\nPrompt: {args.prompt}")
    print(f"Prompt tokens: {len(prompt_ids)}")

    # Generate
    print(f"\nGenerating {args.max_tokens} tokens...")
    output_ids = model.generate(
        prompt_ids,
        max_length=args.max_tokens,
        temperature=args.temperature,
        top_k=args.top_k if args.top_k > 0 else None
    )

    # Decode
    generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)

    print("\n" + "=" * 60)
    print("Generated text:")
    print("=" * 60)
    print(generated_text)
    print("=" * 60)


if __name__ == '__main__':
    main()
