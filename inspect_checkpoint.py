#!/usr/bin/env python3
"""Inspect trained checkpoint weights.

Usage:
    uv run python inspect_checkpoint.py checkpoints/trained_mask-w6_best.pt
    uv run python inspect_checkpoint.py checkpoints/trained_mask-w6_best.pt --verbose
    uv run python inspect_checkpoint.py --all
"""

import argparse
import os
import sys

import torch


def inspect(path, verbose=False):
    sd = torch.load(path, weights_only=True)
    size_kb = os.path.getsize(path) / 1024

    # Separate buffers from learnable params
    buffer_keys = {"fixed_mask", "alibi", "pe"}
    learnable = {k: v for k, v in sd.items() if k not in buffer_keys}
    buffers = {k: v for k, v in sd.items() if k in buffer_keys}

    n_learnable = sum(v.numel() for v in learnable.values())
    n_buffer = sum(v.numel() for v in buffers.values())

    print(f"\n{os.path.basename(path)} ({size_kb:.1f} KB)")
    print(f"  Learnable parameters: {n_learnable}")
    print(f"  Buffer elements: {n_buffer}")
    print(f"  Tensors: {len(sd)}")
    print()

    # Print tensor inventory
    print("  Learnable weights:")
    for k, v in learnable.items():
        stats = f"mean={v.float().mean():.4f}, std={v.float().std():.4f}, range=[{v.min():.3f}, {v.max():.3f}]"
        print(f"    {k:25s} {str(tuple(v.shape)):15s} {v.numel():5d} params  {stats}")

    if buffers:
        print("\n  Buffers (non-learnable):")
        for k, v in buffers.items():
            finite = v[v != float('-inf')]
            if len(finite) > 0:
                stats = f"finite values: mean={finite.mean():.4f}, range=[{finite.min():.3f}, {finite.max():.3f}]"
            else:
                stats = "all -inf"
            print(f"    {k:25s} {str(tuple(v.shape)):15s} {v.numel():5d} elems   {stats}")

    if verbose:
        print("\n  Full weight values:")
        for k, v in learnable.items():
            print(f"\n    {k} {tuple(v.shape)}:")
            if v.dim() == 1:
                print(f"      {v.tolist()}")
            elif v.dim() == 2 and v.shape[0] <= 16:
                for row_idx in range(v.shape[0]):
                    print(f"      [{', '.join(f'{x:.6f}' for x in v[row_idx].tolist())}]")
            else:
                print(f"      (too large to display, shape {tuple(v.shape)})")

    return n_learnable


def main():
    parser = argparse.ArgumentParser(description="Inspect checkpoint weights")
    parser.add_argument("path", nargs="?", help="Path to .pt checkpoint file")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print full weight values")
    parser.add_argument("--all", action="store_true",
                        help="Inspect all checkpoints in checkpoints/")
    args = parser.parse_args()

    if args.all:
        ckpt_dir = "checkpoints"
        if not os.path.isdir(ckpt_dir):
            print(f"No {ckpt_dir}/ directory found")
            sys.exit(1)
        for f in sorted(os.listdir(ckpt_dir)):
            if f.endswith(".pt"):
                inspect(os.path.join(ckpt_dir, f), verbose=args.verbose)
    elif args.path:
        inspect(args.path, verbose=args.verbose)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
