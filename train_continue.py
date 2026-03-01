#!/usr/bin/env python3
"""Continue training a mask-based model from checkpoint.

Loads existing checkpoint and trains further with periodic autoregressive
scoring on 2000 pairs to catch carry edge cases.
"""

import argparse
import sys
import time

import torch
import torch.nn.functional as F

from train_adder import TrainableAdder, CONFIGS, generate_batch, export_submission


def score_autoregressive(model, n_pairs=2000, seed=2025, device='cpu'):
    """Full autoregressive scoring matching verify.py methodology."""
    model.eval()
    correct = 0
    rng = torch.Generator().manual_seed(seed)
    with torch.no_grad():
        for _ in range(n_pairs):
            a_val = torch.randint(0, 10**10, (1,), generator=rng).item()
            b_val = torch.randint(0, 10**10, (1,), generator=rng).item()
            expected = a_val + b_val

            sa = f"{a_val:010d}"[::-1]
            sb = f"{b_val:010d}"[::-1]
            seq = [int(c) for c in sa] + [10] + [int(c) for c in sb] + [11]

            for _ in range(11):
                xt = torch.tensor([seq], dtype=torch.long, device=device)
                out = model(xt)
                seq.append(out[0, -1].argmax().item())

            c_digits = seq[22:33]
            result = sum(d * (10**i) for i, d in enumerate(c_digits))
            if result == expected:
                correct += 1
    return correct / n_pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mask-large")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint to resume from (e.g. trained_mask-large.pt)")
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=500000)
    parser.add_argument("--score-interval", type=int, default=5000)
    parser.add_argument("--score-pairs", type=int, default=2000)
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    device = torch.device(
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )

    model = TrainableAdder(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, weights_only=True))
        print(f"Loaded checkpoint: {args.checkpoint}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.max_steps, eta_min=args.lr * 0.01
    )

    print(f"Config: {args.config} ({n_params} params)")
    print(f"Device: {device}, LR: {args.lr}, Steps: {args.max_steps}")
    print(f"Scoring: {args.score_pairs} pairs every {args.score_interval} steps")
    print()

    best_autoreg = 0.0
    best_step = 0
    t0 = time.time()
    moving_acc = 0.0

    for step in range(args.max_steps + 1):
        model.train()
        seq = generate_batch(batch_size=args.batch_size, max_digits=10, device=device)
        x = seq[:, :-1]
        y = seq[:, 1:]
        logits = model(x)

        loss = F.cross_entropy(
            logits[:, 21:32].reshape(-1, 12), y[:, 21:32].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        preds = logits[:, 21:32].argmax(dim=-1)
        correct = (preds == y[:, 21:32]).all(dim=1).float().mean().item()
        moving_acc = 0.95 * moving_acc + 0.05 * correct

        if step % 1000 == 0:
            elapsed = time.time() - t0
            lr = optimizer.param_groups[0]['lr']
            print(
                f"  Step {step:<6} | Loss {loss.item():.4f} | "
                f"TF {moving_acc*100:5.1f}% | LR {lr:.5f} | {elapsed:.0f}s"
            )
            sys.stdout.flush()

        if step > 0 and step % args.score_interval == 0:
            autoreg = score_autoregressive(
                model, n_pairs=args.score_pairs, seed=2025, device=device
            )
            elapsed = time.time() - t0
            print(
                f"  >>> AUTOREG: {autoreg*100:.1f}% "
                f"({int(autoreg*args.score_pairs)}/{args.score_pairs}) "
                f"[best: {best_autoreg*100:.1f}% @ step {best_step}] "
                f"[{elapsed:.0f}s]"
            )
            sys.stdout.flush()

            if autoreg > best_autoreg:
                best_autoreg = autoreg
                best_step = step
                torch.save(model.state_dict(), f"trained_{args.config}_best.pt")
                print(f"  >>> NEW BEST! Saved trained_{args.config}_best.pt")

            if autoreg >= 0.995:
                print(f"\n  TARGET REACHED: {autoreg*100:.1f}% at step {step}")
                export_submission(model, args.config, n_params)
                return

    # Final scoring and export best
    print(f"\nTraining complete. Best: {best_autoreg*100:.1f}% at step {best_step}")
    if best_autoreg > 0.90:
        model.load_state_dict(torch.load(f"trained_{args.config}_best.pt", weights_only=True))
        export_submission(model, args.config, n_params)


if __name__ == "__main__":
    main()
