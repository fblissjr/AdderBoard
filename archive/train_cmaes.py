#!/usr/bin/env python3
"""CMA-ES evolutionary search for transformer adder weights.

Uses Covariance Matrix Adaptation Evolution Strategy to find weights for a
1-layer transformer that adds two 10-digit numbers. Novel approach: no
gradients, no SGD -- pure black-box optimization.

Supports warm-start from hand-coded solution (submission_1l.py) or cold-start
from zero/random initialization.

Usage:
    uv run python train_cmaes.py --arch a --warm-start submission_1l.py --max-evals 5000
    uv run python train_cmaes.py --arch a --max-evals 50000
    uv run python train_cmaes.py --arch b --max-evals 50000
"""

import argparse
import json
import math
import random
import sys
import time

import cma
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Model architecture (mirrors submission_1l.py but parameterized)
# ---------------------------------------------------------------------------

VOCAB = 11  # 0-9 + SEP(10)
SEP = 10
MAX_SEQ = 34  # 22 input + 11 output + 1 buffer


class TransformerBlock(nn.Module):
    def __init__(self, d_model, n_heads, ff_dim):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.mlp_up = nn.Linear(d_model, ff_dim, bias=True)
        self.mlp_down = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x, mask):
        B, L, _D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) + mask
        attn = F.softmax(scores, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, L, _D)
        out = self.o_proj(out)

        x = x + out
        x = x + self.mlp_down(F.relu(self.mlp_up(x)))
        return x


class AdderModel(nn.Module):
    """Parameterized 1-layer adder model."""

    def __init__(self, d_model, n_heads, ff_dim, anchor_score=80.0):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.ff_dim = ff_dim
        self.anchor_score = anchor_score

        self.embed = nn.Embedding(VOCAB, d_model).double()
        self.block = TransformerBlock(d_model, n_heads, ff_dim).double()
        self.lm_head = nn.Linear(d_model, VOCAB, bias=True).double()

        self._build_mask()

    def _build_mask(self):
        """Fixed per-head attention routing (registered buffer, 0 params).

        Uses the same ALiBi prefix-sum mask structure as submission_1l.py.
        This is architecture, not learned -- CMA-ES only searches over weights.

        anchor_score controls the SEP token softmax anchor. 80.0 = original
        (requires e^80 in O_proj). 0.0 = gentle (no extreme values needed).
        """
        nh = self.n_heads
        anchor = self.anchor_score
        M = torch.full((nh, MAX_SEQ, MAX_SEQ), float('-inf'), dtype=torch.float64)

        for q in range(MAX_SEQ):
            if q < 21:
                for h in range(nh):
                    M[h, q, q] = 0.0
            else:
                k = q - 21

                # Last head: self-attend for residual cancellation
                M[nh - 1, q, q] = 0.0

                # Head 0: digit pair A_k + B_k (with softmax anchor)
                if nh >= 1:
                    if k < 10:
                        M[0, q, k] = 0.0
                        M[0, q, 11 + k] = 0.0
                    M[0, q, 10] = anchor

                # Head 1 (if exists): ALiBi prefix sum for carry
                if nh >= 2:
                    for j in range(min(k, 10)):
                        score = -(k - j) * math.log(10)
                        M[1, q, j] = score
                        M[1, q, 11 + j] = score
                    M[1, q, 10] = anchor

        self.register_buffer('M', M)

    def forward(self, x):
        B, L = x.shape
        h = self.embed(x)
        mask = self.M[:, :L, :L].unsqueeze(0)
        h = self.block(h, mask)
        return self.lm_head(h)


# ---------------------------------------------------------------------------
# Weight flatten/unflatten
# ---------------------------------------------------------------------------


def flatten_weights(model):
    """Extract all nn.Parameter values as a flat numpy vector."""
    parts = []
    for p in model.parameters():
        parts.append(p.data.cpu().numpy().ravel())
    return np.concatenate(parts)


def unflatten_weights(model, vec):
    """Write flat numpy vector back into model parameters."""
    total = sum(p.numel() for p in model.parameters())
    if len(vec) != total:
        raise ValueError(f"Vector length {len(vec)} != model params {total}")

    offset = 0
    with torch.no_grad():
        for p in model.parameters():
            n = p.numel()
            p.copy_(torch.from_numpy(vec[offset:offset + n].reshape(p.shape)).double())
            offset += n


# ---------------------------------------------------------------------------
# Model factory
# ---------------------------------------------------------------------------

ARCH_CONFIGS = {
    'a': {'d_model': 3, 'n_heads': 3, 'ff_dim': 4},               # 141 params, anchor=80
    'a0': {'d_model': 3, 'n_heads': 3, 'ff_dim': 4, 'anchor_score': 0.0},  # gentle mask
    'a5': {'d_model': 3, 'n_heads': 3, 'ff_dim': 4, 'anchor_score': 5.0},  # moderate anchor
    'b': {'d_model': 3, 'n_heads': 1, 'ff_dim': 4},
    'c': {'d_model': 2, 'n_heads': 1, 'ff_dim': 3},
}


def build_arch_a_model(warm_start=False):
    """Build Architecture A model (d=3, 3h, ff=4).

    Args:
        warm_start: If True, load analytical weights from submission_1l.py.
                    If False, use PyTorch default initialization.

    Returns:
        (model, metadata) tuple.
    """
    return build_model_for_arch('a', warm_start=warm_start)


def build_model_for_arch(arch, warm_start=False, warm_start_path=None):
    """Build model for given architecture key."""
    cfg = ARCH_CONFIGS[arch]
    model = AdderModel(**cfg)

    if warm_start:
        _load_warm_start(model, warm_start_path or 'submission_1l.py')

    meta = {
        'name': f'CMA-ES {arch.upper()} (d={cfg["d_model"]}, {cfg["n_heads"]}h, ff={cfg["ff_dim"]})',
        'author': 'Claude (CMA-ES)',
        'params': sum(p.numel() for p in model.parameters()),
        'architecture': f'1L decoder, d={cfg["d_model"]}, {cfg["n_heads"]}h, ff={cfg["ff_dim"]}',
        'tricks': ['CMA-ES evolutionary search'],
    }
    return model, meta


def _load_warm_start(model, path):
    """Load weights from a submission file's build_model() into our model."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("warmstart", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    source_model, _ = mod.build_model()

    # Copy parameters by position (architectures must match)
    source_params = list(source_model.parameters())
    target_params = list(model.parameters())

    if len(source_params) != len(target_params):
        raise ValueError(
            f"Warm start model has {len(source_params)} param tensors, "
            f"target has {len(target_params)}"
        )

    with torch.no_grad():
        for sp, tp in zip(source_params, target_params):
            if sp.shape != tp.shape:
                raise ValueError(f"Shape mismatch: source {sp.shape} vs target {tp.shape}")
            tp.copy_(sp.data)


# ---------------------------------------------------------------------------
# Fitness evaluation
# ---------------------------------------------------------------------------


def _add_with_model(model, a, b):
    """Run autoregressive addition inference (same logic as submission_1l.add)."""
    a_digits = [(a // (10**i)) % 10 for i in range(10)]
    b_digits = [(b // (10**i)) % 10 for i in range(10)]
    seq = a_digits + [SEP] + b_digits + [SEP]

    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(11):
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(x)
            next_token = logits[0, -1].argmax().item()
            seq.append(next_token)

    out_digits = seq[22:33]
    result = sum(d * (10**i) for i, d in enumerate(out_digits))
    return result


def evaluate_fitness(model, n_pairs=500, seed=42):
    """Evaluate model accuracy on random addition pairs.

    Args:
        model: AdderModel instance.
        n_pairs: Number of random pairs to test.
        seed: RNG seed for reproducible pair generation.

    Returns:
        Accuracy as float in [0, 1].
    """
    rng = random.Random(seed)
    correct = 0

    for _ in range(n_pairs):
        a = rng.randint(0, 9_999_999_999)
        b = rng.randint(0, 9_999_999_999)
        expected = a + b
        try:
            result = _add_with_model(model, a, b)
            if result == expected:
                correct += 1
        except Exception:
            pass  # count as wrong

    return correct / n_pairs


# ---------------------------------------------------------------------------
# Batched fitness for speed (processes multiple pairs per forward pass)
# ---------------------------------------------------------------------------


def evaluate_fitness_batched(model, n_pairs=500, seed=42):
    """Batched evaluation -- much faster than sequential for CMA-ES loop.

    Generates all pairs, runs them through the model in batches.
    """
    rng = random.Random(seed)
    pairs = [(rng.randint(0, 9_999_999_999), rng.randint(0, 9_999_999_999))
             for _ in range(n_pairs)]

    # Build input sequences
    seqs = []
    for a, b in pairs:
        a_digits = [(a // (10**i)) % 10 for i in range(10)]
        b_digits = [(b // (10**i)) % 10 for i in range(10)]
        seqs.append(a_digits + [SEP] + b_digits + [SEP])

    device = next(model.parameters()).device

    with torch.no_grad():
        for step in range(11):
            x = torch.tensor(seqs, dtype=torch.long, device=device)
            logits = model(x)
            next_tokens = logits[:, -1].argmax(dim=-1).tolist()
            for i, tok in enumerate(next_tokens):
                seqs[i].append(tok)

    correct = 0
    for i, (a, b) in enumerate(pairs):
        out_digits = seqs[i][22:33]
        result = sum(d * (10**j) for j, d in enumerate(out_digits))
        if result == a + b:
            correct += 1

    return correct / n_pairs


# ---------------------------------------------------------------------------
# Teacher-forced fitness (continuous signal for CMA-ES)
# ---------------------------------------------------------------------------


def evaluate_fitness_teacher_forced(model, n_pairs=500, seed=42, max_digits=10):
    """Teacher-forced cross-entropy loss -- gives continuous signal for CMA-ES.

    Instead of autoregressive generation (which gives all-or-nothing accuracy),
    this feeds the correct full sequence and measures how well the model predicts
    each output digit. Returns CE loss (lower = better, for CMA-ES minimization)
    and digit-level accuracy.

    Args:
        model: AdderModel instance.
        n_pairs: Number of pairs to evaluate.
        seed: RNG seed.
        max_digits: Max digits per operand (1-10). Supports curriculum learning.
                    With max_digits=1, operands are 0-9 (single digit addition).
                    With max_digits=10, operands are 0-9999999999 (full problem).
    """
    rng = random.Random(seed)
    max_val = 10**max_digits - 1

    # Build full teacher-forced sequences: input (22 tokens) + target output (11 tokens)
    seqs = []
    for _ in range(n_pairs):
        a = rng.randint(0, max_val)
        b = rng.randint(0, max_val)
        s = a + b

        a_digits = [(a // (10**i)) % 10 for i in range(10)]
        b_digits = [(b // (10**i)) % 10 for i in range(10)]
        s_digits = [(s // (10**i)) % 10 for i in range(11)]

        # Full sequence: a0..a9 SEP b0..b9 SEP s0..s10
        seq = a_digits + [SEP] + b_digits + [SEP] + s_digits
        seqs.append(seq)

    device = next(model.parameters()).device

    # Only evaluate active digit positions (0..max_digits inclusive for carry).
    # This prevents the model from getting "free" accuracy by predicting 0 for
    # unused high positions.
    n_active = min(max_digits + 1, 11)  # +1 for potential carry digit

    with torch.no_grad():
        x = torch.tensor(seqs, dtype=torch.long, device=device)  # (N, 33)
        logits = model(x)  # (N, 33, VOCAB)

        # Predictions at positions 21..21+n_active-1
        pred_logits = logits[:, 21:21 + n_active, :]  # (N, n_active, VOCAB)
        targets = x[:, 22:22 + n_active]  # (N, n_active)

        # Cross-entropy loss on active positions only
        ce_loss = F.cross_entropy(
            pred_logits.reshape(-1, VOCAB),
            targets.reshape(-1),
            reduction='mean'
        ).item()

        # Digit-level accuracy on active positions only
        pred_digits = pred_logits.argmax(dim=-1)  # (N, n_active)
        digit_acc = (pred_digits == targets).float().mean().item()

    return ce_loss, digit_acc


# ---------------------------------------------------------------------------
# CMA-ES search
# ---------------------------------------------------------------------------


def run_cmaes(arch='a', warm_start_path=None, sigma0=None, max_evals=50000,
              target_accuracy=0.995, verbose=True, zero_init=False,
              curriculum=True, pop_size=None):
    """Run CMA-ES search for optimal weights.

    Uses teacher-forced cross-entropy loss as the fitness function (gives
    continuous gradient signal even for random weights). Supports curriculum
    learning from 1-digit to 10-digit addition.

    Args:
        arch: Architecture key ('a', 'b', 'c').
        warm_start_path: Path to submission file for warm-start, or None for cold-start.
        sigma0: Initial step size. Default: 0.1 for warm-start, 1.0 for cold-start.
        max_evals: Maximum number of fitness evaluations.
        target_accuracy: Stop when this accuracy is reached on 2000 pairs.
        verbose: Print progress.
        zero_init: If True, start from zeros instead of PyTorch default init.
        curriculum: If True, start with 1-digit numbers and scale up.
        pop_size: CMA-ES population size. Default: 4 + 3*ln(N).

    Returns:
        (best_weights, best_accuracy, model) tuple.
    """
    warm = warm_start_path is not None
    if sigma0 is None:
        sigma0 = 0.1 if warm else 1.0

    # Curriculum: digit complexity schedule (generation -> max_digits)
    # Start simple, increase when loss plateaus
    if curriculum and not warm:
        digit_schedule = {0: 1, 200: 2, 500: 3, 1000: 5, 2000: 10}
    else:
        digit_schedule = {0: 10}

    # Build model and get initial weights
    model, _ = build_model_for_arch(arch, warm_start=warm, warm_start_path=warm_start_path)

    if zero_init and not warm:
        # Start from zeros -- less biased than random init
        x0 = np.zeros(sum(p.numel() for p in model.parameters()))
    else:
        x0 = flatten_weights(model)

    n_dims = len(x0)

    if pop_size is None:
        pop_size = 4 + int(3 * math.log(n_dims))

    if verbose:
        print(f"CMA-ES search: arch={arch}, dims={n_dims}, sigma0={sigma0}")
        print(f"  Warm start: {warm_start_path or 'None (cold start)'}")
        print(f"  Zero init: {zero_init}")
        print(f"  Curriculum: {curriculum}")
        print(f"  Pop size: {pop_size}")
        print(f"  Max evals: {max_evals}")
        print(f"  Target accuracy: {target_accuracy:.1%}")
        print()

    # CMA-ES options
    opts = cma.CMAOptions()
    opts['maxfevals'] = max_evals
    opts['popsize'] = pop_size
    # Disable premature stopping -- we need many generations for cold start
    opts['tolx'] = 0
    opts['tolfun'] = 0
    opts['tolstagnation'] = max_evals  # effectively disable
    opts['verb_disp'] = 0  # we do our own logging

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    # Evaluate initial weights
    best_weights = x0.copy()
    unflatten_weights(model, x0)
    init_loss, init_digit_acc = evaluate_fitness_teacher_forced(model, n_pairs=200, seed=9999, max_digits=1)
    best_loss = init_loss
    best_autoreg_acc = 0.0
    if verbose:
        print(f"  Initial loss (1-digit): {init_loss:.4f}, digit acc: {init_digit_acc:.4f}")

    gen = 0
    current_max_digits = min(digit_schedule.values())  # start at easiest
    n_pairs = 300  # fixed eval size
    start_time = time.time()

    while not es.stop():
        # Update digit complexity based on curriculum schedule
        scheduled_digits = current_max_digits
        for gen_threshold in sorted(digit_schedule.keys()):
            if gen >= gen_threshold:
                scheduled_digits = digit_schedule[gen_threshold]
        if scheduled_digits != current_max_digits:
            current_max_digits = scheduled_digits
            best_loss = float('inf')  # reset when difficulty increases
            if verbose:
                print(f"\n  === Curriculum: increasing to {current_max_digits}-digit numbers ===\n")

        # Rotating seed per generation
        gen_seed = 10000 + gen

        solutions = es.ask()

        # Evaluate fitness: teacher-forced CE loss (CMA-ES minimizes)
        fitnesses = []
        gen_digit_accs = []
        for sol in solutions:
            unflatten_weights(model, sol)
            loss, digit_acc = evaluate_fitness_teacher_forced(
                model, n_pairs=n_pairs, seed=gen_seed, max_digits=current_max_digits
            )
            fitnesses.append(loss)
            gen_digit_accs.append(digit_acc)

            if loss < best_loss:
                best_loss = loss
                best_weights = sol.copy()

        es.tell(solutions, fitnesses)

        # Progress logging
        if verbose and (gen % 10 == 0 or gen < 5):
            elapsed = time.time() - start_time
            mean_loss = np.mean(fitnesses)
            best_gen_loss = min(fitnesses)
            best_gen_dacc = max(gen_digit_accs)
            total_evals = es.result.evaluations
            print(
                f"  gen {gen:4d} | evals {total_evals:6d} | "
                f"loss {best_gen_loss:.4f} (mean {mean_loss:.4f}) | "
                f"digit_acc {best_gen_dacc:.3f} | "
                f"best_loss {best_loss:.4f} | sigma {es.sigma:.2e} | "
                f"digits {current_max_digits} | {elapsed:.1f}s"
            )

        # Periodically check autoregressive accuracy when digit acc is promising
        if gen % 100 == 0 and gen > 0 and max(gen_digit_accs) > 0.3:
            unflatten_weights(model, best_weights)
            autoreg_acc = evaluate_fitness_batched(model, n_pairs=500, seed=gen_seed)
            best_autoreg_acc = max(best_autoreg_acc, autoreg_acc)
            if verbose:
                print(f"    --> autoreg accuracy check: {autoreg_acc:.4f}")

            if autoreg_acc >= target_accuracy:
                if verbose:
                    print(f"\n  Target {target_accuracy:.1%} reached! Verifying on 2000 pairs...")
                verify_acc = evaluate_fitness_batched(model, n_pairs=2000, seed=99999)
                if verbose:
                    print(f"  Verification accuracy: {verify_acc:.4f}")
                if verify_acc >= target_accuracy:
                    best_autoreg_acc = verify_acc
                    if verbose:
                        print("  CONFIRMED. Stopping search.")
                    break

        gen += 1

    elapsed = time.time() - start_time

    # Final evaluation
    unflatten_weights(model, best_weights)
    final_loss, final_digit_acc = evaluate_fitness_teacher_forced(model, n_pairs=2000, seed=77777)
    final_autoreg_acc = evaluate_fitness_batched(model, n_pairs=2000, seed=77777)

    if verbose:
        print(f"\nSearch complete in {elapsed:.1f}s ({es.result.evaluations} evals, {gen} gens)")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Final loss (2000 pairs, 10-digit): {final_loss:.4f}")
        print(f"Final digit accuracy: {final_digit_acc:.4f}")
        print(f"Final autoregressive accuracy: {final_autoreg_acc:.4f}")

    return best_weights, final_autoreg_acc, model


# ---------------------------------------------------------------------------
# Intrinsic dimensionality CMA-ES
# ---------------------------------------------------------------------------


def run_cmaes_intrinsic(arch='a0', intrinsic_dim=40, sigma0=5.0, max_evals=100000,
                        target_accuracy=0.995, verbose=True, curriculum=True,
                        pop_size=None, adam_refine=False, adam_steps=50,
                        adam_lr=0.005):
    """CMA-ES in a low-dimensional random subspace of the full parameter space.

    Instead of searching all N=141 dims, searches k << N dims and projects
    to full space: theta = theta_0 + P @ z, where P is a random N x k matrix.

    This keeps CMA-ES in its sweet spot (k=30-50) while covering the full
    parameter space through the random projection.

    Args:
        arch: Architecture key.
        intrinsic_dim: Dimension of the search subspace (k).
        sigma0: Initial CMA-ES step size in the subspace.
        max_evals: Max fitness evaluations.
        target_accuracy: Stop when autoregressive accuracy exceeds this.
        verbose: Print progress.
        curriculum: Use digit curriculum.
        pop_size: CMA-ES population size.
        adam_refine: If True, apply Adam refinement after CMA-ES converges.
        adam_steps: Adam steps for final refinement.
        adam_lr: Adam LR for final refinement.
    """
    if curriculum:
        digit_schedule = {0: 1, 200: 2, 500: 3, 1000: 5, 2000: 10}
    else:
        digit_schedule = {0: 10}

    model, _ = build_model_for_arch(arch)
    n_full = sum(p.numel() for p in model.parameters())

    # Random projection matrix: (n_full, intrinsic_dim)
    # Use orthogonal initialization for better coverage
    rng = np.random.RandomState(42)
    P = rng.randn(n_full, intrinsic_dim)
    # Normalize columns to unit norm
    P /= np.linalg.norm(P, axis=0, keepdims=True)
    # Scale so that ||P @ z||_2 ~ ||z||_2 for typical z
    P *= np.sqrt(n_full / intrinsic_dim)

    # Base weights (zero)
    theta_0 = np.zeros(n_full)

    # Search variable z
    z0 = np.zeros(intrinsic_dim)

    if pop_size is None:
        pop_size = 4 + int(3 * math.log(intrinsic_dim))

    if verbose:
        print(f"CMA-ES intrinsic: arch={arch}, full_dims={n_full}, intrinsic_dims={intrinsic_dim}")
        print(f"  sigma0={sigma0}, pop_size={pop_size}")
        print(f"  Max evals: {max_evals}")
        print(f"  Curriculum: {curriculum}")
        print(f"  Adam refine after: {adam_refine} ({adam_steps} steps, lr={adam_lr})")
        print()

    opts = cma.CMAOptions()
    opts['maxfevals'] = max_evals
    opts['popsize'] = pop_size
    opts['tolx'] = 0
    opts['tolfun'] = 0
    opts['tolstagnation'] = max_evals
    opts['verb_disp'] = 0

    es = cma.CMAEvolutionStrategy(z0, sigma0, opts)

    best_z = z0.copy()
    best_weights = theta_0.copy()
    best_loss = float('inf')
    best_autoreg_acc = 0.0
    gen = 0
    current_max_digits = min(digit_schedule.values())
    n_pairs = 300
    start_time = time.time()

    # Initial evaluation
    unflatten_weights(model, theta_0)
    init_loss, init_dacc = evaluate_fitness_teacher_forced(model, n_pairs=200, seed=9999, max_digits=1)
    best_loss = init_loss
    if verbose:
        print(f"  Initial loss (1-digit): {init_loss:.4f}, digit_acc: {init_dacc:.4f}")

    while not es.stop():
        # Curriculum
        scheduled_digits = current_max_digits
        for gen_threshold in sorted(digit_schedule.keys()):
            if gen >= gen_threshold:
                scheduled_digits = digit_schedule[gen_threshold]
        if scheduled_digits != current_max_digits:
            current_max_digits = scheduled_digits
            best_loss = float('inf')
            if verbose:
                print(f"\n  === Curriculum: {current_max_digits}-digit numbers ===\n")

        gen_seed = 10000 + gen
        z_solutions = es.ask()

        # Project to full parameter space and evaluate
        fitnesses = []
        gen_digit_accs = []
        LOSS_CLIP = 10.0  # Clip extreme losses to preserve ranking signal
        for z in z_solutions:
            theta = theta_0 + P @ z
            unflatten_weights(model, theta)
            loss, dacc = evaluate_fitness_teacher_forced(
                model, n_pairs=n_pairs, seed=gen_seed, max_digits=current_max_digits
            )
            fitnesses.append(min(loss, LOSS_CLIP))
            gen_digit_accs.append(dacc)

            if loss < best_loss:
                best_loss = loss
                best_z = z.copy()
                best_weights = theta.copy()

        es.tell(z_solutions, fitnesses)

        # Progress logging
        if verbose and (gen % 10 == 0 or gen < 5):
            elapsed = time.time() - start_time
            mean_loss = np.mean(fitnesses)
            best_gen_loss = min(fitnesses)
            best_gen_dacc = max(gen_digit_accs)
            total_evals = es.result.evaluations
            print(
                f"  gen {gen:4d} | evals {total_evals:6d} | "
                f"loss {best_gen_loss:.4f} (mean {mean_loss:.4f}) | "
                f"digit_acc {best_gen_dacc:.3f} | "
                f"best_loss {best_loss:.4f} | sigma {es.sigma:.2e} | "
                f"digits {current_max_digits} | {elapsed:.1f}s"
            )

        # Autoregressive check
        if gen % 100 == 0 and gen > 0 and max(gen_digit_accs) > 0.3:
            unflatten_weights(model, best_weights)
            autoreg_acc = evaluate_fitness_batched(model, n_pairs=500, seed=gen_seed)
            best_autoreg_acc = max(best_autoreg_acc, autoreg_acc)
            if verbose:
                print(f"    --> autoreg accuracy: {autoreg_acc:.4f}")
            if autoreg_acc >= target_accuracy:
                verify_acc = evaluate_fitness_batched(model, n_pairs=2000, seed=99999)
                if verify_acc >= target_accuracy:
                    best_autoreg_acc = verify_acc
                    if verbose:
                        print(f"  CONFIRMED at {verify_acc:.4f}. Stopping.")
                    break

        gen += 1

    elapsed_cmaes = time.time() - start_time

    # Optional Adam refinement in full parameter space
    if adam_refine:
        if verbose:
            print(f"\nApplying Adam refinement ({adam_steps} steps, lr={adam_lr})...")
        unflatten_weights(model, best_weights)
        refined_loss, refined_dacc = adam_local_search(
            model, n_steps=adam_steps, lr=adam_lr,
            n_pairs=500, max_digits=10, seed_base=80000
        )
        best_weights = flatten_weights(model)
        if verbose:
            print(f"  After Adam: loss={refined_loss:.4f}, digit_acc={refined_dacc:.4f}")

    elapsed = time.time() - start_time
    unflatten_weights(model, best_weights)
    final_loss, final_dacc = evaluate_fitness_teacher_forced(model, n_pairs=2000, seed=77777)
    final_autoreg = evaluate_fitness_batched(model, n_pairs=2000, seed=77777)

    if verbose:
        print(f"\nSearch complete in {elapsed:.1f}s ({es.result.evaluations} evals, {gen} gens)")
        print(f"  CMA-ES phase: {elapsed_cmaes:.1f}s")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Final loss (2000 pairs, 10-digit): {final_loss:.4f}")
        print(f"Final digit accuracy: {final_dacc:.4f}")
        print(f"Final autoregressive accuracy: {final_autoreg:.4f}")

    return best_weights, final_autoreg, model


# ---------------------------------------------------------------------------
# Local gradient refinement (Adam)
# ---------------------------------------------------------------------------


def _teacher_forced_loss_grad(model, n_pairs=200, seed=42, max_digits=10):
    """Compute teacher-forced CE loss WITH gradient (for Adam local search)."""
    rng = random.Random(seed)
    max_val = 10**max_digits - 1
    n_active = min(max_digits + 1, 11)

    seqs = []
    for _ in range(n_pairs):
        a = rng.randint(0, max_val)
        b = rng.randint(0, max_val)
        s = a + b
        a_digits = [(a // (10**i)) % 10 for i in range(10)]
        b_digits = [(b // (10**i)) % 10 for i in range(10)]
        s_digits = [(s // (10**i)) % 10 for i in range(11)]
        seqs.append(a_digits + [SEP] + b_digits + [SEP] + s_digits)

    device = next(model.parameters()).device
    x = torch.tensor(seqs, dtype=torch.long, device=device)
    logits = model(x)

    pred_logits = logits[:, 21:21 + n_active, :]
    targets = x[:, 22:22 + n_active]

    loss = F.cross_entropy(
        pred_logits.reshape(-1, VOCAB),
        targets.reshape(-1),
        reduction='mean'
    )
    return loss


def adam_local_search(model, n_steps=50, lr=0.001, n_pairs=200, max_digits=10,
                      seed_base=50000):
    """Run a few steps of Adam to locally refine model weights.

    Returns:
        (final_loss, final_digit_acc) tuple.
    """
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    for step in range(n_steps):
        optimizer.zero_grad()
        loss = _teacher_forced_loss_grad(model, n_pairs=n_pairs,
                                          seed=seed_base + step,
                                          max_digits=max_digits)
        loss.backward()
        optimizer.step()

    # Evaluate final state
    with torch.no_grad():
        final_loss, final_dacc = evaluate_fitness_teacher_forced(
            model, n_pairs=n_pairs, seed=seed_base + n_steps, max_digits=max_digits
        )
    return final_loss, final_dacc


# ---------------------------------------------------------------------------
# Memetic CMA-ES (CMA-ES + Adam local search)
# ---------------------------------------------------------------------------


def run_memetic(arch='a0', sigma0=5.0, max_evals=50000, target_accuracy=0.995,
                verbose=True, adam_steps=20, adam_lr=0.01, curriculum=True,
                pop_size=None):
    """Memetic algorithm: CMA-ES global search + Adam local refinement.

    Each generation:
    1. CMA-ES proposes candidate weight vectors
    2. Each candidate gets N steps of Adam to locally refine
    3. Refined candidates are evaluated and returned to CMA-ES

    This combines CMA-ES's ability to explore broadly with Adam's ability
    to find nearby local optima using gradient information.

    Args:
        arch: Architecture key.
        sigma0: Initial CMA-ES step size.
        max_evals: Max fitness evaluations (each Adam-refined candidate counts as 1).
        target_accuracy: Stop when autoregressive accuracy exceeds this.
        verbose: Print progress.
        adam_steps: Number of Adam steps per candidate.
        adam_lr: Adam learning rate for local search.
        curriculum: Use digit curriculum.
        pop_size: CMA-ES population size.
    """
    if curriculum:
        digit_schedule = {0: 1, 100: 2, 300: 3, 600: 5, 1000: 10}
    else:
        digit_schedule = {0: 10}

    model, _ = build_model_for_arch(arch)
    x0 = np.zeros(sum(p.numel() for p in model.parameters()))
    n_dims = len(x0)

    if pop_size is None:
        pop_size = 4 + int(3 * math.log(n_dims))

    if verbose:
        print(f"Memetic CMA-ES+Adam: arch={arch}, dims={n_dims}")
        print(f"  sigma0={sigma0}, pop_size={pop_size}")
        print(f"  Adam: {adam_steps} steps, lr={adam_lr}")
        print(f"  Max evals: {max_evals}")
        print(f"  Curriculum: {curriculum}")
        print()

    opts = cma.CMAOptions()
    opts['maxfevals'] = max_evals
    opts['popsize'] = pop_size
    opts['tolx'] = 0
    opts['tolfun'] = 0
    opts['tolstagnation'] = max_evals
    opts['verb_disp'] = 0

    es = cma.CMAEvolutionStrategy(x0, sigma0, opts)

    best_weights = x0.copy()
    best_loss = float('inf')
    best_autoreg_acc = 0.0
    gen = 0
    current_max_digits = min(digit_schedule.values())
    start_time = time.time()

    # Initial evaluation
    unflatten_weights(model, x0)
    init_loss, init_dacc = evaluate_fitness_teacher_forced(model, n_pairs=300, seed=9999, max_digits=1)
    if verbose:
        print(f"  Initial loss (1-digit): {init_loss:.4f}, digit_acc: {init_dacc:.4f}")

    while not es.stop():
        # Curriculum
        scheduled_digits = current_max_digits
        for gen_threshold in sorted(digit_schedule.keys()):
            if gen >= gen_threshold:
                scheduled_digits = digit_schedule[gen_threshold]
        if scheduled_digits != current_max_digits:
            current_max_digits = scheduled_digits
            best_loss = float('inf')
            if verbose:
                print(f"\n  === Curriculum: {current_max_digits}-digit numbers ===\n")

        gen_seed = 10000 + gen
        solutions = es.ask()

        fitnesses = []
        refined_solutions = []
        gen_digit_accs = []

        for sol in solutions:
            # Load candidate weights
            unflatten_weights(model, sol)

            # Adam local refinement
            refined_loss, refined_dacc = adam_local_search(
                model, n_steps=adam_steps, lr=adam_lr,
                n_pairs=200, max_digits=current_max_digits,
                seed_base=gen_seed * 1000
            )

            # Extract refined weights
            refined_vec = flatten_weights(model)
            refined_solutions.append(refined_vec)

            # Evaluate refined model (separate seed from Adam)
            eval_loss, eval_dacc = evaluate_fitness_teacher_forced(
                model, n_pairs=300, seed=gen_seed, max_digits=current_max_digits
            )
            fitnesses.append(eval_loss)
            gen_digit_accs.append(eval_dacc)

            if eval_loss < best_loss:
                best_loss = eval_loss
                best_weights = refined_vec.copy()

        # Tell CMA-ES about the REFINED solutions (Lamarckian evolution)
        es.tell(refined_solutions, fitnesses)

        # Progress logging
        if verbose and (gen % 5 == 0 or gen < 3):
            elapsed = time.time() - start_time
            mean_loss = np.mean(fitnesses)
            best_gen_loss = min(fitnesses)
            best_gen_dacc = max(gen_digit_accs)
            total_evals = es.result.evaluations
            print(
                f"  gen {gen:4d} | evals {total_evals:6d} | "
                f"loss {best_gen_loss:.4f} (mean {mean_loss:.4f}) | "
                f"digit_acc {best_gen_dacc:.3f} | "
                f"best_loss {best_loss:.4f} | sigma {es.sigma:.2e} | "
                f"digits {current_max_digits} | {elapsed:.1f}s"
            )

        # Autoregressive check
        if gen % 50 == 0 and gen > 0 and max(gen_digit_accs) > 0.5:
            unflatten_weights(model, best_weights)
            autoreg_acc = evaluate_fitness_batched(model, n_pairs=500, seed=gen_seed)
            best_autoreg_acc = max(best_autoreg_acc, autoreg_acc)
            if verbose:
                print(f"    --> autoreg accuracy: {autoreg_acc:.4f}")
            if autoreg_acc >= target_accuracy:
                verify_acc = evaluate_fitness_batched(model, n_pairs=2000, seed=99999)
                if verify_acc >= target_accuracy:
                    best_autoreg_acc = verify_acc
                    if verbose:
                        print(f"  CONFIRMED at {verify_acc:.4f}. Stopping.")
                    break

        gen += 1

    elapsed = time.time() - start_time
    unflatten_weights(model, best_weights)
    final_loss, final_dacc = evaluate_fitness_teacher_forced(model, n_pairs=2000, seed=77777)
    final_autoreg = evaluate_fitness_batched(model, n_pairs=2000, seed=77777)

    if verbose:
        print(f"\nSearch complete in {elapsed:.1f}s ({es.result.evaluations} evals, {gen} gens)")
        print(f"Best loss: {best_loss:.4f}")
        print(f"Final loss (2000 pairs, 10-digit): {final_loss:.4f}")
        print(f"Final digit accuracy: {final_dacc:.4f}")
        print(f"Final autoregressive accuracy: {final_autoreg:.4f}")

    return best_weights, final_autoreg, model


# ---------------------------------------------------------------------------
# Save / load checkpoints
# ---------------------------------------------------------------------------


def save_checkpoint(weights, arch, accuracy, path):
    """Save best weights and metadata to JSON + npy files.

    Args:
        path: Base path without extension. Creates {path}.json and {path}.npy.
    """
    # Strip any extension to get clean base
    base = path
    for ext in ('.json', '.npy'):
        if base.endswith(ext):
            base = base[:-len(ext)]

    npy_path = base + '.npy'
    meta_path = base + '.json'

    np.save(npy_path, weights)
    with open(meta_path, 'w') as f:
        json.dump({'arch': arch, 'accuracy': accuracy, 'npy_path': npy_path}, f)
    print(f"Saved checkpoint: {meta_path} + {npy_path} (accuracy={accuracy:.4f})")


def load_checkpoint(path):
    """Load weights and metadata from checkpoint files."""
    meta_path = path if path.endswith('.json') else path + '.json'
    with open(meta_path) as f:
        meta = json.load(f)
    weights = np.load(meta['npy_path'])
    return weights, meta['arch'], meta['accuracy']


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(description="CMA-ES search for transformer adder weights")
    parser.add_argument('--arch', choices=list(ARCH_CONFIGS.keys()), default='a',
                        help='Architecture: a=d3/3h/ff4, a0=gentle(anchor=0), a5=anchor5, b=d3/1h/ff4, c=d2/1h/ff3')
    parser.add_argument('--warm-start', type=str, default=None,
                        help='Path to submission file for warm-start initialization')
    parser.add_argument('--sigma0', type=float, default=None,
                        help='Initial CMA-ES step size (default: 0.1 warm, 1.0 cold)')
    parser.add_argument('--max-evals', type=int, default=50000,
                        help='Maximum fitness evaluations')
    parser.add_argument('--target', type=float, default=0.995,
                        help='Target accuracy for early stopping')
    parser.add_argument('--zero-init', action='store_true',
                        help='Initialize weights from zeros instead of random')
    parser.add_argument('--no-curriculum', action='store_true',
                        help='Disable curriculum learning (start with 10-digit)')
    parser.add_argument('--pop-size', type=int, default=None,
                        help='CMA-ES population size (default: 4 + 3*ln(N))')
    parser.add_argument('--memetic', action='store_true',
                        help='Use memetic mode (CMA-ES + Adam local search)')
    parser.add_argument('--intrinsic', type=int, default=None,
                        help='Use intrinsic dim CMA-ES with given subspace dimension (e.g., 40)')
    parser.add_argument('--adam-refine', action='store_true',
                        help='Apply Adam refinement after CMA-ES (for intrinsic mode)')
    parser.add_argument('--adam-steps', type=int, default=50,
                        help='Adam steps for refinement')
    parser.add_argument('--adam-lr', type=float, default=0.005,
                        help='Adam learning rate for refinement')
    parser.add_argument('--save', type=str, default=None,
                        help='Path to save best weights (.json)')
    args = parser.parse_args()

    if args.intrinsic:
        best_weights, final_acc, model = run_cmaes_intrinsic(
            arch=args.arch,
            intrinsic_dim=args.intrinsic,
            sigma0=args.sigma0 or 5.0,
            max_evals=args.max_evals,
            target_accuracy=args.target,
            curriculum=not args.no_curriculum,
            pop_size=args.pop_size,
            adam_refine=args.adam_refine,
            adam_steps=args.adam_steps,
            adam_lr=args.adam_lr,
        )
    elif args.memetic:
        best_weights, final_acc, model = run_memetic(
            arch=args.arch,
            sigma0=args.sigma0 or 5.0,
            max_evals=args.max_evals,
            target_accuracy=args.target,
            adam_steps=args.adam_steps,
            adam_lr=args.adam_lr,
            curriculum=not args.no_curriculum,
            pop_size=args.pop_size,
        )
    else:
        best_weights, final_acc, model = run_cmaes(
            arch=args.arch,
            warm_start_path=args.warm_start,
            sigma0=args.sigma0,
            max_evals=args.max_evals,
            target_accuracy=args.target,
            zero_init=args.zero_init,
            curriculum=not args.no_curriculum,
            pop_size=args.pop_size,
        )

    save_path = args.save or f"cmaes_best_{args.arch}.json"
    save_checkpoint(best_weights, args.arch, final_acc, save_path)


if __name__ == '__main__':
    main()
