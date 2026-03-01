# AGENTS.md

Last updated: 2026-02-28

Co-authored by Claude (Opus 4), Gemini (2.5 Pro), and [fblissjr](https://github.com/fblissjr).

This document describes how the submissions in this repo were developed, what tools and models were involved, what data was used, and how to reproduce the results.

## Tools and Models Used

- **Claude Code** (CLI, powered by Claude Opus 4): Primary development agent. Wrote all training scripts, model architectures, hand-coded weight derivations, and experiment orchestration. All code in this repo was written or modified through Claude Code sessions.
- **Gemini 2.5 Pro**: Independently developed a hand-coded solution (now in `archive/gemini_submission.py`) and provided the key insight that carry detection could be done in attention rather than the MLP, which led to the 1-layer architecture.
- **Human (fblissjr)**: Direction, architecture decisions, experiment prioritization, sanity checking. All training was run locally on Apple Silicon (MPS backend).

## Submissions Overview

| File | Category | Params | Accuracy | Method |
|---|---|---|---|---|
| `submission_trained.py` | Trained | 162 | 100% (local) | Fixed mask + AdamW |
| `submission_1l.py` | Hand-coded | 33 unique | 100% (local) | Analytical weights |
| `submission.py` | Hand-coded | 249 | 100% (local) | Analytical weights (2L) |

## Trained Submission (162 params)

### Architecture

```
1-layer decoder transformer
  d_model = 3
  n_heads = 3 (d_head = 1)
  d_ff = 6
  vocab = 12 (digits 0-9, SEP, SEP2)
  sequence = 33 tokens (10 + 1 + 10 + 1 + 11)
  input format: reversed LSB-first digits
```

Trainable parameter breakdown:

| Component | Shape | Count |
|---|---|---|
| Embedding | (12, 3) | 36 |
| Q projection | (3, 3) | 9 |
| K projection | (3, 3) | 9 |
| V projection | (3, 3) | 9 |
| O projection | (3, 3) | 9 |
| MLP up + bias | (6, 3) + (6,) | 24 |
| MLP down | (3, 6) | 18 |
| LM head + bias | (12, 3) + (12,) | 48 |
| **Total** | | **162** |

Fixed (non-learnable) components:
- **Attention mask buffer** (3, 35, 35): Pre-computed routing mask. Not a parameter.
- **RMSNorm layers** (3x): Parameterless `x * rsqrt(mean(x^2) + eps)`.

### The Fixed Mask

The key idea: use the attention routing pattern from our hand-coded solution as a fixed mask, then let gradient descent learn the value computation.

The mask is a (3, 35, 35) tensor of attention score biases. It encodes:

**Head 0 (digit pair extraction)**: For output position k, allows attention to input positions A_k and B_k (the k-th digits of each operand) plus the separator. All other positions are masked to -inf.

**Head 1 (carry prefix sum)**: For output position k, allows attention to all preceding digit positions (A_0..A_{k-1}, B_0..B_{k-1}) with scores set to `-(k-j) * log(10)`. After softmax, this produces weights proportional to `10^(j-k)`, creating a base-10 exponential prefix sum. The floor of this sum equals the carry into position k.

**Head 2 (residual management)**: Each position attends only to itself. This head learns to cancel or modify the residual connection.

**Prompt positions** (0-20): All heads self-attend only (identity passthrough).

The mask is derived from `build_fixed_mask()` in `train_adder.py:77-123`. The same routing logic is used in the hand-coded solution `submission_1l.py`.

### Training Data

No external dataset. All training data generated on the fly:

- Two random integers sampled uniformly from [0, 10^d) where d is uniformly random in [1, 10]
- 20% of samples use all-9s operands (e.g., 9999999999) to increase carry chain exposure
- Formatted as reversed LSB-first digits: `a0 a1 ... a9 SEP b0 b1 ... b9 SEP2 c0 c1 ... c10`
- Batch size: 512 pairs per step
- New random batch every step (no epoch-based training)

### Training Procedure

**Phase 1: Exploration (50K steps)**

```
Optimizer: AdamW
  learning_rate: 0.01
  weight_decay: 0.01
  betas: (0.9, 0.999)  # PyTorch defaults
LR schedule: Cosine annealing from 0.01 to 0.0001 over 500K steps
Gradient clipping: max_norm = 1.0
Loss: Cross-entropy on answer positions only (indices 21-31 of the sequence)
Batch size: 512
Device: Apple Silicon MPS
```

Warm initialization:
- Embedding: digit d maps to [d, 0, 0]; separators map to [0, 0, 0]
- V projection: [[1,0,0], [1,0,0], [-1,0,0]] (extract digit value per head)
- Q and K projections: random normal with std=0.01 (near zero so fixed mask dominates)
- All other weights: PyTorch default init

Progress during Phase 1:

| Step | Teacher-Forced Acc | Autoregressive Acc (2K pairs) |
|---|---|---|
| 5,000 | 45% | -- |
| 10,000 | 67% | 46% |
| 20,000 | 95% | 90.6% |
| 30,000 | 97% | 98.3% |
| 50,000 | 93% | 98.6% (best checkpoint saved) |

The model oscillated between 94-98.6% autoregressive during Phase 1. Teacher-forced accuracy hit 100% in bursts but wasn't stable.

**Phase 2: Stabilization (5K steps from best Phase 1 checkpoint)**

```
Optimizer: AdamW (same settings)
  learning_rate: 0.001
LR schedule: Cosine annealing from 0.001 to 0.00001 over 200K steps
Starting from: best Phase 1 checkpoint (98.6% autoregressive)
```

| Step | Teacher-Forced Acc | Autoregressive Acc (2K pairs) |
|---|---|---|
| 1,000 | 100% | -- |
| 5,000 | 100% | 100% (2000/2000) |

The lower learning rate immediately stabilized the model. It went from oscillating in the 94-99% range to solid 100%.

**Total wall clock time**: ~12 minutes on Apple MPS.

### Verification

```
$ python verify.py submission_trained.py
Model: 162-Parameter Trained Adder
Author: fblissjr
Parameters (unique): 162
Architecture: 1L d=3 3h ff=6 reversed-digits
Tricks: Reversed LSB-First, Teacher Forcing, RMSNorm, Fixed Mask (hand-coded routing)

Results: 10010/10010 correct (100.00%)
Time: 21.3s (469 additions/sec)
Status: QUALIFIED (threshold: 99%)
```

### Reproducing the Result

```bash
# Install dependencies
uv sync

# Phase 1: Exploration
uv run python train_continue.py \
  --config mask-w6 \
  --lr 0.01 \
  --max-steps 100000 \
  --score-interval 10000 \
  --score-pairs 2000

# Phase 2: Stabilization from best checkpoint
uv run python train_continue.py \
  --config mask-w6 \
  --checkpoint trained_mask-w6_best.pt \
  --lr 0.001 \
  --max-steps 50000 \
  --score-interval 5000 \
  --score-pairs 2000

# Verify
uv run python verify.py submission_trained_mask-w6.py
```

Phase 1 saves the best checkpoint as `trained_mask-w6_best.pt` in the working directory. Phase 2 loads from this path.

Due to random initialization, exact weight values will differ across runs. The approach has converged reliably in our testing (W6 converged on first attempt, W12 also converged independently), but the specific checkpoint that achieves 100% may require slightly different step counts.

### Other Trained Experiments

We trained several model sizes. Summary of all runs:

| Config | Params | d_ff | Best Autoreg | Steps | Outcome |
|---|---|---|---|---|---|
| mask-large (d=6) | 456 | 12 | 99.72% (verify.py) | 200K + 15K finetune | QUALIFIED |
| mask-w12 | 204 | 12 | 100% (verify.py) | 30K | QUALIFIED |
| mask-w6 | 162 | 6 | 100% (verify.py) | 50K + 5K finetune | QUALIFIED |
| mask-ff16 (cold, no warm init) | 232 | 16 | 81.0% | 65K (killed) | Still climbing |
| mask-w16 (warm) | 232 | 16 | 0% at 30K | 35K (killed) | Init-dependent failure |
| mask-w10 (warm) | 190 | 10 | 0% at 20K | 30K (killed) | Init-dependent failure |
| mask-w8 (warm) | 176 | 8 | 0% at 20K | 30K (killed) | Init-dependent failure |
| mask-w5 | 155 | 5 | 0% at 10K | 20K (killed) | Too early to tell |
| mask-w4 | 148 | 4 | 0% at 10K | 20K (killed) | Too early to tell |

Observations:
- W6 (162p) converged while W8/W10 (176p/190p) did not. Random init sensitivity is high with d=3. The smaller MLP apparently has a simpler optimization landscape.
- Cold-start (no warm init, no mask priors) models learn more slowly but the ff16 cold run was still climbing at 81% when killed. Given enough time, these might also converge.
- The two-phase training pattern (high LR exploration + low LR stabilization) was critical. The original 456p model sat at 84% until we dropped LR from 0.003 to 0.001.

### Failed Approaches (before fixed mask)

- **CMA-ES evolutionary search**: 141-dim search space, couldn't find solution from random init. Cold start produced 0% accuracy after 50K evaluations.
- **Standard SGD/Adam on 777p model**: d=7, 1h, ff=14. Peaked at 2.5% after 107K steps. Missing rank-3 factorization and needed much longer training.
- **ALiBi-only training** (no fixed mask): Various configs (d=3-6, 1-3 heads). Best reached ~35% teacher-forced. The model couldn't learn the routing pattern from data alone in reasonable time.
- **Sinusoidal PE training**: Standard causal attention with sinusoidal PE. Similar ceiling to ALiBi-only.

## Hand-Coded Submissions

### 1-Layer (33 unique params)

The key insight (contributed by Gemini's v2 solution): carry detection can be done in attention using ALiBi prefix sums, eliminating the need for a second layer.

Architecture: 1L decoder, d=3, 3 heads (d_head=1), ff=4.

Techniques:
- **ALiBi prefix sum**: Head 1 attention scores -(k-j)*log(10). Softmax produces weights proportional to 10^(j-k). The prefix sum `S_k = sum((A_j + B_j) * 10^(j-k))` encodes carry as `floor(S_k)`.
- **e^80 softmax anchoring**: Separator token gets attention score 80.0. O-projection multiplies by e^80 to recover exact scalar values from softmax output. Error ~3.6e-35 vs thresholds at 1e-11.
- **Residual cancellation**: Head 2 (V = -token_value, self-attend) cancels the residual connection so the MLP receives clean attention output.
- **2-hinge ReLU**: `ReLU(x + eps1) - ReLU(x + eps2)` with epsilon pair gives exact 0/1 step function. Used for carry detection and mod-10 overflow.
- **Parabolic LM head**: `logit[c] = 2c*x - c^2 = -(c-x)^2 + x^2`. Peaks at c = x. Argmax selects the correct digit.

33 unique parameter values, 141 total elements (many weights are reused across positions).

### 2-Layer (249 params)

Original approach before the carry-in-attention insight. Uses two layers because carry detection requires comparing previous output to previous pair sum, which needs one layer to compute the comparison and another to apply mod-10.

Architecture: 2L decoder, d=5, 2 heads (d_head=2), ff=4 per layer.

## File Index

| File | Description |
|---|---|
| `submission_trained.py` | 162-param trained submission (primary) |
| `submission_trained_mask-w6.py` | Same as above (named by config) |
| `submission_trained_mask-w12.py` | 204-param trained backup |
| `submission_trained_mask-large.py` | 456-param trained proof of concept |
| `submission_1l.py` | 33-param hand-coded 1L |
| `submission.py` | 249-param hand-coded 2L |
| `train_adder.py` | Training script (all configs, model, mask, export) |
| `train_continue.py` | Continuation training with autoregressive eval |
| `inspect_checkpoint.py` | Inspect checkpoint weights, stats, and values |
| `verify.py` | Official test harness (upstream) |
| `test_dataset.json` | 10,010 test cases (upstream) |
| `reports/trained_submission.md` | Detailed training report |
| `reports/claude_approach.md` | Hand-coded approach report |
| `reports/gemini_approach.md` | Gemini approach analysis |
| `reports/comparison_and_postmortem.md` | Comparison and lessons learned |
| `CHANGELOG.md` | Version history |
| `checkpoints/` | Training checkpoints (.pt files, gitignored) |
| `archive/` | Scratch files and old experiments (gitignored) |

## Checkpoint Files

All checkpoints are PyTorch `state_dict` files saved with `torch.save()`. Each contains the model's learnable parameters plus the fixed attention mask buffer.

Inspect any checkpoint:
```bash
uv run python inspect_checkpoint.py checkpoints/trained_mask-w6_best.pt
uv run python inspect_checkpoint.py checkpoints/trained_mask-w6_best.pt --verbose  # print all weight values
uv run python inspect_checkpoint.py --all  # inspect every checkpoint
```

Load and evaluate a checkpoint:
```bash
uv run python -c "
import torch
from train_adder import TrainableAdder, CONFIGS

cfg = CONFIGS['mask-w6']
model = TrainableAdder(**cfg)
model.load_state_dict(torch.load('checkpoints/trained_mask-w6_best.pt', weights_only=True))
print('Loaded:', sum(p.numel() for p in model.parameters()), 'params')
"
```

| Checkpoint | Config | Learnable Params | Best Autoreg | Description |
|---|---|---|---|---|
| `trained_mask-w6_best.pt` | mask-w6 | 162 | 100% (verify.py) | Primary. Phase 1 best (98.6%), then stabilized to 100% in Phase 2. d=3, ff=6, warm init. |
| `trained_mask-w12_best.pt` | mask-w12 | 204 | 100% (verify.py) | Backup. Converged in 30K steps single phase. d=3, ff=12, warm init. |
| `trained_mask-large_best.pt` | mask-large | 456 | 99.8% (2K pairs) | Proof of concept. d=6, ff=12, fine-tuned from mask-large.pt. |
| `trained_mask-large.pt` | mask-large | 456 | 84.4% (verify.py) | Early checkpoint before fine-tuning. Shows pre-stabilization state. |
| `trained_mask-ff16_best.pt` | mask-ff16 | 232 | 81.0% (2K pairs) | Cold start (no warm init), killed at 65K steps while still improving. |
| `trained_mask-warm8.pt` | mask-warm8 | 176 | 69% (TF level 10) | From previous session. d=3, ff=8, warm init, curriculum training. |
| `trained_mask-none.pt` | mask-none | 148 | 19% (autoreg) | Early fixed-mask experiment. d=3, ff=4, no warm init, curriculum. |
| `trained_steep-3h.pt` | steep-3h | 148 | <5% | ALiBi-only experiment (no fixed mask). Failed approach. |
| `trained_steep-med.pt` | steep-med | 144 | <5% | ALiBi-only, d=4 2h. Failed approach. |

Each checkpoint's `state_dict` contains these keys:
- `fixed_mask` or `alibi`: (n_heads, 35, 35) attention bias buffer (non-learnable)
- `embed.weight`: (12, d) token embeddings
- `q_proj.weight`, `k_proj.weight`, `v_proj.weight`, `o_proj.weight`: (d, d) attention projections
- `mlp_up.weight` + `mlp_up.bias`: (d_ff, d) + (d_ff,) MLP up-projection
- `mlp_down.weight`: (d, d_ff) MLP down-projection
- `lm_head.weight` + `lm_head.bias`: (12, d) + (12,) output logit projection

## Dependencies

```
python >= 3.11
torch >= 2.0
cma (pycma) -- only needed for CMA-ES experiments
```

Install with: `uv sync`
