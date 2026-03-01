# Trained Submission: Fixed-Mask Hybrid Approach

Last updated: 2026-02-28

## Summary

**162 parameters, 100.00% accuracy on 10,010 test cases.**

A 1-layer transformer that adds two 10-digit numbers, trained via AdamW with a novel hybrid approach: the attention routing mask is fixed (hand-coded, 0 learnable parameters) while the value/output projections, MLP, and LM head are learned from data.

## Architecture

```
1-layer transformer, reversed LSB-first digits
  d_model = 3
  n_heads = 3 (d_head = 1)
  d_ff = 6 (MLP hidden dim)
  vocab = 12 (digits 0-9, SEP, SEP2)
  max_seq = 33 (10 + 1 + 10 + 1 + 11)
```

### Trainable Parameters (162 total)

| Component | Shape | Parameters |
|---|---|---|
| Embedding | (12, 3) | 36 |
| Q projection | (3, 3) | 9 |
| K projection | (3, 3) | 9 |
| V projection | (3, 3) | 9 |
| O projection | (3, 3) | 9 |
| MLP up | (6, 3) + bias (6,) | 24 |
| MLP down | (3, 6) | 18 |
| LM head | (12, 3) + bias (12,) | 48 |
| **Total** | | **162** |

### Fixed Components (0 learnable parameters)

- **Attention mask**: Pre-computed 3-head routing mask encoding which positions each output digit should attend to. Derived from the analytical hand-coded solution.
- **RMSNorm**: 3 normalization layers (parameterless RMSNorm variant).

### Fixed Mask Design

The attention mask encodes the structure of addition:

**Head 0 (Digit Pair Extraction)**: Output position k attends to input positions A_k and B_k (the k-th digits of both operands), plus the separator token.

**Head 1 (Carry Prefix Sum)**: Output position k attends to all preceding digit positions (A_0..A_{k-1}, B_0..B_{k-1}) with ALiBi-style slopes of -log(10) per position distance. This creates a weighted prefix sum that encodes carry information: the softmax output approximates `sum((A_j + B_j) * 10^(j-k))`, which when floored gives the carry into position k.

**Head 2 (Residual Management)**: Each position self-attends only. Combined with appropriate V-projection, this allows the model to manage the residual connection.

**Prompt positions** (indices 0-20): All heads self-attend only (identity behavior).

This mask structure was derived from analyzing our hand-coded 1-layer solution (`submission_1l.py`), which achieves 100% accuracy with 33 unique parameters using the same routing pattern.

## Training Details

### Data

- **Training data**: Randomly generated addition pairs (no fixed dataset)
- **Format**: Reversed LSB-first digits. Example: `1234567890 + 9876543210` becomes `[0,9,8,7,6,5,4,3,2,1, SEP, 0,1,2,3,4,5,6,7,8,9, SEP2, 0,1,1,1,1,1,1,1,1,1,1]`
- **Sampling**: All digit lengths 1-10 sampled uniformly per batch. 20% of samples use all-9s operands to boost carry exposure.
- **Evaluation**: 2,000 autoregressive pairs (seed=2025) for checkpoint selection.

### Training Procedure

**Phase 1: High-LR exploration (50K steps)**
```
Optimizer: AdamW (weight_decay=0.01)
Learning rate: 0.01 (cosine decay to 0.0001)
Batch size: 512
Loss: Cross-entropy on answer digit positions only (positions 21-31)
Gradient clipping: max norm 1.0
```

Key milestones during Phase 1:
- Step 5K: 45% teacher-forced
- Step 10K: 67% TF, 46% autoregressive
- Step 20K: 95% TF, 90.6% autoregressive
- Step 30K: 97% TF, 98.3% autoregressive
- Step 50K: 93% TF, 98.6% autoregressive (best checkpoint saved)

The model oscillated between 94-98.6% autoregressive during Phase 1, indicating the solution was nearly grokked but unstable at this learning rate.

**Phase 2: Low-LR stabilization (5K steps from best Phase 1 checkpoint)**
```
Optimizer: AdamW (weight_decay=0.01)
Learning rate: 0.001 (cosine decay to 0.00001)
Batch size: 512
Starting from: best Phase 1 checkpoint (98.6% autoregressive)
```

- Step 1K: 100% teacher-forced
- Step 5K: 100% teacher-forced, 100% autoregressive (2000/2000 pairs)

Total training: ~55K gradient steps, ~35K unique steps with meaningful learning, ~10 minutes wall clock on Apple M-series MPS.

### Warm Initialization

The embedding and V-projection were initialized to match the hand-coded solution's structure:

- **Embedding**: Digit tokens map to `[digit_value, 0, 0]`. Separator tokens map to `[0, 0, 0]`.
- **V-projection**: Head 0 and Head 1 extract dim 0 (`+1, 0, 0`), Head 2 negates dim 0 (`-1, 0, 0`).
- All other weights: PyTorch default initialization. Q and K initialized near zero (std=0.01) so the fixed mask dominates early training.

### Hyperparameter Search

No formal hyperparameter search. The approach was developed iteratively:

1. Proved concept with mask-large (d=6, ff=12, 456 params) -- 99.72% on verify.py
2. Shrunk to mask-w12 (d=3, ff=12, 204 params) -- 100% on verify.py
3. Shrunk further to mask-w6 (d=3, ff=6, 162 params) -- 100% on verify.py
4. Attempted mask-w4 (d=3, ff=4, 148 params) and mask-w5 (d=3, ff=5, 155 params) -- training not converged at time of writing

Key finding: LR 0.01 works for initial training, LR 0.001 needed for stabilization.

## What Makes This Novel

1. **Fixed mask from analytical solution**: The attention routing pattern (which positions attend to which) is derived from our hand-coded solution, not learned. This provides perfect positional routing with 0 learnable parameters.

2. **Hybrid hand-coded + trained**: We exploit knowledge of the addition algorithm's structure (carry propagation, digit pairing) while letting gradient descent find the optimal numerical implementation. This bridges the "hand-coded" and "trained" categories.

3. **Two-phase training**: High-LR exploration to find the basin, low-LR stabilization to converge precisely. Simple but effective for this task.

4. **ALiBi slopes in fixed mask**: The carry head's attention scores follow `-(k-j) * log(10)` spacing, which causes softmax to produce weights proportional to `10^(j-k)` -- the exact weighting needed for carry computation via prefix sum.

## Verification

```
$ uv run python verify.py submission_trained.py
Model: 162-Parameter Trained Adder
Author: fblissjr
Parameters (unique): 162
Architecture: 1L d=3 3h ff=6 reversed-digits

Results: 10010/10010 correct (100.00%)
Time: 21.3s (469 additions/sec)
Status: QUALIFIED (threshold: 99%)
```

## Files

- `submission_trained.py` -- The 162-param submission (primary)
- `submission_trained_mask-w6.py` -- Same as above (named by config)
- `submission_trained_mask-w12.py` -- 204-param backup submission (100%)
- `submission_trained_mask-large.py` -- 456-param proof of concept (99.72%)
- `train_adder.py` -- Training script with all configs
- `train_continue.py` -- Continuation/fine-tuning script with autoregressive eval
- `trained_mask-w6_best.pt` -- Best W6 checkpoint (Phase 1)
- `trained_mask-large.pt` -- mask-large checkpoint

## Reproducibility

To reproduce the 162-param submission:

```bash
# Phase 1: High-LR training
uv run python train_continue.py \
  --config mask-w6 \
  --lr 0.01 \
  --max-steps 100000 \
  --score-interval 10000 \
  --score-pairs 2000

# Phase 2: Low-LR stabilization from best checkpoint
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

Note: Due to random initialization, exact reproduction requires the same random seed. However, the approach reliably converges across multiple runs -- W6 converged on the first attempt, and W12 also converged independently.

## Comparison to Other Approaches

| Submission | Params | Accuracy | Method |
|---|---|---|---|
| **This (mask-w6)** | **162** | **100%** | **Fixed mask + AdamW** |
| Trained leader as of 2/28/26 | 311 | 99%+ | SGD/Adam grokking |
| Hand-coded leader as of 2/28/26 | 36 | 100% | Analytical |
| Our hand-coded 1L | 33 unique | 100% | Analytical (ALiBi prefix sum) |
| Our hand-coded 2L | 249 | 100% | Analytical (carry in MLP) |
