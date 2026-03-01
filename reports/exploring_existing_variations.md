# Exploring Existing Variations

Last updated: 2026-03-01

## Summary

The AdderBoard 162-parameter trained model (`submission_trained.py`) uses a fixed-mask hybrid approach where a hand-coded attention routing mask (0 learnable params) encodes the structure of decimal addition, while 162 trainable weights in V/O projections, MLP, and LM head learn the numerical computation via AdamW. This report explores seven variations that can be built with minimal changes to the existing codebase -- specifically to `build_fixed_mask()` in `train_adder.py` (lines 106-152), `generate_batch()` (lines 319-357), and the training configuration. The variations are ordered from trivial single-constant changes to ambitious redesigns, and each one addresses a distinct research question about what the fixed-mask architecture can and cannot do.


## Variation 1: Different Number Bases

**Difficulty: Trivial**

### What and Why

The ALiBi slopes in Head 1 of `build_fixed_mask()` use `math.log(10)` because the carry computation is base-10 specific: after softmax, the weights become proportional to `10^(j-k)`, which is what makes the prefix sum S_k encode a base-10 carry as `floor(S_k)`. Changing this single constant to `math.log(B)` for any base B makes the carry computation work in base B instead.

This is interesting because it tests the claim that the mask encodes the *structure* of addition rather than just base-10 arithmetic. Binary addition (base-2) should be the easiest case: carry is always 0 or 1, digit values are only 0 or 1, and the MLP needs to compute `mod 2` instead of `mod 10`.

### Code Changes

**`build_fixed_mask()` -- 1 line:**
```python
# Line 142: change from
score = -(k - j) * math.log(10)
# to
score = -(k - j) * math.log(base)
```

Add a `base` parameter to the function signature:
```python
def build_fixed_mask(max_seq=35, mask_type="slopes", anchor=None, base=10):
```

**`generate_batch()` -- minor changes:**
- Replace `10**d` with `base**d` in digit generation
- Replace `f"{a_val:010d}"` with a base-B digit formatting function
- Replace `f"{a_val + b_val:011d}"` similarly

**Model config:**
- Vocab size changes from 12 (digits 0-9 + SEP + SEP2) to `base + 2`
- Embedding shape: `(base+2, d)` instead of `(12, 3)`
- LM head shape: `(base+2, d)` instead of `(12, 3)`

**Warm init:**
- `_warm_init()` adapts trivially: `emb[d, 0] = float(d)` for d in `range(base)` instead of `range(10)`

### Mask Transferability

Fully transferable. The mask structure (3 heads, same routing pattern) is identical. Only the slope constant changes.

### Parameter Count Impact

For base B with d_model=3, d_ff=6:

| Component | Base-10 (current) | Base-B |
|---|---|---|
| Embedding | (12, 3) = 36 | (B+2, 3) = 3(B+2) |
| LM head | (12, 3) + (12,) = 48 | (B+2, 3) + (B+2) = 4(B+2) |
| Other | 78 | 78 |
| **Total** | **162** | **7B + 92** |

- Base-2: 7(2) + 92 = **106 params** (smallest possible)
- Base-3: **113 params**
- Base-8: **148 params**
- Base-16: **204 params**

### Research Question

Does base affect convergence speed and minimum viable MLP width? Base-2 has the simplest carry logic (`mod 2` vs `mod 10`) and the smallest digit value range (0-1 vs 0-9). Hypothesis: base-2 converges fastest and might succeed with `d_ff=4` (148p equivalent for base-10 would be 106p -- below our current minimum). Conversely, base-16 has carries up to 1 but digit sums up to 30, requiring wider MLP thresholds.


## Variation 2: Subtraction

**Difficulty: Moderate**

### What and Why

Subtraction (A - B for A >= B) has the same digit-by-digit structure as addition, but with *borrow propagation* instead of carry. The borrow logic is: if A_k - B_k - borrow_in < 0, then borrow 10 from the next position, output (A_k - B_k - borrow_in + 10), and propagate borrow_out = 1. This mirrors carry propagation exactly, with different MLP thresholds.

This is the most natural "neighboring problem" to test whether the fixed-mask architecture generalizes beyond addition.

### Code Changes

**`build_fixed_mask()` -- no structural change:**

The mask routes output position k to digit pair (A_k, B_k) via Head 0, and to prefix positions via Head 1. For subtraction, the prefix sum computes borrow instead of carry, but the *routing* is identical. The key insight: `S_k = sum_{j<k} (A_j + B_j) * 10^(j-k)` for addition becomes `S_k = sum_{j<k} (A_j - B_j) * 10^(j-k)` for subtraction, and this difference is absorbed by the V-projection sign, not the mask.

The mask can be used unchanged. Alternatively, a `sign` parameter could flip Head 1's V-projection during warm init:
```python
# _warm_init for subtraction:
self.v_proj.weight.copy_(torch.tensor([
    [1.0, 0.0, 0.0],   # Head 0: +A_k (for A_k - B_k, sign handled in O)
    [1.0, 0.0, 0.0],   # Head 1: +digit (prefix sum; borrow = floor of negated sum)
    [-1.0, 0.0, 0.0],  # Head 2: -digit (residual cancellation, unchanged)
]))
```

**`generate_batch()` -- moderate changes:**
- Generate A, B such that A >= B (or learn signed output)
- Compute answer as A - B instead of A + B
- Zero-pad answer to 11 digits (same format, but leading zeros likely)

**MLP thresholds:**
- Addition MLP detects `sum >= 10` (overflow) and `carry >= 1`
- Subtraction MLP detects `diff < 0` (borrow) and `borrow_in >= 1`
- The threshold shifts from `-10 + eps` to `0 + eps` for the overflow/borrow neuron
- This is a training-learned change (no code needed if starting from scratch), but warm init biases would change

### Mask Transferability

Fully transferable. Same 3-head structure, same ALiBi slopes, same routing. The difference is in what the V-projection and MLP learn to compute, not where attention goes.

### Parameter Count Impact

Same 162 params. The V and MLP weights change values but not shape. If restricting to A >= B, the problem is arguably simpler (no sign handling), so 162 params should suffice. Allowing negative results would require a sign token and possibly more params.

### Research Question

Does the same 162-param budget suffice for subtraction, or does the borrow threshold's proximity to zero (vs carry's proximity to 10) create optimization difficulties? The prefix sum for subtraction can go negative, which is a different regime for the MLP's ReLU activations. Also: can a single model learn both addition and subtraction (e.g., operation token after SEP2)?


## Variation 3: Multiply by Small Constant

**Difficulty: Moderate**

### What and Why

Instead of A + B, compute A * c for a small constant c (2 through 9). This simplifies the input (single operand) but complicates the MLP: a digit d times c can produce values up to 9 * 9 = 81, and carry can exceed 1 (e.g., `9 * 9 + 8 = 89`, carry = 8).

This tests whether the architecture can handle multi-value carry, which requires more MLP capacity than binary carry.

### Code Changes

**`build_fixed_mask()` -- simplify Head 0:**

Head 0 no longer needs to attend to two operands. For multiply-by-c:
```python
# Head 0: single digit extraction (A_k only)
if k < 10:
    M[0, q, k] = 0.0       # A_k
# No B_k -- the constant c is baked into weights
M[0, q, 10] = sep_score    # SEP (still needed for anchor)
```

Head 1 unchanged in structure: prefix sum over preceding positions computes carry.

But the ALiBi slope changes subtly. For addition, carry is always 0 or 1, so `floor(S_k)` is binary. For multiply-by-c, carry at position k can be up to `floor((9*c + carry_max) / 10)`. For c=9: max carry = `floor(89/10) = 8`. The prefix sum still works mathematically -- `S_k` still encodes the correct carry -- but the MLP must now extract `floor(S_k)` for S_k that can be 0 through 8, not just 0 or 1.

### Mask Transferability

Partially transferable. Head 0 simplifies (1 operand instead of 2). Head 1 routing is identical but the magnitude of S_k grows. Head 2 unchanged. The mask structure works; the training challenge is in the MLP.

### Parameter Count Impact

The MLP must implement multi-value carry extraction. With `d_ff=6` and binary carry, the model uses 2 neuron pairs (4 neurons for carry, 2 for overflow via 2-hinge). Multi-value carry needs more neurons:
- `d_ff=6`: might handle c=2 (max carry 1, same as addition)
- `d_ff=8-12`: needed for c=3-5 (max carry 2-4)
- `d_ff=16+`: needed for c=6-9 (max carry 5-8)

Estimated param counts (d_model=3):

| Multiplier c | Max carry | d_ff needed | Est. params |
|---|---|---|---|
| 2 | 1 | 6 | ~162 |
| 3 | 2 | 8-10 | ~176-190 |
| 5 | 4 | 12-14 | ~204-218 |
| 9 | 8 | 16-20 | ~232-260 |

The input sequence also changes: `A SEP2 Answer` (10 + 1 + 11 = 22 tokens), shorter than addition. The constant c could be encoded in the embedding or baked into the V-projection.

### Research Question

What is the maximum multiplier c before params explode? Is there a clean transition, or does multi-value carry create a combinatorial cliff? Also: can the constant c be provided as an input token (making this a general single-digit multiplier), or must it be baked into weights?


## Variation 4: Multi-Operand Addition

**Difficulty: Moderate-Hard**

### What and Why

Add three or more numbers: A + B + C (or more operands). The carry range grows with operand count: with n operands of max digit 9, the max digit sum is 9n, giving max carry of `floor(9n / 10)`. For 3 operands, max carry = 2. For 4 operands, max carry = 3.

This tests how the architecture scales with input complexity.

### Code Changes

**`build_fixed_mask()` -- generalize Head 0:**

```python
def build_fixed_mask(max_seq=35, mask_type="slopes", anchor=None, n_operands=2, n_digits=10):
    # Sequence layout: A0..A9 SEP B0..B9 SEP C0..C9 ... SEP2 Answer
    # Each operand is n_digits tokens, followed by SEP (for all but last, which has SEP2)
    # Total input: n_operands * n_digits + n_operands (separators)
    input_len = n_operands * n_digits + n_operands
    # Answer: n_digits + 1 (carry out)
    answer_len = n_digits + 1

    for q in range(max_seq):
        if q < input_len:
            # Prompt: self-attend
            ...
        else:
            k = q - input_len  # output digit index

            # Head 0: digit tuple -- attend to k-th digit of ALL operands
            for op in range(n_operands):
                op_start = op * (n_digits + 1)  # start of operand op's digits
                if k < n_digits:
                    M[0, q, op_start + k] = 0.0  # op's k-th digit

            # Head 1: carry prefix sum over ALL preceding positions
            if k > 0:
                for j in range(min(k, n_digits)):
                    score = -(k - j) * math.log(10)
                    for op in range(n_operands):
                        op_start = op * (n_digits + 1)
                        M[1, q, op_start + j] = score
```

**`generate_batch()` -- generate n operands:**
```python
operands = [torch.randint(0, 10**d, (1,)).item() for _ in range(n_operands)]
answer = sum(operands)
```

**Sequence format:**
- 2 operands (current): 10 + 1 + 10 + 1 + 11 = 33 tokens
- 3 operands: 10 + 1 + 10 + 1 + 10 + 1 + 11 = 44 tokens
- n operands: n*10 + n + 11 = 11n + 11 tokens

`max_seq` must increase accordingly.

### Mask Transferability

Structurally transferable but requires generalization. Head 0 generalizes from 2 to n attention targets per output position. Head 1 generalizes from 2*j positions to n*j positions per output. Head 2 unchanged. The ALiBi slope `math.log(10)` stays the same -- the prefix sum math still works regardless of operand count.

### Parameter Count Impact

The mask itself is still 0 learnable params (just larger buffer). But the MLP must handle multi-value carry:

| n_operands | Max digit sum | Max carry | d_ff needed | Est. params |
|---|---|---|---|---|
| 2 (current) | 18 | 1 | 6 | 162 |
| 3 | 27 | 2 | 8-10 | 176-190 |
| 4 | 36 | 3 | 10-12 | 190-204 |
| 5 | 45 | 4 | 12-14 | 204-218 |

The embedding and LM head stay the same size (vocab is still 12).

### Research Question

How does param count scale with operand count? Is it linear, sublinear, or is there a phase transition? The carry extraction problem for n operands is equivalent to implementing `floor(x)` for x in `[0, floor(9n/10)]`, which requires `O(max_carry)` neuron pairs with the 2-hinge approach. Can the network discover a more efficient encoding?


## Variation 5: Different Digit Counts

**Difficulty: Trivial**

### What and Why

The current model handles 10-digit addition. Parameterizing digit count tests whether the same 162 parameters generalize to smaller or larger problems. Smaller problems (5-digit, 3-digit) should train faster and might converge with fewer MLP neurons. Larger problems (15-digit, 20-digit) test whether the model's capacity is the bottleneck or the mask size.

Key observation: the trainable parameter count is independent of digit count. The embedding, projections, MLP, and LM head have fixed shapes. Only the fixed mask buffer changes size. So 162 parameters should theoretically handle *any* digit count -- the question is whether they actually do.

### Code Changes

**`build_fixed_mask()` -- parameterize loop bounds:**
```python
def build_fixed_mask(max_seq=35, mask_type="slopes", anchor=None, n_digits=10):
    input_len = 2 * n_digits + 2   # A + SEP + B + SEP2
    answer_len = n_digits + 1       # digits + carry-out

    for q in range(max_seq):
        if q < input_len:
            for h in range(3):
                M[h, q, q] = 0.0
        else:
            k = q - input_len
            if k < n_digits:
                M[0, q, k] = 0.0
                M[0, q, n_digits + 1 + k] = 0.0
            M[0, q, n_digits] = sep_score  # SEP position changes
            # Head 1: range(min(k, n_digits)) instead of range(min(k, 10))
            ...
```

**`generate_batch()`:**
- Replace hardcoded `10` with `n_digits`
- `f"{a_val:0{n_digits}d}"` instead of `f"{a_val:010d}"`
- Answer padding: `n_digits + 1` digits

**Config:**
- `max_seq = 2 * n_digits + 2 + n_digits + 1 + 2` (with buffer)

### Mask Transferability

Fully transferable. Identical structure, different size. The ALiBi slope `math.log(10)` is unchanged.

### Parameter Count Impact

Zero. The 162 trainable parameters are in embedding/projections/MLP/LM-head, none of which depend on sequence length. Only the mask buffer (non-trainable) changes size.

Interesting edge cases:
- **1-digit addition**: No carry needed. MLP only needs `mod 10`. Should train in <1K steps.
- **20-digit addition**: Carry chains up to 20 positions. Same parameters, but the prefix sum must be accurate over more positions. The ALiBi slopes at `-(20-0)*log(10) = -46.05` create extreme softmax sparsity.
- **100-digit addition**: The slope at position 100 is `-(100)*log(10) = -230.26`. Softmax weights at distant positions become vanishingly small (~10^{-100}). This might cause numerical issues in float32 but is mathematically sound.

### Research Question

Is there a phase transition in digit count vs accuracy for the fixed 162-param model? At what digit count does float32 precision in the prefix sum become the bottleneck? Can the same checkpoint (trained on 10-digit) generalize to 20-digit without retraining (zero-shot length generalization)?


## Variation 6: Comparison Operators

**Difficulty: Hard**

### What and Why

Given two N-digit numbers A and B, output a single token: 1 if A > B, 0 if A <= B (or variants: <, ==, >=). This is fundamentally different from addition because the output is a single bit, not a sequence of digits.

Comparison can be reduced to subtraction (compare the sign of A - B), but a direct approach is also possible via a "most significant differing digit" scan, which is a natural fit for the ALiBi prefix sum mechanism but requires a different kind of computation.

### Code Changes

**Approach 1: Via subtraction prefix sum**

The prefix sum `S = sum_{j=0}^{N-1} (A_j - B_j) * 10^j` gives A - B exactly (as a signed integer). The sign of S determines the comparison. However, reconstructing the sign from the prefix sum representation (which is a fractional value after the ALiBi softmax) is non-trivial.

**Approach 2: Most-significant-digit-first scan**

For comparison, the most significant differing digit determines the result. This suggests a *reversed* ALiBi slope: the highest-order digit should have the *largest* weight, not the smallest. This means the mask needs:

```python
# Head 1: REVERSED slopes for comparison (MSB-first priority)
score = -(n_digits - 1 - j) * math.log(10)  # opposite direction
```

Or equivalently, process digits MSB-first instead of LSB-first.

**Sequence format:**
- Input: `A0..A9 SEP B0..B9 SEP2` (22 tokens)
- Output: single token (0 or 1)
- `max_seq = 23`

**MLP:**
- Must extract sign from a single scalar. ReLU can do this: `sign(x) = ReLU(x) - ReLU(-x)` normalized.
- But distinguishing A == B from A > B requires exact zero detection, which is hard with ReLU.

### Mask Transferability

Requires significant modification. Head 0 still extracts digit pairs (same routing), but Head 1 needs reversed slopes or a different weighting scheme. The single-token output also changes how the mask's output positions are structured. Not a drop-in change.

### Parameter Count Impact

Unknown, but potentially smaller than addition:
- LM head shrinks from (12, 3) to (2, 3) -- saves 40 params
- MLP might simplify (no mod-10 needed) -- `d_ff=4` might suffice
- Embedding unchanged (still need digit tokens + separators)

Optimistic estimate: ~100-120 params. But the comparison logic might require `d_model > 3` or multiple layers to implement the "find most significant difference" scan.

### Research Question

Can a <100 param model compare two 10-digit numbers? Is the fixed-mask approach viable for non-additive arithmetic, or is the ALiBi prefix sum inherently tied to carry propagation? Comparison is arguably *easier* than addition (single bit output), yet the right mask design is less obvious.


## Variation 7: Sorting Small Sequences

**Difficulty: Hard**

### What and Why

Given N small numbers (e.g., 3 single-digit numbers), output them in sorted order. This is fundamentally different from addition: the output is a *permutation* of the input, not a derived computation. The model must track relative ordering, not cumulative sums.

This is a stress test for the fixed-mask paradigm. Addition has a known, clean algorithmic decomposition (digit pair + carry prefix sum). Sorting's decomposition into attention patterns is less obvious.

### Code Changes

**`build_fixed_mask()` -- complete redesign:**

For sorting N numbers, a comparison-based approach needs:
1. Compare all pairs (i, j) -- requires N*(N-1)/2 comparisons
2. Aggregate comparison results to determine rank of each element
3. Route elements to their ranked positions

This maps poorly to 3 heads. Possible architecture:

- **Head 0**: Each output position k attends to all input elements. Scores encode "rank = k" detection.
- **Head 1**: Pairwise comparison aggregation (how many elements are smaller than element i?).
- **Head 2**: Residual management.

But implementing rank counting in softmax attention is unclear. The ALiBi trick works for carry because carry is a prefix sum with known base-10 structure. Rank counting is not a weighted prefix sum.

**Alternative: Multi-layer approach**

Layer 1: Pairwise comparison (each position computes comparison results with others).
Layer 2: Rank aggregation and routing.

This breaks the 1-layer constraint.

**Sequence format (sort 3 single-digit numbers):**
- Input: `A B C SEP` (4 tokens)
- Output: 3 tokens (sorted)
- `max_seq = 7`

### Mask Transferability

Not transferable. The mask for sorting requires fundamentally different routing logic. No ALiBi prefix sum. No digit-pair extraction. Complete redesign needed.

### Parameter Count Impact

For sorting 3 numbers (simplest non-trivial case):
- If 1-layer is sufficient: possibly ~150-200 params (similar to addition)
- If 2-layer required: ~300-400 params (double the projections)
- For N=5 or N=10: unclear scaling

The key bottleneck is not param count but architectural feasibility. If the comparison + rank logic cannot be expressed in the fixed-mask + MLP framework, no number of parameters will help within 1 layer.

### Research Question

Is sorting possible in a purpose-built tiny transformer? If so, what is the minimum number of layers? Sorting 3 elements requires only 3 comparisons (bubble sort), which *might* fit in 3 attention heads. But translating comparison outcomes into permutation routing is the hard part. This would be a genuinely novel result if achieved.


## Summary Table

| Variation | Difficulty | Mask Change | Param Impact | Priority |
|---|---|---|---|---|
| Different bases | Trivial | 1 constant (`log(10)` -> `log(B)`) | Smaller for B<10, larger for B>10 | 1 (validation) |
| Different digits | Trivial | Loop bounds only | None (same 162) | 2 (validation) |
| Subtraction | Moderate | Same structure | Same (~162) | 3 (novel) |
| Multiply by constant | Moderate | Head 0 simplifies | +30-50% for c>2 | 4 (novel) |
| Multi-operand add | Moderate-Hard | Generalize Head 0 | +10-30% per extra operand | 5 (research) |
| Comparison | Hard | Reversed slopes or redesign | Possibly smaller (~100-120) | 6 (research) |
| Sorting | Hard | Complete redesign | 2-3x if multi-layer needed | 7 (ambitious) |


## Recommended Experiment Priority

### Tier 1: Validation (1-2 days)

**Base conversion and digit count variation** should be done first. These require the smallest code changes (literally 1 constant and loop bounds) and serve as a validation that the approach is robust rather than fragile. If base-2 addition trains in 5K steps instead of 50K, that confirms the mask's role in reducing sample complexity. If 20-digit addition works with the same 162 params, that confirms the architecture's capacity is not digit-count-bound. These results also have standalone value: "162-param model generalizes across bases and digit counts" is a publishable finding.

### Tier 2: Novel Results (3-5 days)

**Subtraction and multiply-by-constant** produce novel results with moderate effort. Subtraction, in particular, is interesting because the mask is identical -- only training changes. If a single model can learn both addition and subtraction (with an operation token), that demonstrates the mask captures the *structure* of digit-by-digit arithmetic, not just addition. Multiplication by a constant is the first variation that tests multi-value carry, which is the key capacity question for the MLP.

### Tier 3: Ambitious Research (1-2 weeks)

**Multi-operand addition, comparison, and sorting** are research-grade problems. Multi-operand addition is the most natural extension (mask generalizes cleanly, only MLP capacity is in question). Comparison is interesting because it requires a fundamentally different output structure (single token vs sequence). Sorting is the most ambitious -- it tests whether fixed-mask transformers can implement non-arithmetic algorithms at all.


## Refactoring Notes

To support all variations without code duplication, `build_fixed_mask()` should be parameterized along four axes:

```python
def build_fixed_mask(
    n_digits: int = 10,       # digits per operand
    n_operands: int = 2,      # number of operands (A, B, C, ...)
    base: int = 10,           # number base (changes ALiBi slope)
    operation: str = "add",   # "add", "sub", "mul_const", "compare"
    max_seq: int = 35,
    mask_type: str = "slopes",
    anchor: float | None = None,
):
    log_base = math.log(base)
    input_len = n_operands * n_digits + n_operands  # digits + separators
    answer_len = n_digits + 1 if operation != "compare" else 1

    M = torch.full((3, max_seq, max_seq), float('-inf'))
    sep_score = anchor if anchor is not None else 0.0

    for q in range(max_seq):
        if q < input_len:
            for h in range(3):
                M[h, q, q] = 0.0
        else:
            k = q - input_len

            # Head 0: digit extraction (varies by n_operands and operation)
            for op in range(n_operands):
                op_start = op * (n_digits + 1)
                if k < n_digits:
                    M[0, q, op_start + k] = 0.0
            # SEP: first separator position
            M[0, q, n_digits] = sep_score

            # Head 1: carry/borrow prefix sum (varies by base)
            if k == 0:
                M[1, q, n_digits] = sep_score
            else:
                for j in range(min(k, n_digits)):
                    score = -(k - j) * log_base  # <-- base parameter
                    for op in range(n_operands):
                        op_start = op * (n_digits + 1)
                        M[1, q, op_start + j] = score

            # Head 2: self-attend (universal)
            M[2, q, q] = 0.0

    return M
```

Similarly, `generate_batch()` should accept `n_digits`, `n_operands`, `base`, and `operation` parameters. The training loop and config system in `train_adder.py` can be extended with these parameters without changing the core training logic.

The key architectural constants affected by these parameters:

| Parameter | Affects | Current value |
|---|---|---|
| `base` | `math.log()` slope, vocab size, embedding/LM head shape | 10 |
| `n_digits` | mask size, `max_seq`, sequence layout, generate_batch | 10 |
| `n_operands` | Head 0 routing, Head 1 routing, sequence layout | 2 |
| `operation` | V-proj warm init, MLP thresholds, answer format | "add" |
