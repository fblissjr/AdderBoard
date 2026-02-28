# Claude's Approach: Hand-Coded 2L Transformer Adder

Last updated: 2026-02-28

## Summary

Built a 2-layer decoder transformer with analytically set weights that achieves
100% accuracy on the full 10,010-case test suite. 249 parameters total. Took
roughly 2 hours from blank slate to qualified submission.

---

## Strategy

### Two Parallel Tracks

**Track 1 (Hand-Coded):** Build a transformer with manually derived weights that
implements addition through explicit mathematical construction. Priority: get it
working first, optimize params later.

**Track 2 (Trained):** Standard transformer trained from scratch with curriculum
learning on synthetic data. Backup approach in case the math is harder than
expected.

### Why This Ordering

Hand-coded models dominate the leaderboard at low param counts (36 vs 311 for
trained). The math behind addition is well-understood, so constructing an
analytical solution seemed tractable. Training was the fallback.

---

## Architecture: What We Built

```
2-layer decoder transformer
d_model = 5
n_heads = 2 (d_head = 2 each, with 1 dim unused)
ff_dim  = 4 per layer
vocab   = 11 (digits 0-9 + separator)
dtype   = float64
```

### Input Format

Reversed digits (LSB-first) with separator tokens:

```
a0 a1 a2 ... a9 SEP b0 b1 b2 ... b9 SEP
```

22 tokens total. Output: 11 sum digits autoregressively (LSB-first).

### Data Flow

When predicting digit s_i (the model outputs logits at position 21+i):

**Layer 1 Attention** (fixed routing via 0-param buffer):
- Head 0 attends to positions i and 11+i with equal weight
  -> produces A_i + B_i (current digit pair sum, scaled by 2 to undo averaging)
- Head 1 attends to positions i-1 and 10+i with equal weight
  -> produces A_{i-1} + B_{i-1} (previous pair sum, for carry detection)

After attention + residual:
- dim 0 = current token value (s_{i-1} from embedding, or 0 for first digit)
- dim 1 = A_i + B_i (range 0-18)
- dim 2 = A_{i-1} + B_{i-1} (range 0-18)
- dim 3 = 0
- dim 4 = 0

**Layer 1 MLP** (carry detection):
- Computes diff = dim2 - dim0 = (A_{i-1}+B_{i-1}) - s_{i-1}
- diff is always in {-1, 0, 9, 10} (proved below)
- carry = ReLU(diff - 0.5) - ReLU(diff - 1.5)
- This gives exact 0 or 1 for all four possible diff values
- Stores result in dim 4

After Layer 1 MLP:
- dim 4 = carry_in (0 or 1)

**Layer 2** (no attention, just MLP):
- Computes S = dim1 + dim4 (raw sum, range 0-19)
- n0 = ReLU(S + 0.5) [always positive, = S + 0.5]
- n1 = ReLU(2S - 19) [positive when S >= 10]
- n2 = ReLU(2S - 20) [positive when S >= 11]
- step(S>=10) = n1 - n2 (exact 0 or 1 for integer S)
- result = n0 - 10*n1 + 10*n2 = (S mod 10) + 0.5
- Stores in dim 3

**LM Head** (parabolic decode):
- logit[c] = -(dim3 - c - 0.5)^2
- Expanded: W[c,3] = 2(c+0.5), B[c] = -(c+0.5)^2
- Peaks at the correct digit c = (A_i + B_i + carry) mod 10

---

## Key Mathematical Insights

### 1. The Carry Detection Theorem

For correct autoregressive addition where s_{i-1} = (A_{i-1} + B_{i-1} + C_{i-2}) mod 10:

```
diff = A_{i-1} + B_{i-1} - s_{i-1}
```

**Claim**: diff is always in {-1, 0, 9, 10}.

**Proof**: Let x = A_{i-1} + B_{i-1} (range 0-18).

Case 1 (C_{i-2} = 0):
  raw = x, s_{i-1} = x mod 10
  If x < 10: s = x, diff = x - x = 0
  If x >= 10: s = x - 10, diff = x - (x-10) = 10

Case 2 (C_{i-2} = 1):
  raw = x + 1, s_{i-1} = (x+1) mod 10
  If x < 9: s = x+1, diff = x - (x+1) = -1
  If x >= 9: s = x+1-10 = x-9, diff = x - (x-9) = 9

So diff in {-1, 0, 9, 10}. Carry = 1 iff diff >= 9.

### 2. Clamped ReLU Step Function

For diff in {-1, 0, 9, 10}, the function:

```
carry = ReLU(diff - 0.5) - ReLU(diff - 1.5)
```

produces:
- diff = -1: ReLU(-1.5) - ReLU(-2.5) = 0 - 0 = 0
- diff = 0:  ReLU(-0.5) - ReLU(-1.5) = 0 - 0 = 0
- diff = 9:  ReLU(8.5) - ReLU(7.5) = 8.5 - 7.5 = 1
- diff = 10: ReLU(9.5) - ReLU(8.5) = 9.5 - 8.5 = 1

Exact. No approximation needed.

### 3. Two-Hinge Mod-10

For integer S in [0, 19]:

```
step(S >= 10) = ReLU(2S - 19) - ReLU(2S - 20)
```

- S=9:  ReLU(-1) - ReLU(-2) = 0
- S=10: ReLU(1) - ReLU(0) = 1
- S=11: ReLU(3) - ReLU(2) = 1
- S=19: ReLU(19) - ReLU(18) = 1

Then: S mod 10 = S - 10 * step(S >= 10).

### 4. Linear LM Head Cannot Do Mod-10 (impossibility proof)

**Claim**: No linear function logit[c] = W[c]*x + B[c] can have
argmax_c = x mod 10 for all x in {0, ..., 19}.

**Proof sketch**: For x=5 and x=15, both should output digit 5. This requires:
- W[5]*5 + B[5] > W[c]*5 + B[c] for all c != 5
- W[5]*15 + B[5] > W[c]*15 + B[c] for all c != 5

For x=0 and x=10, both should output digit 0:
- B[0] > B[5] (from x=0 constraint)
- 10*(W[0]-W[5]) > B[5]-B[0] (from x=10 constraint)

Combining with the x=5 constraint: need 5*(W[5]-W[0]) > B[0]-B[5] > 0
and 10*(W[0]-W[5]) > -(B[0]-B[5]) < 0, so 10w < d and 5w > d where
w = W[5]-W[0], d = B[0]-B[5] > 0. This gives 5w > d > 10w, so 5w > 10w,
meaning w < 0. But then 5w > d > 0 with w < 0 is impossible.

**Consequence**: Mod-10 must be computed BEFORE the LM head, which requires
chaining carry detection -> mod-10 in the MLP. A single MLP layer can't chain
two operations, so 2 layers are required (or alternative architectures like
ALiBi/RoPE that handle alignment differently).

---

## What Worked

1. **Fixed attention bias buffers**: Zero parameters for routing. The model knows
   exactly where each digit pair lives via hardcoded attention masks.

2. **2x scaling in V projection**: Undoes the softmax averaging (equal weight
   across 2 attended positions) to recover exact digit sums.

3. **Separate dimensions for each signal**: dim0=token, dim1=current sum,
   dim2=prev sum, dim3=result, dim4=carry. Clean data flow with no interference.

4. **float64 throughout**: All integer arithmetic is exact. No precision issues.

5. **Parabolic decode with 0.5 offset**: The mod-10 result has a +0.5 bias from
   the ReLU computation. The LM head accounts for this with shifted parabolas.

---

## What Failed

### Track 2: Trained Model (777 params, 0% -> 2.5% accuracy)

Architecture: 1L decoder, d_model=7, n_head=1, d_ff=14.
Curriculum: 3 phases (1-3 digits for 2K steps, 1-6 for 5K, 1-10 for 20K).
Extended training: 80K additional steps at lower LR.

**Symptoms**:
- Loss decreased steadily (1.92 -> 0.38) showing the model learned something
- But accuracy never exceeded 2.5% -- the model learned per-token patterns
  but couldn't generalize the carry chain
- Loss plateau at ~0.38 suggests the model hit a local minimum where it
  predicts reasonable-looking digits without actually adding

**Root causes**:
- No rank-3 factorization (all successful trained models on the leaderboard use
  this -- it constrains the weight space in a way that forces the model toward
  the addition algorithm)
- No weight tying between embedding and LM head (reduces search space)
- LR likely too high for the grokking phase -- successful models train for
  100K+ steps at very low LR
- Input format may matter -- MSB-first output (some successful models use this)
  vs our LSD-first output

---

## Parameter Breakdown

```
Component          Shape       Params
--------------------------------------
embed.weight       11 x 5        55
v1.weight           4 x 5        20
o1.weight           5 x 4        20
ff1_up.weight       4 x 5        20
ff1_up.bias         4              4
ff1_down.weight     5 x 4        20
ff2_up.weight       4 x 5        20
ff2_up.bias         4              4
ff2_down.weight     5 x 4        20
lm_head.weight     11 x 5        55
lm_head.bias       11            11
--------------------------------------
Total:                           249
```

Most of these are zeros. The actual non-zero unique values number around 25-30.

---

## Verification Results

```
Model: Hand-Coded 2L Adder
Parameters (unique): 249
Architecture: 2L decoder, d=5, 2h, ff=4

Results: 10010/10010 correct (100.00%)
Time: 16.7s (598 additions/sec)
Status: QUALIFIED (threshold: 99%)
```

---

## Optimization Opportunities

The 249-param count is grossly unoptimized. Immediate wins:

1. **Sparse representation**: Most weight matrices are >90% zeros. Using sparse
   tensors or rank-1 factorization could cut params dramatically.

2. **Weight tying**: The V projection for both heads uses the same value (2.0).
   Many constants are reused across layers.

3. **Reduce d_model**: dim 3 is only used as MLP scratch space. Could potentially
   merge it with another dim.

4. **ALiBi/RoPE routing**: Replace fixed attention bias with positional encoding
   that achieves the same routing with fewer structural parameters.

5. **Single-layer with ALiBi**: The 36-param leader uses ALiBi with slope=log(10)
   which computes carry via a base-10 prefix sum in a single attention head,
   eliminating the need for a separate carry detection layer.
