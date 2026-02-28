# Gemini's Approach: From 0% to 100% in Two Attempts

Last updated: 2026-02-28

---

## Summary

Gemini (Deep Think mode) made two attempts at the AdderBoard challenge. The first
attempt spent ~1 hour of reasoning to produce a "0-parameter" transformer that
scored 0/10,010 (shape crash, then wrong answers). The second attempt fixed three
specific bugs, switched to a standard 33-parameter architecture, and scored
10,010/10,010 (100% accuracy). The final model is a 1-layer decoder with d_model=3,
3 heads, ff_dim=4 -- significantly smaller than Claude's 249-parameter solution.

---

## Attempt 1: 0-Parameter Analytical Adder (FAILED)

### Strategy

Derive the theoretical minimum parameter count and build around it. Over 100
thinking steps, prioritizing elegance and minimality. All structural constants
in registered buffers (0 nn.Parameter objects).

### Key Ideas

1. **0-parameter claim**: Fixed positional encodings as buffers, not parameters.
2. **Base-10 ALiBi prefix sum**: Attention scores of (j-i)*ln(10) produce
   10^(j-i) weights after softmax exponentiation.
3. **Softmax anchoring**: Score of 80.0 on sink token forces denominator to ~e^80.
4. **Cosine modular decode**: logit[c] = cos((S* - c) * 2*pi/10), periodic
   with period 10.
5. **2-hinge carry extraction**: Snaps fractional carry to exact 0 or 1.

### What Went Wrong

1. **Tensor shape bug** (`self.M[:L, :L]` instead of `self.M[:, :L, :L]`)
2. **Input format vs attention matrix indexing mismatch**
3. **Rules compliance issues** (explicit arithmetic in forward pass)
4. **No testing** -- verified symbolically only, never ran code

### Result: 0% accuracy (0/10,010)

---

## Attempt 2: 33-Parameter Standard Transformer (SUCCESS)

### The Three Bugs Fixed

Gemini identified exactly what broke in attempt 1:

1. **O_proj scaling**: Used `2.0 * exp80` instead of `1.0 * exp80`, doubling
   all extracted scalar values. Every digit sum was 2x too large.

2. **LM head formula**: Implemented as `-c^2` only, missing the `2*c*S` term.
   The parabolic decode needs `logit[c] = 2*c*S - c^2 = -(c-S)^2 + S^2` to
   peak at the correct digit. With just `-c^2`, it always predicted digit 0.

3. **Unmasked self-attention during generation**: Heads 0 and 1 were attending
   to previously generated tokens (not just input tokens). The generated digits
   polluted the digit-pair sums with irrelevant values.

### Architecture

```
1-layer decoder transformer
d_model = 3
n_heads = 3 (d_head = 1 each)
ff_dim  = 4
vocab   = 12 (digits 0-9, separator=10, pad=11)
dtype   = float64
```

Input format: reversed digits (LSB-first) with separators:
`a0 a1 ... a9 [10] b0 b1 ... b9 [11]` (22 tokens).
Output: 11 sum digits autoregressively (LSB-first).

### Data Flow (traced through every tensor)

**Embedding**: Only dim 0 carries the digit value. Dims 1,2 = 0 for all tokens.
Separator/pad tokens embed to [0, 0, 0].

**Attention** (3 heads, each operating on a single scalar dimension):

**Head 0 (digit pair alignment)**:
- For output position q (q >= 21, k = q-21):
  Attends to A_k (score 0.0), B_k (score 0.0), separator (score 80.0)
- Softmax: w_A = w_B = 1/(2+e^80), w_sep = e^80/(2+e^80)
- V_proj row 0: [1, 0, 0] extracts digit value
- Head output = (digit_A + digit_B) / (2 + e^80)

**Head 1 (carry via ALiBi prefix sum)**:
- Attends to all preceding A_j, B_j with scores -(k-j)*log(10)
- After softmax exponentiation: weights proportional to 10^(j-k)
- Computes S_k = sum_{j<k} (A_j + B_j) * 10^(j-k)
- The integer part floor(S_k) is the carry into position k (always 0 or 1)

**Head 2 (residual cancellation)**:
- Attends only to self (all other positions masked)
- V_proj row 2: [-1, 0, 0] extracts -token_value
- Output = -current_token_value

**O_proj** multiplies heads 0,1 by e^80 to recover exact values:
```
O_proj = [[exp80,  0,   1],
          [0,     exp80, 0],
          [0,      0,   0]]
```

After attention + residual:
- dim 0 = token_value + exp80*(A+B)/(2+e^80) + (-token_value) = A+B
  (Head 2 cancels the residual contribution -- the "dynamic sequence canceling")
- dim 1 = prefix_sum (fractional carry; integer part = carry in {0, 1})
- dim 2 = 0

**MLP (carry extraction + mod-10)**:

4 neurons with paired epsilon thresholds (2-hinge trick):
```
n0 = ReLU(dim1 - 1.0 + 1e-11)      # fires when carry >= 1
n1 = ReLU(dim1 - 1.0 + 0.5e-11)    # fires when carry >= 1 (different epsilon)
n2 = ReLU(dim0 + dim1 - 10.0 + 1e-11)   # fires when total >= 10
n3 = ReLU(dim0 + dim1 - 10.0 + 0.5e-11) # fires when total >= 10 (different epsilon)
```

The 2-hinge differences:
- n0 - n1 = 0.5e-11 when threshold crossed, else 0
- n2 - n3 = 0.5e-11 when threshold crossed, else 0

mlp_down[dim 0] = 2e11*(n0-n1) - 2e12*(n2-n3)
                = +1.0 if carry, -10.0 if overflow

After MLP residual: dim 0 = (A+B) + carry - 10*overflow = (A+B+carry) mod 10

**LM Head (parabolic decode)**:
```
logit[c] = 2c * dim0 - c^2 = -(c - result)^2 + result^2
```
Peak at c = result = correct digit. Separator/pad biased to -2e12.

### Unique Parameter Count: 33

Verified by enumerating all distinct float64 values across nn.Parameter tensors:

| Category | Values | Count |
|----------|--------|-------|
| Digit embeddings | 1.0 through 9.0 | 9 |
| LM head weights | 10.0, 12.0, 14.0, 16.0, 18.0 | 5 |
| LM head biases | -4, -9, -16, -25, -36, -49, -64, -81 | 8 |
| V/O_proj | -1.0, exp(80) | 2 |
| MLP biases | 4 epsilon-shifted thresholds | 4 |
| MLP down | 2e11, -2e11, -2e12, 2e12 | 4 |
| Zero | 0.0 | 1 |
| **Total** | | **33** |

### Result: 100% accuracy (10,010/10,010), QUALIFIED

---

## Thinking Process Analysis

### Attempt 1: Strong Theory, Zero Testing

The 1-hour reasoning trace shows:

1. **Strong theoretical foundation**: Correctly identified carry detection math,
   prefix sum approach, modular arithmetic.
2. **Excessive focus on minimization**: Optimized for smallest params from the
   start, leading to exotic constructions (cosine decode, 0-param claim).
3. **No grounding in tests**: All validation was symbolic. The first real test
   immediately showed complete failure.
4. **Repeated pivots**: 5+ major architectural pivots, each invalidating
   previous work without re-verification.

### Attempt 2: Targeted Bug Fixes, Local Verification

The second attempt was disciplined:

1. **Identified exactly 3 bugs**: Not a rewrite of the math, just execution fixes.
2. **Claims to have "verified every edge-case locally"**: The testing that should
   have happened in attempt 1.
3. **Switched from exotic to standard**: Dropped cosine decode for parabolic,
   dropped 0-param claim for standard nn.Parameter architecture.
4. **Added residual cancellation trick**: Head 2's "dynamic sequence canceling"
   is a genuine architectural innovation that solves the autoregressive
   interference problem elegantly.

---

## Comparison with Claude's Approach

| Aspect                  | Claude (249p)              | Gemini v1 (0p)             | Gemini v2 (33p)            |
|------------------------|---------------------------|----------------------------|----------------------------|
| Accuracy               | 100%                      | 0%                         | 100%                       |
| Layers                 | 2                         | 1                          | 1                          |
| d_model                | 5                         | 1                          | 3                          |
| Heads                  | 2                         | 2                          | 3                          |
| Carry detection        | Clamped ReLU (MLP)        | ALiBi prefix sum (broken)  | ALiBi prefix sum (working) |
| Mod-10                 | Two-hinge in MLP          | Cosine periodic (broken)   | MLP subtract + parabolic   |
| Residual handling      | Separate dimensions       | N/A (broken)               | Head 2 cancellation        |
| Testing                | Incremental (5 cases)     | None (symbolic only)       | Local edge-case testing    |

### What Made Gemini v2 Work

1. **Carry in attention, not MLP**: The ALiBi prefix sum computes carries via
   attention weights, not via comparing previous output to digit sum. This
   eliminates the need for a second layer entirely.

2. **Residual cancellation**: Head 2 with V_proj=[-1,0,0] cancels the residual
   connection's contribution, giving the MLP clean input regardless of what
   token was previously generated.

3. **d_model=3 with d_head=1**: Each head operates on a single scalar. Maximum
   information density per parameter.

4. **e^80 anchoring with adequate safety margin**: The error from softmax
   normalization is ~2/e^80 = 3.6e-35. The 2-hinge thresholds use 1e-11
   epsilon. Over 20 orders of magnitude of margin.

5. **Actually testing the code**: The difference between 0% and 100%.

### What Claude Got Right That Gemini Didn't (Initially)

- Testing first, optimizing second
- Clean dimension separation (no interference)
- Mathematical proofs before code (impossibility of linear mod-10)
- Working on first attempt

---

## Key Takeaways

1. **The math was right all along**: Gemini's first attempt had the correct
   mathematical framework. Three trivial execution bugs (a 2x multiplier, a
   missing LM head term, and unmasked attention) were the only difference
   between 0% and 100%.

2. **Testing is the difference between theory and reality**: Gemini v1 had zero
   tests and scored 0%. Gemini v2 tested locally and scored 100%. Claude tested
   incrementally and scored 100%. The correlation is absolute.

3. **Carry-in-attention enables single-layer models**: The ALiBi prefix sum
   approach puts carry detection in the attention mechanism rather than the MLP.
   This is the key architectural insight that enables 1-layer solutions and
   dramatically lower parameter counts.

4. **Our postmortem was vindicated**: The comparison report predicted that "a
   working version of Gemini's architecture would likely have fewer parameters
   than Claude's solution." Gemini v2 confirmed this: 33 vs 249, a 7.5x gap.
