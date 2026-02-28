# Gemini's Approach: 0-Parameter Analytical Adder

Last updated: 2026-02-28

## Summary

Gemini (Deep Think mode) spent approximately 1 hour of reasoning to produce a
"0-parameter" transformer that claims to solve 10-digit addition. The model uses
clever mathematical tricks (base-10 ALiBi, cosine modular decode, softmax
anchoring) but fails completely on verification: 0/10,010 correct. The first
version crashed with a tensor shape bug; the second "fixed" version produces
wrong answers.

---

## Strategy

Gemini's approach was mathematically ambitious: derive the theoretical minimum
parameter count and build the model around it. The reasoning trace shows deep
engagement with the math (over 100 thinking steps), but ultimately prioritized
elegance and minimality over working code.

### Key Ideas

1. **0-parameter claim**: All structural constants live in a registered buffer
   (the attention bias matrix M). No nn.Parameter objects except a single dummy
   for device detection. The claim is that fixed positional encodings don't count
   as parameters per the rules.

2. **Base-10 ALiBi prefix sum**: Head 2 uses attention scores of
   (j-i)*ln(10) for preceding digit positions, which after exponentiation in
   softmax produces exact 10^(j-i) weights. This computes the weighted digit
   sum needed for carry detection.

3. **Softmax anchoring**: A score of 80.0 on a "sink" position forces the
   softmax denominator to approximately e^80, which can be multiplied out to
   recover exact digit sums.

4. **Cosine modular decode**: logit[c] = cos((S* - c) * 2*pi/10), which is
   periodic with period 10, naturally wrapping values > 9 to the correct digit.

5. **2-hinge carry extraction**: C_prev = 1e11 * ReLU(O2 - 1.0 + 1e-11) -
   1e11 * ReLU(O2 - 1.0), which snaps the fractional carry indicator to
   exactly 0 or 1.

---

## What Went Wrong

### Bug 1: Tensor Shape Mismatch (trivial)

```python
self.M[:L, :L]  # Wrong: slices dims 0 and 1
self.M[:, :L, :L]  # Correct: preserves head dimension
```

This crashed every single test case with a shape error. A trivial indexing bug
that should have been caught by any test.

### Bug 2: The Math Doesn't Actually Work

Even after fixing the shape bug, the model produces wrong answers. The
fundamental issues:

1. **Input format incompatible with routing**: The model uses interleaved input
   `[0] + a_digits + [0] + b_digits + [0, 0]` but the attention bias matrix M
   was constructed for a different layout. The absolute position indices in M
   don't match where the digits actually land in the sequence.

2. **Softmax anchoring is fragile**: Multiplying by e^80 after softmax
   normalization works in theory but introduces floating-point errors that
   compound across digits. The model assumes O1 and O2 are exact integers, but
   they're not -- they're approximate values with errors on the order of 1e-15
   that accumulate.

3. **Head 2 carry detection breaks for long carry chains**: The base-10 prefix
   sum computes a fractional carry indicator, but for chains like 999999999 + 1,
   the fractional parts compound and the 2-hinge ReLU can't cleanly separate
   the carry signal from accumulated error.

4. **Cosine decode has ambiguity at half-integers**: For sum values that land
   near x.5, the cosine function produces nearly equal logits for two adjacent
   digits. With floating-point errors from the attention computation, this
   causes incorrect digit selection.

5. **No self-testing during development**: Gemini's trace shows it verified
   the math symbolically but never ran the code against actual test cases.
   When errors emerged, it fixed symptoms (the shape bug) rather than validating
   the underlying computation.

### Bug 3: Rules Compliance Issues

The first submission was arguably non-compliant: the forward() method contains
explicit arithmetic operations (O1 + C_prev, the ReLU carry logic) that
implement addition-specific control flow. This was flagged by the user, prompting
a redesign that still didn't produce correct results.

---

## Thinking Process Analysis

Gemini's 1-hour reasoning trace reveals a pattern:

1. **Strong theoretical foundation**: Correctly identified carry detection math,
   prefix sum approach, modular arithmetic. The carry theorem (diff in {-1,0,9,10})
   was discovered independently.

2. **Excessive focus on param minimization**: Rather than building a working model
   first, Gemini optimized for the smallest possible parameter count from the
   start. This led to exotic constructions (single-parameter factorization,
   cosine decode) that are mathematically interesting but practically fragile.

3. **No grounding in tests**: The reasoning trace mentions "confirming" and
   "validating" but these were symbolic checks, not actual code execution. The
   first real test (running verify.py) immediately showed complete failure.

4. **Repeated pivots**: The trace shows at least 5 major architectural pivots
   (interleaved -> separated, relative -> absolute bias, cosine -> parabolic
   and back, 0-param -> 1-param -> 0-param). Each pivot invalidated previous
   work without fully re-verifying.

5. **Overconfidence in float64 precision**: Several critical steps assume exact
   integer arithmetic through softmax, but softmax introduces normalization
   errors. The e^80 anchoring trick mitigates but doesn't eliminate these.

---

## Comparison with Claude's Approach

| Aspect                  | Claude                    | Gemini                     |
|------------------------|---------------------------|----------------------------|
| Strategy               | Get working first         | Minimize params first      |
| Layers                 | 2                         | 1                          |
| Parameters             | 249                       | 0 (claimed)                |
| Accuracy               | 100% (10010/10010)        | 0% (crashed/wrong)         |
| Time to solution       | ~2 hours                  | ~1 hour thinking + broken  |
| Carry detection        | Clamped ReLU step         | 2-hinge ReLU on prefix sum |
| Mod-10                 | Two-hinge ReLU            | Cosine periodic decode     |
| Digit routing          | Fixed attention bias      | ALiBi with sink token      |
| Testing                | Incremental (5 cases)     | None (symbolic only)       |
| Main weakness          | High param count (249)    | Doesn't work               |

### What Gemini Got Right

- The carry detection formula (diff in {-1,0,9,10}) -- same as our derivation
- ALiBi with log(10) slope for base-10 weighting -- matches the leaderboard leader
- The idea that fixed attention biases should be 0-param buffers
- Cosine modular decode is actually a valid technique (used by some leaderboard entries)

### What Gemini Got Wrong

- Built for elegance before correctness
- Never tested against actual inputs
- Softmax anchoring introduces precision issues it didn't account for
- Input format and attention matrix indexing were inconsistent
- Rules compliance was an afterthought

---

## Key Takeaway

The mathematical framework behind Gemini's approach is largely sound. The
base-10 ALiBi prefix sum is the exact technique used by the 36-param leaderboard
leader. The failure was in execution: no incremental testing, indexing bugs,
and fragile numerical assumptions. A working version of Gemini's architecture
would likely have fewer parameters than Claude's solution.
