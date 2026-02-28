# AdderBoard: Comparison, Postmortem, and Future Directions

Last updated: 2026-02-28

---

## The Challenge

Build the smallest transformer that adds two 10-digit numbers with >= 99%
accuracy on 10,000 held-out random pairs. Two categories: hand-coded weights
(constructive proof) and trained weights (learned from data).

Current leaders: 36 params hand-coded (alexlitz), 311 params trained (rezabyt).

---

## What We Tried

### Claude: Hand-Coded 2L Decoder (SUCCESS)

- **Result**: 100% accuracy, 249 params, QUALIFIED
- **Architecture**: 2L decoder, d=5, 2h, ff=4
- **Approach**: Fixed attention routing + clamped ReLU carry + two-hinge mod-10
  + parabolic LM head
- **Time**: ~2 hours from plan to verification

### Claude: Trained 1L Decoder (FAILED)

- **Result**: 2.5% max accuracy after 107K steps
- **Architecture**: 1L decoder, d=7, 1h, ff=14 (777 params)
- **Approach**: Curriculum learning (3 phases) + extended grokking
- **Time**: ~45 min training on CPU

### Gemini v1: 0-Parameter Analytical Adder (FAILED)

- **Result**: 0% accuracy (shape crash, then wrong answers)
- **Architecture**: 1L, d=1, 2h with ALiBi + cosine decode
- **Approach**: Base-10 prefix sum via ALiBi, softmax anchoring
- **Time**: ~1 hour reasoning, no testing

### Gemini v2: 33-Parameter Standard Transformer (SUCCESS)

- **Result**: 100% accuracy, 33 params, QUALIFIED
- **Architecture**: 1L decoder, d=3, 3h (d_head=1), ff=4
- **Approach**: Fixed 3 bugs from v1, added residual cancellation head,
  switched to parabolic decode, tested locally before submitting
- **Time**: Second iteration after failure analysis

---

## Postmortem: Why Things Succeeded or Failed

### Why Claude's hand-coded model worked (but is too large)

1. **Incremental verification**: Tested on 5 concrete examples (including edge
   cases like 999999999+1 and 9999999999+9999999999) before running the full
   suite. Bugs would have been caught immediately.

2. **Clean data flow**: Each dimension had a single purpose. No information
   interference between carry detection and sum computation.

3. **Exact arithmetic**: Every operation (ReLU thresholding, softmax averaging,
   parabolic decode) produces exact results for integer inputs in float64.

4. **Mathematical proof before code**: The carry detection theorem (diff in
   {-1,0,9,10}) and the impossibility of linear mod-10 decode were proven
   before writing a single line, preventing architectural dead ends.

5. **But: wrong architectural choice**: Using 2 layers and d_model=5 was
   correct but far from minimal. The carry-in-MLP approach requires 2 layers;
   the carry-in-attention approach (ALiBi prefix sum) needs only 1. This
   architectural decision is the root cause of the 7.5x parameter gap vs
   Gemini v2.

### Why Claude's training failed

1. **Missing rank-3 factorization**: Every successful trained model on the
   leaderboard uses rank-3 factorized projections. This constrains the weight
   space to force the model toward the addition algorithm.

2. **No weight tying**: Tying the embedding and LM head halves the search space
   for the output mapping.

3. **Insufficient training duration**: The grokking phenomenon requires training
   well past memorization. Our 107K steps may not have been enough -- some
   successful models train for 500K+ steps.

4. **Wrong evaluation frequency**: Checking accuracy every 1000 steps with
   only 200 samples gives noisy estimates. Grokking can be sudden.

### Why Gemini v1 failed (0%)

1. **No testing**: Proved correctness symbolically but never ran the code.
2. **Three trivial execution bugs**: 2x multiplier in O_proj, missing LM head
   term, unmasked attention during generation. Any single test case would have
   caught all three.
3. **Overfit to elegance**: The 0-parameter claim and cosine decode were
   mathematically interesting but added fragility.

### Why Gemini v2 succeeded (100%, 33 params)

1. **Identified exact bugs**: Didn't redesign the math -- fixed three specific
   implementation errors.
2. **Added residual cancellation**: Head 2 with V=[-1,0,0] cancels the residual
   connection, giving clean A+B to the MLP regardless of previous output.
3. **Carried in attention**: ALiBi prefix sum computes carries in one attention
   pass, enabling single-layer architecture.
4. **d_model=3 with d_head=1**: Maximum information density per parameter.
5. **Actually tested locally**: The testing that should have happened in v1.

---

## The Critical Architectural Decision

The central lesson is about WHERE carry detection happens:

| Approach | Carry location | Layers needed | Params |
|----------|---------------|---------------|--------|
| Claude | MLP (compare prev output to prev digit sum) | 2 | 249 |
| Gemini v2 | Attention (ALiBi prefix sum) | 1 | 33 |
| alexlitz (leader) | Attention (ALiBi slope=log10) | 2 (but minimal) | 36 |

**Carry-in-attention** is the key insight. The ALiBi prefix sum
S_k = sum_{j<k} (A_j + B_j) * 10^(j-k) naturally encodes the carry as
floor(S_k) in {0, 1}. This is computed entirely within the attention mechanism
using fixed attention biases (0-param buffers), meaning the MLP only needs to:
1. Extract the integer carry (+1 if carry)
2. Subtract 10 if total >= 10

**Carry-in-MLP** (our approach) requires the MLP to first detect whether a carry
occurred by comparing the previous output digit to the previous digit pair sum.
This comparison requires reading the previous output, which means:
1. Layer 1 attention gathers the previous digit pair AND previous output
2. Layer 1 MLP detects the carry
3. Layer 2 MLP computes mod-10

Two operations chained through the MLP require two layers. The linear mod-10
impossibility proof is correct (a linear LM head can't do mod-10 for [0,19]),
but it's irrelevant to Gemini's approach because the MLP computes mod-10 before
the LM head sees it.

---

## Approaches We Didn't Try (and Why)

### 1. Single-Layer with ALiBi + Residual Cancellation (Gemini v2's approach)

This is exactly what Gemini v2 built: d_model=3, 3 heads, ALiBi prefix sum
for carry, head 2 for residual cancellation, 2-hinge MLP, parabolic decode.

**Why we didn't**: We chose the more straightforward 2-layer approach to
maximize chance of first-attempt success. The ALiBi prefix sum requires
understanding the fractional carry representation, which is subtler than
explicit carry detection. In hindsight, we should have spent more time studying
the 36-param leader's architecture before committing to 2 layers.

**Result**: 33 params, 100% accuracy. Gemini proved it works.

### 2. RoPE with Period Matching (Wonderfall's approach, 40 params)

RoPE with period 19 creates identical rotary embeddings for positions that
differ by 19. With the right input format, this makes A_i and B_i
"look identical" to the attention mechanism, enabling natural digit pairing.

**Why we didn't**: Requires MLX or careful PyTorch RoPE construction. The
period-19 trick is specific to a particular input format.

### 3. Rank-1 Factorized Projections (compression of our model)

Express each weight matrix as an outer product W = u * v^T. Gives 2*d instead
of d^2 params per matrix.

**Why we didn't**: Optimization step, not architecture change. Would reduce
our 249 to ~80-100 but still wouldn't match the 33-param single-layer approach.

### 4. Evolutionary/Search-Based Weight Finding

Use CMA-ES or random search to find weights for a small architecture. Bridges
the gap between "hand-coded" and "trained."

**Why we didn't**: Time constraint. Requires implementing a search loop.

### 5. Knowledge Distillation

Use our 249-param model as teacher for a smaller student.

**Why we didn't**: Requires training infrastructure. The hand-coded approach
was already working.

---

## What We'd Do Differently

### 1. Study the leader before building

We studied the leaderboard entries (alexlitz 36p, Wonderfall 40p) but didn't
fully internalize the ALiBi prefix sum technique. We understood the math but
chose the "safer" 2-layer approach. In hindsight, the 1-layer ALiBi approach is
not harder to implement -- it's just harder to derive from scratch. We should
have spent 30 more minutes understanding the carry-in-attention trick before
committing to an architecture.

### 2. Start at d_model=3, not d_model=5

Our d_model=5 wastes 2 dimensions on scratch space. The key data to route is:
digit value (1 dim), digit pair sum (1 dim), carry indicator (1 dim). That's 3.
Gemini's d_head=1 per head is the natural minimum. We should have asked "what's
the minimum d_model?" before "how many layers?"

### 3. Address the residual problem directly

We handled the residual connection by keeping token identity in a separate
dimension and routing around it. Gemini's Head 2 cancellation is cleaner: it
actively removes the residual contribution rather than working around it. This
is a design pattern worth remembering: if the residual is in your way, cancel
it with a dedicated attention head.

### 4. The trained track needed domain-specific tricks

Vanilla transformer + curriculum learning was never going to work for this
problem at 777 params. Every successful trained model uses:
- Rank-3 factorized projections
- Weight-tied embedding/LM head
- Very long training (500K+ steps)
- Specific optimizer settings for grokking

We should have either implemented these tricks or skipped the trained track
entirely in favor of optimizing the hand-coded model.

---

## Revised Path Forward

### For hand-coded category: Adopt Gemini v2's architecture

The 33-param model is the clear template. To beat the 36-param leader:

1. The 33-param model already beats alexlitz (36p) on unique param count.
   The question is whether the leaderboard counts unique values or tensor
   elements -- this determines whether Gemini's submission is competitive.

2. Potential optimizations: weight tying between embed and LM head could
   reduce unique values further. The parabolic decode W[c,0] = 2c and embed
   values 0-9 are related (W[c,0] = 2*embed[c,0]).

### For trained category: Architecture-guided training

1. Start with Gemini v2's 33-param architecture as initialization
2. Add rank-3 factorization to weight matrices
3. Train with curriculum + extended grokking (500K+ steps)
4. Use weight tying between embed and LM head
5. Monitor for sudden grokking transition with frequent evaluation

### For understanding: Trace every intermediate value

Both working models (Claude 249p, Gemini 33p) should be instrumented to dump
intermediate tensor values for specific test cases. This would:
- Confirm the mathematical analysis
- Identify any numerical edge cases
- Provide debugging traces for future optimization attempts

---

## Lessons Learned

1. **Test early, test often**: Claude tested after 30 minutes. Gemini v1 never
   tested and scored 0%. Gemini v2 tested locally and scored 100%. Testing is
   the single strongest predictor of success.

2. **Architecture > optimization**: The choice of carry-in-attention (1 layer)
   vs carry-in-MLP (2 layers) matters more than any parameter compression trick.
   Gemini v2's 33 params vs our 249 is an architectural difference, not an
   optimization difference.

3. **Prove impossibility, but know its scope**: The linear-mod-10 impossibility
   proof is correct and useful -- it rules out certain architectures. But it
   doesn't rule out ALL 1-layer approaches, only those that defer mod-10 to the
   LM head. Computing mod-10 in the MLP (as both models do) sidesteps the proof.

4. **Float64 anchoring works if margins are sufficient**: Our earlier skepticism
   about e^80 softmax anchoring was wrong. The error is ~3.6e-35, and the
   thresholds use 1e-11 epsilon. That's 24 orders of magnitude of margin. The
   technique is robust.

5. **Residual cancellation is a reusable pattern**: Dedicating an attention head
   to cancel the residual connection's contribution is a powerful trick for
   hand-coded transformers. It gives the MLP clean input without needing extra
   dimensions or layers.

6. **Failure is data**: Gemini's 0% first attempt directly produced the bug
   analysis that led to the 100% second attempt. Without the failure, the
   three specific bugs might never have been identified. Our postmortem
   predicted this: "The mathematical framework is largely sound... the failure
   was in execution."

7. **Working beats minimal, but minimal+working beats both**: A 249-param model
   that scores 100% is better than a 0-param model that scores 0%. But a
   33-param model that scores 100% is better than both.
