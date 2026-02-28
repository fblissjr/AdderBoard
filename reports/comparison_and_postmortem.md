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

### Gemini: 0-Parameter Analytical Adder (FAILED)

- **Result**: 0% accuracy (shape crash, then wrong answers)
- **Architecture**: 1L, d=1, 2h with ALiBi + cosine decode
- **Approach**: Base-10 prefix sum via ALiBi, softmax anchoring
- **Time**: ~1 hour reasoning

---

## Postmortem: Why Things Succeeded or Failed

### Why the hand-coded model worked

1. **Incremental verification**: Tested on 5 concrete examples (including edge
   cases like 999999999+1 and 9999999999+9999999999) before running the full
   suite. Bugs would have been caught immediately.

2. **Clean data flow**: Each dimension had a single purpose. No information
   interference between carry detection and sum computation.

3. **Exact arithmetic**: Every operation (ReLU thresholding, softmax averaging,
   parabolic decode) produces exact results for integer inputs in float64.
   No accumulated error.

4. **Mathematical proof before code**: The carry detection theorem (diff in
   {-1,0,9,10}) and the impossibility of linear mod-10 decode were proven
   before writing a single line, preventing architectural dead ends.

### Why training failed

1. **Missing rank-3 factorization**: Every successful trained model on the
   leaderboard (311-777 params) uses rank-3 factorized projections. This
   constrains the weight space to force the model toward the addition algorithm.
   Without it, the model has too many degrees of freedom and gets stuck in
   local minima that look like addition but aren't.

2. **No weight tying**: Tying the embedding and LM head (as all successful
   models do) halves the search space for the output mapping.

3. **Insufficient training duration**: The grokking phenomenon requires training
   well past memorization. Our 107K steps may not have been enough -- some
   successful models train for 500K+ steps.

4. **Wrong evaluation frequency**: Checking accuracy every 1000 steps with
   only 200 samples gives noisy estimates. Grokking can be sudden -- the model
   might jump from 5% to 95% between two checkpoints.

### Why Gemini's approach failed

1. **No testing loop**: Gemini proved correctness symbolically but never
   executed the code. The first bug (tensor shape) would have been caught by
   any single test case.

2. **Indexing inconsistency**: The attention bias matrix M was built for one
   input format, but the add() function used a different one. Positions didn't
   match.

3. **Fragile numerical chain**: Softmax anchoring (multiply by e^80) followed
   by carry detection (threshold at 1.0 with epsilon 1e-11) creates a long
   chain of floating-point operations where errors accumulate.

4. **Optimized for elegance, not correctness**: The 0-parameter claim is
   technically interesting but practically useless if the model doesn't work.

---

## Approaches We Didn't Try (and Why)

### 1. Single-Layer with ALiBi (would likely work, fewer params)

The 36-param leader uses ALiBi with slope=log(10) to implement base-10 prefix
summation in a single attention head. This computes the carry for position i
by attending to all preceding digits with exponentially decaying weights that
match powers of 10.

**Why we didn't**: The ALiBi approach requires understanding a subtler
mathematical construction (the fractional carry emerges from the decimal
expansion of the prefix sum, not from explicit carry detection). We chose the
more straightforward 2-layer approach to maximize chance of success.

**Estimated params**: 40-80 range if implemented correctly.

### 2. RoPE with Period Matching (Wonderfall's approach, 40 params)

RoPE with period 19 creates identical rotary embeddings for positions that
differ by 19. With the right input format, this makes A_i and B_i
"look identical" to the attention mechanism, enabling natural digit pairing.

**Why we didn't**: Requires MLX (Wonderfall's implementation) or careful
PyTorch RoPE construction. The period-19 trick is specific to a particular
input format (padded to make offsets align), adding complexity.

**Estimated params**: 40-60.

### 3. Rank-1 Factorized Projections (would reduce our param count)

Instead of full d_model x d_model weight matrices, express each as an outer
product of two vectors: W = u * v^T. This gives 2*d instead of d^2 params.

**Why we didn't**: Our priority was correctness, not compression. This is a
pure optimization step that doesn't change the mathematical approach.

**Estimated savings**: Could reduce our 249 to ~80-100.

### 4. Sparse Embedding with Weight Sharing

Our embedding uses 10 non-zero values in a 11x5 matrix. Could share the
embedding with the LM head (parabolic decode already uses structured weights).

**Why we didn't**: Same as above -- optimization, not architecture change.

**Estimated savings**: ~30-40 params.

### 5. Evolutionary/Search-Based Weight Finding

Use an optimization algorithm (CMA-ES, random search) to find weights for a
small hand-coded architecture, bridging the gap between "hand-coded" and
"trained."

**Why we didn't**: Time constraint. This requires implementing a search loop
and running many verification iterations.

### 6. Knowledge Distillation from Our Working Model

Use our 249-param working model as a teacher to train a smaller student model.
The teacher provides soft targets that are much easier to learn from than raw
addition data.

**Why we didn't**: Requires training infrastructure that we'd need to debug.
The hand-coded approach was already working.

---

## Hybrid Approach: Architecture + Data-Driven Strategy

The most promising path forward combines both approaches:

### Phase 1: Architecture-Guided Initialization

Start with our hand-coded 249-param model and identify which weight values are
"structural" (routing, thresholding) vs "computational" (digit values, scales).
Use the structural weights as initialization for a smaller model.

### Phase 2: Rank-3 Factorized Training

Replace full weight matrices with rank-3 factorizations (the key trick from
all successful trained models). Initialize the factorized weights to approximate
our hand-coded solution. This constrains the search space while allowing the
optimizer to find a more compact representation.

### Phase 3: Curriculum with Warm Start

Train with curriculum learning, but starting from a model that already works
for simple cases (since it was initialized from the hand-coded solution).
This should dramatically accelerate convergence compared to random init.

### Phase 4: Grokking Phase with Careful Monitoring

After the curriculum, run extended training at low LR with frequent evaluation.
Watch for the grokking transition -- the moment where the model generalizes
from memorization to algorithm.

### Expected Outcome

This hybrid approach should:
- Achieve 99%+ accuracy (since we start from a working solution)
- Reduce param count to 100-200 range via factorization
- Potentially reach the 300-500 range for the trained category
- Avoid the training pitfalls we hit (stuck at 2.5%) because the model
  starts in a good region of weight space

---

## Lessons Learned

1. **Test early, test often**: Claude tested after 30 minutes of work. Gemini
   tested after 1 hour. The model that tested first was the one that worked.

2. **Prove impossibility before building**: The linear-mod-10 impossibility
   proof saved an entire architecture iteration. Know what CAN'T work before
   investing time in what might.

3. **Separate concerns**: The 2-layer approach works because each layer has
   one job. Trying to combine carry detection and mod-10 in a single layer
   leads to mathematical dead ends.

4. **Float64 is not magic**: Even with float64, softmax introduces normalization
   errors. Exact arithmetic requires careful construction (integer inputs,
   no accumulated error chains).

5. **The leaderboard is a goldmine**: The successful models share common
   techniques (rank-3 factorization, ALiBi, parabolic decode, RoPE digit
   routing). Understanding these techniques before building saves enormous time.

6. **Working beats minimal**: A 249-param model that scores 100% is infinitely
   more valuable than a 0-param model that scores 0%.
