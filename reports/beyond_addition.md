# Beyond Addition: A Roadmap for Tiny Purpose-Built Transformers

Last updated: 2026-03-01

## Executive Summary

The AdderBoard project demonstrates that a 162-parameter, 1-layer transformer can perfectly add two 10-digit numbers when equipped with a hand-coded fixed attention mask and trained weights. This report synthesizes findings from two companion analyses -- [Exploring Existing Variations](exploring_existing_variations.md) and [New Domain Exploration](new_domain_exploration.md) -- to answer a broader question: what else can this approach do, and how far can it go?

The answer is surprisingly far. The fixed-mask hybrid approach is not specific to addition. It exploits five general-purpose techniques that transfer to any domain with position-determined routing and small per-position computation:

1. **ALiBi prefix sum**: Attention scores `-(k-j)*log(B)` produce softmax weights proportional to `B^(j-k)`, computing a base-B weighted prefix sum. The carry (or any propagating state expressible as a weighted scan) falls out as `floor(S_k)`. Transfers to any base, subtraction (borrow), and base conversion.

2. **Residual cancellation**: A dedicated head with V = -token_value and self-attention cancels the residual connection, giving the MLP a clean signal. Transfers universally -- any domain where the MLP should see only the attention output.

3. **Softmax anchoring**: A high-score anchor token (e.g., SEP with score 80.0) dominates the softmax denominator. O_proj scaling by the anchor's exponential recovers exact numerical values. Transfers to any domain needing precise value extraction from attention.

4. **2-hinge ReLU step function**: `ReLU(x+eps1) - ReLU(x+eps2)`, scaled to produce exact 0/1 indicators for threshold detection. Generalizes to multi-threshold extraction via neuron pairs. Transfers to carry detection, borrow detection, modular overflow, and any discrete state extraction.

5. **Parabolic LM head**: `logit[c] = 2cx - c^2 = -(c-x)^2 + x^2` peaks at c = x. Decodes any integer-valued MLP output to the correct token. Transfers to any domain with integer-valued outputs (mod-N arithmetic, state IDs, digit values).

These are not ad hoc tricks. They form a compositional toolkit: the mask handles routing (WHERE to attend), the V/O projections handle value extraction (WHAT to compute from attention), the MLP handles per-position logic (HOW to combine values), and the LM head handles decoding (WHICH token to output). Each component has a clean, transferable abstraction.

## What You Can Do Today

From [Exploring Existing Variations](exploring_existing_variations.md), seven variations are possible with the existing codebase. They fall into three tiers of effort and payoff.

### Tier 1: Validation (hours of work)

**Different number bases** and **different digit counts** require near-zero code changes and produce immediate results.

| Variation | Code Change | Expected Params | Key Test |
|---|---|---|---|
| Base-2 addition | `log(10)` -> `log(2)`, vocab 12->4 | ~106 | Fastest convergence? |
| Base-16 addition | `log(10)` -> `log(16)`, vocab 12->18 | ~204 | Handles hex carries? |
| 5-digit addition | Loop bounds only | 162 (same) | Trains in <5K steps? |
| 20-digit addition | Loop bounds only | 162 (same) | Zero-shot length generalization? |

The 20-digit zero-shot test is particularly compelling: take the existing `submission_trained.py` checkpoint, build a larger mask, and test inference without retraining. If it works, the model has generalized the addition algorithm rather than memorizing 10-digit patterns. If it fails, the failure mode (which digit positions break?) reveals what the model actually learned.

Base-2 addition is the simplest possible validation: binary carry is always 0 or 1, digit values are only 0 or 1, and `mod 2` is simpler than `mod 10`. With vocab shrinking from 12 to 4, the embedding and LM head shed 56 parameters. The question is whether `d_ff=4` suffices for binary (it doesn't for decimal), which would push the parameter count below 100.

### Tier 2: Novel Results (days of work)

**Subtraction** and **multiply by small constant** produce genuinely new results with moderate effort.

Subtraction is the cleanest test of mask generality. The mask is *identical* to addition -- only the trained weights change. Borrow propagation has the same structure as carry propagation: `borrow_k = floor(sum((B_j - A_j) * 10^(j-k)))` when A < B. The V-projection sign flip and MLP threshold shift (from 10 to 0) are absorbed by training. If the same 162-parameter budget learns subtraction, it confirms the mask captures digit-by-digit arithmetic structure, not just addition.

The dual-operation model is even more interesting: can a single model learn both addition and subtraction, switching behavior based on an operation token? This would use the same mask (routing is operation-independent) with operation-conditioned MLP weights, demonstrating that the fixed mask encodes a family of algorithms.

Multiply by small constant is the first test of multi-value carry. For `A * 3`, the carry at each position can be 0, 1, or 2 (since max digit 9 * 3 + 2 = 29, carry = 2). The 2-hinge ReLU trick needs additional neuron pairs: one pair per carry threshold. This directly tests how MLP width (`d_ff`) scales with state complexity.

| Multiplier | Max Carry | Est. d_ff | Est. Params |
|---|---|---|---|
| 2 | 1 | 6 | ~162 |
| 3 | 2 | 8-10 | ~176-190 |
| 5 | 4 | 12-14 | ~204-218 |
| 9 | 8 | 16-20 | ~232-260 |

### Tier 3: Ambitious Research (weeks of work)

**Multi-operand addition**, **comparison operators**, and **sorting** push the architecture to its limits.

Multi-operand addition (3+ operands) generalizes Head 0 from 2 to N attention targets and increases the carry range. The mask structure extends cleanly, but the MLP must handle carry values up to `floor(9N/10)`. For 3 operands: max carry 2, similar to multiply-by-3. The interesting scaling question: does parameter count grow linearly with operand count (additional neuron pairs per carry threshold) or is there a more efficient encoding the network can discover?

Comparison operators require a fundamentally different output structure (single token instead of digit sequence) and possibly reversed ALiBi slopes (MSB has highest priority in comparison, vs LSB-first in addition). The mask design is non-obvious, making this a genuine research problem rather than a variation.

Sorting is the hardest target. Permutation routing is fundamentally different from prefix sums, and multi-layer architectures are likely required even for N=4 elements (3 rounds of compare-and-swap). This tests whether the fixed-mask paradigm extends beyond arithmetic at all.

## New Frontiers

From [New Domain Exploration](new_domain_exploration.md), ten candidate domains were evaluated against seven suitability criteria. The strongest candidates are those that score highest on: deterministic output, short sequences, propagating state, known structure, small vocabulary, weighted-sum state, and local computation.

### Feasibility vs. Interest

```
Interest
  5 |                DFA              Base conv.
    |
  4 |   BPE          CRC        Game theory
    |
  3 |                Sorting          Regex
    |
  2 |                DNA
    |
  1 |
    +----+----+----+----+----+---
    1    2    3    4    5    Feasibility

                    Bin arith.  Caesar/Vig.
```

The sweet spot is the top-right quadrant: **base conversion** and **DFA simulation** offer the highest combination of novelty and tractability. Binary arithmetic and Caesar cipher in the bottom-right are validation targets -- high feasibility, moderate interest. CRC and game theory in the center offer selective upside.

### Top 5 New Domains (Prioritized)

**1. Binary Arithmetic** (feasibility: 5/5, interest: 2/5)
- One-constant change from decimal addition: `log(10)` -> `log(2)`
- Vocab shrinks from 12 to 4, params from 162 to ~80-100
- AND/OR are stateless (each bit independent), XOR needs 2 ReLU neurons per gate
- Goal: establish the absolute parameter floor for a non-trivial autoregressive task

**2. Caesar/Vigenere Cipher** (feasibility: 5/5, interest: 2/5)
- No state propagation -- each output depends on one plaintext char + key
- Simpler than addition: mask needs only 2 heads (plaintext lookup + key lookup)
- MLP computes `(p + k) mod 26` -- same ReLU threshold trick, different modulus
- Goal: prove the approach works for non-arithmetic, non-carrying tasks
- Estimated params: 80-120

**3. Base Conversion** (feasibility: 4/5, interest: 5/5)
- Natural ALiBi extension: binary-to-decimal uses `log(2)` slopes on input, `log(10)` on output
- State is multi-valued (remainder mod B, not just 0/1 carry)
- Tests how MLP width scales with state complexity
- Estimated params: 150-250
- This is the most publishable target: same technique, genuinely different domain, controlled state complexity experiment

**4. Game Theory Strategies** (feasibility: 4/5, interest: 4/5)
- Each strategy (TFT, Pavlov, Grudger) is a tiny state machine
- TFT is trivial: copy last opponent move. Mask: attend to position n-1.
- Grudger requires OR-over-history: steep ALiBi slopes where any defection dominates
- Multi-strategy model: strategy token selects behavior via mask routing
- Estimated params: 50-100 per strategy

**5. DFA Simulation** (feasibility: 3/5, interest: 5/5)
- The most general target: many problems reduce to DFA simulation
- Fixed-DFA variant: mask routes to (prev_state, current_input), MLP learns transition table
- MLP cost: `O(|Q| * |Sigma|)` neurons for transition table lookup
- For a 3-state binary DFA: ~150-300 params
- If successful, establishes a general-purpose framework for structured computation

### Domains to Skip (For Now)

**BPE tokenizer**: Cascading merges require multiple layers and variable-length spans. Large vocabulary. Fascinating but needs breakthroughs in mask design.

**Regex matching**: Better approached via DFA simulation. Direct regex requires variable-structure masks.

**Sorting networks**: Multi-layer mandatory even for N=4. Too many parameters for a tight result with current approach.

**DNA codon translation**: No state propagation. Pure 64-entry MLP lookup. Uninteresting as a transformer challenge.

## Recommended Roadmap

### Phase 1: Validation (1-2 weeks)

Prove the approach generalizes. Two experiments:

1. **Binary addition**: Change `log(10)` to `log(2)`, reduce vocab to 4, train. Target: <100 params, 99%+ accuracy on 16-bit addition. This is the minimum viable experiment -- if binary addition fails, the approach doesn't generalize.

2. **Caesar cipher**: New mask (simpler than addition), mod-26 MLP. Target: <120 params, 100% accuracy. This proves the approach works for non-arithmetic tasks without state propagation.

Both experiments can reuse the existing training infrastructure (`train_adder.py`) with minor modifications to `build_fixed_mask()` and `generate_batch()`. No refactoring needed yet.

**Bonus**: Run the zero-shot 20-digit addition test on the existing `submission_trained.py` checkpoint. This costs zero training time and answers a fundamental question about generalization.

### Phase 2: Novel Results (2-4 weeks)

Produce publishable findings on new domains:

3. **Base conversion (binary-to-decimal)**: The ALiBi prefix sum with `log(2)` slopes computes the binary value; the MLP extracts decimal digits via multi-valued mod-10 + floor. This is the most interesting Phase 2 target because it extends the core technique to richer state.

4. **Game theory strategies**: Train TFT, Pavlov, Grudger as separate models. Then attempt a unified multi-strategy model. This tests whether fixed masks can encode parameterized families of state machines.

5. **Subtraction**: Same mask as addition, different trained weights. Test the dual-operation model (addition + subtraction with operation token).

6. **CRC-4**: Tackle the XOR-in-ReLU challenge. CRC-4 on 4-bit messages is small enough to be tractable. Success here opens the door to general bitwise computation.

### Phase 3: Ambitious Targets (1-3 months)

Establish "tiny purpose-built transformers" as a research direction:

7. **DFA simulation**: Fixed-DFA variant first (e.g., "divisibility by 3" checker). Then attempt variable-DFA with transition table in the input sequence.

8. **Multi-operand addition**: Extend to 3+ operands with generalized mask. Map the parameter scaling curve.

9. **Multiply by constant**: Map the carry-complexity vs. parameter count curve across multipliers 2-9.

10. **Paper**: "Fixed-Mask Transformers for Structured Computation" -- presenting the framework, the toolkit, and results across 5+ domains.

## Codebase Evolution Strategy

The current codebase is purpose-built for decimal addition. Supporting multiple domains requires refactoring, but the refactoring should follow the experiments, not precede them.

### Stage 1: Ad Hoc Extensions (Phase 1)

For binary addition and Caesar cipher, modify `train_adder.py` directly:
- Add `base` and `n_digits` parameters to `build_fixed_mask()`
- Add a `build_caesar_mask()` function alongside
- Add `--domain` flag to training script to switch between data generators

This is quick, dirty, and appropriate for 2 experiments. Don't over-engineer.

### Stage 2: Domain Abstraction (Phase 2)

After 3-4 domains are working, the common patterns will be clear. Extract into a `domains/` directory:

```
domains/
  __init__.py          -- registry and discovery
  base.py              -- abstract Domain class
  addition.py          -- decimal addition (refactored)
  binary_add.py        -- binary addition
  caesar.py            -- Caesar cipher
  base_convert.py      -- base conversion
```

Each domain provides: `build_mask()`, `generate_batch()`, `encode()`, `decode()`, `verify()`. The training loop, optimizer, two-phase schedule, and export logic are domain-independent and stay in a unified `train.py`.

### Stage 3: Framework (Phase 3)

If DFA simulation works, the framework generalizes further:
- Mask generator: `generate_mask(domain_type, **params)` produces masks from domain specs
- Config system: `CONFIGS` dict grows a `task` sub-dict capturing `{domain, base, n_digits, n_operands, ...}`
- Inference engine: generalize `infer.py`'s encode/decode for arbitrary domains
- Verification suite: domain-aware `verify.py` with per-domain test generators

The key principle: **refactor after discovery, not before**. The right abstractions emerge from implementing 3-4 concrete domains. Premature abstraction (designing the `Domain` ABC before implementing binary addition) risks encoding the wrong boundaries.

### Parameterized Mask Function

The first concrete refactoring step is a parameterized `build_fixed_mask()` that covers all arithmetic variations without code duplication:

```python
def build_fixed_mask(
    n_digits: int = 10,       # digits per operand
    n_operands: int = 2,      # number of operands
    base: int = 10,           # number base (changes ALiBi slope)
    operation: str = "add",   # "add", "sub", "mul_const", "compare"
    max_seq: int = 35,
    mask_type: str = "slopes",
    anchor: float | None = None,
):
    log_base = math.log(base)
    input_len = n_operands * n_digits + n_operands
    answer_len = n_digits + 1 if operation != "compare" else 1
    ...
```

This covers variations 1-5 from the existing variations report. Comparison and sorting need separate mask builders. Non-arithmetic domains (Caesar, DFA, game theory) need their own mask functions entirely.

## Open Questions

**Is position-determined routing the fundamental requirement?**

The fixed mask works because the attention routing pattern is determined by position, not content. Output digit k always attends to input positions A_k, B_k, and preceding positions -- regardless of what values those positions hold. For problems where routing depends on content (e.g., "attend to the matching bracket" or "attend to the most recently seen keyword"), the mask cannot be pre-computed. The approach's scope is thus: problems where WHAT to compute depends on the input, but WHERE to look is fixed by the task structure.

This is a meaningful class of problems. Most digit-by-digit arithmetic, fixed-structure ciphers, DFA simulation with known graph topology, and lookup tables all have position-determined routing.

**Can the ALiBi trick compute non-geometric weightings?**

The trick produces softmax weights proportional to `B^(j-k)` -- a geometric (exponential) decay. For state propagation that requires non-geometric weights (e.g., linear decay, or arbitrary per-position weights), the approach needs modification. Two possibilities:
- Multiple heads with different slopes can approximate non-geometric profiles via weighted combination
- O_proj can apply a per-position correction to the raw prefix sum

The deeper constraint: softmax weights are always positive and sum to ~1 (modulo anchoring). Computations requiring negative coefficients in the scan cannot use the ALiBi trick directly.

**What is the theoretical parameter floor for each domain?**

The parameter count decomposes into:
- Embedding: `vocab * d_model` (information representation)
- Attention projections: `4 * d_model^2` (routing implementation)
- MLP: `2 * d_model * d_ff + d_ff` (per-position computation)
- LM head: `vocab * d_model + vocab` (output decoding)

The MLP is the variable component. For binary carry: 2 neuron pairs suffice (4 neurons). For N-valued state: N-1 neuron pairs (2N-2 neurons). For a K-entry lookup table: O(K) neurons. The MLP width sets the parameter floor for each domain.

Binary addition with d_model=2, d_ff=4, vocab=4: `4*2 + 4*4 + 2*4*4+4 + 4*2+4 = 8+16+36+12 = 72` params. This may be close to the absolute minimum for a non-trivial autoregressive task with propagating state.

**Is there a meta-learning path?**

Three levels of generality:
1. **Per-domain hand-coding** (current): Design a mask for each domain by understanding the algorithm. This is how AdderBoard works today.
2. **Parameterized mask families**: A function that generates masks from domain parameters (base, operand count, state size). Covers arithmetic domains and DFAs but requires knowing which family applies.
3. **Learned mask generation**: A neural network that takes a domain specification (grammar, state machine, equations) and outputs a mask. This is program synthesis -- speculative but connects to architecture search research.

Level 2 is achievable in the near term. The parameterized `build_fixed_mask()` is already Level 2 for arithmetic. Extending it to DFA simulation (generate mask from transition graph) is straightforward. Level 3 is a research direction, not an engineering task.

## Conclusion

The AdderBoard's 162-parameter adder is not an endpoint -- it's a proof of concept for a general methodology. The fixed-mask hybrid approach decomposes structured computation into routing (mask), value extraction (attention), per-position logic (MLP), and decoding (LM head). Each component transfers independently to new domains.

The near-term path is clear: validate on binary addition and Caesar cipher (hours), produce novel results on base conversion and game theory (weeks), then attempt the general case with DFA simulation (months). The codebase should evolve incrementally -- ad hoc extensions first, then domain abstraction once patterns stabilize.

The most exciting open question is not which specific domain to tackle next, but whether the fixed-mask approach defines a useful computational model: a class of "purpose-built" transformers where human knowledge of the algorithm provides the routing, and gradient descent provides the numerical implementation. If DFA simulation works, this model is Turing-complete (for bounded computation) with remarkably few parameters.
