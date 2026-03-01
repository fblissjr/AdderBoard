# New Domain Exploration

Last updated: 2026-03-01

## Introduction

The AdderBoard project produced a 162-parameter trained transformer that adds two 10-digit numbers at 100% accuracy. The core innovation -- a fixed-mask hybrid where hand-coded attention routing (0 learnable params) combines with trained value/output/MLP/head weights (162 learnable params) -- raises an obvious question: what else can this approach do?

Not every problem is suitable. The fixed-mask approach succeeds on addition because the task has specific structural properties that align with what small transformers can express. This report identifies those properties, evaluates 10 candidate domains against them, and proposes a phased implementation plan.

A domain is a good fit for this approach when the problem's computational structure can be decomposed into: (1) a routing pattern expressible as a fixed attention mask, and (2) a value computation small enough for a tiny MLP. The 7 criteria below formalize this intuition.

## Suitability Criteria

### 1. Deterministic

Output is uniquely determined by input. No randomness, no ambiguity, no multiple valid answers. The model must converge to a single correct function -- stochastic targets create irreducible loss that prevents 100% accuracy with a tiny model. Addition satisfies this trivially: `a + b` has exactly one answer.

### 2. Short sequences

Input + output fits in roughly 50-100 tokens. The fixed mask is `O(H * L^2)` in memory and must be pre-computed for all positions. At 33 tokens, addition's mask is `3 * 33 * 33 = 3267` entries. Beyond ~100 tokens, the mask becomes unwieldy and the attention patterns more complex. Shorter sequences also mean faster training and simpler position-dependent routing.

### 3. Propagating state

The computation involves state that flows across output positions. This is what makes the problem non-trivial for a single-layer model: each output depends not just on local input but on accumulated state from prior positions. In addition, this is the carry. Without propagating state, the problem decomposes into independent per-position lookups, which are trivially parallelizable but uninteresting as a transformer challenge.

### 4. Known structure

We understand the algorithm well enough to design a fixed mask by hand. This is the fundamental requirement of the hybrid approach: the mask encodes WHERE to attend, and must be correct for the algorithm to work. We don't need to know the exact numerical values (those are trained), but we need to know the routing topology. For addition, we know that output digit k needs digits A_k, B_k (digit pair) and A_0..A_{k-1}, B_0..B_{k-1} (carry prefix).

### 5. Small vocabulary

Vocab size < 50. The embedding table is `vocab * d_model` parameters, and the LM head is `vocab * d_model + vocab` (with bias). At d_model=3, each vocab entry costs 3 (embed) + 4 (head) = 7 parameters. Addition uses vocab 12 (84 params in embed+head). At vocab 50, this alone would be 350 params, dominating the model.

### 6. Weighted-sum state

State can be computed via a weighted prefix sum, enabling the ALiBi trick. This is the key insight from addition: attention scores `-(k-j)*log(10)` produce softmax weights proportional to `10^(j-k)`, giving prefix sum `S_k = sum((A_j + B_j) * 10^(j-k))`. The floor of this sum is the carry. If a domain's propagating state can be expressed as a weighted sum over prior positions with geometrically-decaying weights, the ALiBi mask directly applies. This criterion is the most restrictive and the most powerful.

### 7. Local computation

Each output position depends on nearby inputs plus propagated state, not on all inputs simultaneously. In addition, output digit k depends on A_k, B_k (local) and carry_k (propagated). The MLP only needs to compute `(A_k + B_k + carry_k) mod 10` and the carry-out -- a function of 3 scalars. If an output position required attending to all N input positions with equal weight, the attention pattern would not compress into a sparse mask, and the MLP would need to process a high-dimensional aggregation.

## Candidate Domains

### 1. BPE Tokenizer Simulation

**What**: Given a character sequence, produce the sequence of BPE token IDs after greedy merge.

**Architecture mapping**:
- Input: character sequence (e.g., `h e l l o w o r l d`), vocab = 256 byte values + special tokens
- Output: BPE token IDs (variable length, depends on merge table)
- Sequence: `chars SEP token_ids`

**Techniques that transfer**:
- Fixed mask could encode merge priority (which adjacent pairs to merge first)
- Multi-layer attention could implement iterative merging rounds

**Suitability**: 3/7
- Deterministic: Yes (greedy BPE is deterministic for a fixed merge table)
- Short sequences: Marginal (input can be long, output varies)
- Propagating state: Yes, but complex (merge decisions cascade)
- Known structure: Yes (BPE algorithm is well-defined)
- Small vocabulary: No (256+ byte values, potentially thousands of merge tokens)
- Weighted-sum state: No (merge decisions are discrete, not weighted sums)
- Local computation: No (a merge at position i changes what's adjacent at position i+1)

**Feasibility assessment**: Low. The cascading nature of BPE merges means each merge changes the context for subsequent merges. This requires multiple passes (layers), and the state is not a simple scalar but a variable-length list of active spans. The large vocabulary is also a problem.

**Estimated parameters**: 1000+ (multi-layer, large vocab)

**Research angle**: Can a transformer learn the BPE merge table? More precisely, can it learn to simulate an iterative rewrite system? This connects to the broader question of whether transformers can learn string rewriting.

### 2. CRC/Checksum Computation

**What**: Given a byte sequence, compute a CRC checksum (CRC-4, CRC-8, or CRC-16).

**Architecture mapping**:
- Input: message bits (e.g., 8-16 bits), vocab = {0, 1, SEP}
- Output: checksum bits (4-16 bits)
- Sequence: `message_bits SEP checksum_bits`

**Techniques that transfer**:
- Fixed mask: output bit k attends to all message bits (polynomial division is a scan)
- State propagation: CRC state flows left-to-right as XOR with polynomial
- Residual cancellation: useful for isolating CRC state from raw bits

**Key challenge**: XOR is the fundamental operation. ReLU cannot compute XOR directly because XOR is not linearly separable by a single hyperplane. However, XOR of two binary inputs can be decomposed:

```python
# XOR(a, b) = a + b - 2*min(a, b) = a + b - 2*a*b (for binary a, b)
# In ReLU: XOR(a, b) = ReLU(a + b - 0.5) - ReLU(a + b - 1.5)
# This gives 1 when a+b=1, 0 when a+b=0 or a+b=2
```

This requires 2 ReLU neurons per XOR gate. CRC-4 over an 8-bit message requires ~32 XOR operations in the polynomial division, meaning ~64 ReLU neurons minimum. At d_ff=64, d_model=3, this is already 64*3 + 64 + 3*64 = 448 MLP params alone.

**Suitability**: 5/7
- Deterministic: Yes
- Short sequences: Yes (8-bit message + 4-bit CRC = ~15 tokens)
- Propagating state: Yes (CRC register propagates across message bits)
- Known structure: Yes (polynomial division is well-defined)
- Small vocabulary: Yes (binary + SEP = 3 tokens)
- Weighted-sum state: No (XOR is not a weighted sum)
- Local computation: Marginal (each CRC bit depends on all message bits via polynomial)

**Feasibility assessment**: Medium. The XOR-in-ReLU decomposition works in principle, but the MLP size scales with message_length * polynomial_degree. CRC-4 on 4-bit messages might work as a proof of concept with ~200 params.

**Estimated parameters**: 200-500 (CRC-4 on short messages)

**Research angle**: Can ReLU networks learn XOR-based algorithms efficiently? This is a fundamental question about the computational expressiveness of piecewise-linear functions. The CRC domain provides a clean testbed because the algorithm is pure XOR + shift.

### 3. Caesar/Vigenere Cipher

**What**: Given plaintext and a key, produce ciphertext. Caesar shifts all characters by a constant; Vigenere shifts each character by a key-dependent amount.

**Architecture mapping**:
- Input: `key plaintext`, vocab = 26 letters + SEP + key digits (0-25) = ~28 tokens
- Output: ciphertext (same length as plaintext)
- Caesar: `k SEP a b c SEP2 d e f` (key k, plaintext abc, ciphertext def)
- Vigenere: `k0 k1 k2 SEP a b c SEP2 d e f` (key k0k1k2, plaintext abc)

**Techniques that transfer**:
- **Parabolic LM head**: `logit[c] = 2cx - c^2` peaks at c=x. For mod-26, this gives exact integer decoding. Directly applicable.
- **Fixed mask**: Each output position attends to (a) corresponding plaintext character and (b) corresponding key character. Simple, sparse, no prefix sum needed.
- Residual cancellation: useful for clean MLP input.

**Key insight**: Caesar cipher is `(p + k) mod 26`. The MLP must compute modular addition, which is exactly what the AdderBoard MLP already does (mod 10). The only change is the modulus. With the parabolic LM head trick, we need:

```python
# MLP computes: sum = p + k (from attention)
# If sum >= 26: subtract 26
# This is a single ReLU threshold: ReLU(sum - 25.5) * (-26) + sum
# Needs 2 MLP neurons: one for identity pass-through, one for overflow
```

No propagating state needed. Each output character is independent.

**Suitability**: 6/7
- Deterministic: Yes
- Short sequences: Yes (~30-50 tokens)
- Propagating state: No (each output is independent) -- this is the missing criterion
- Known structure: Yes
- Small vocabulary: Yes (28 tokens)
- Weighted-sum state: N/A (no state to propagate)
- Local computation: Yes (each output depends on one plaintext + one key character)

**Feasibility assessment**: High. This is essentially a 1D version of addition without carry. The mask is simpler (no prefix sum head needed), the MLP is simpler (mod 26 instead of mod 10 + carry), and the LM head technique transfers directly.

**Estimated parameters**: 80-120 (simpler than addition -- no carry head needed, d_ff=2-4 suffices)

**Research angle**: What is the minimum model for correct mod-26 arithmetic? Since there's no carry propagation, the model can theoretically be even smaller than the adder. This also serves as a stepping stone to Vigenere, which adds periodic key indexing (a simple form of positional dependence).

### 4. Base Conversion

**What**: Given a number in base A, produce the same number in base B. Start with binary-to-decimal (base 2 to base 10) or decimal-to-binary.

**Architecture mapping**:
- Binary to decimal: input is binary digits (LSB-first), output is decimal digits (LSB-first)
- Sequence: `b0 b1 b2 ... b_n SEP d0 d1 ... d_m`
- Vocab: {0, 1, 2, ..., 9, SEP} = 11 tokens (for binary-to-decimal)

**Techniques that transfer**:
- **ALiBi prefix sum**: The value of a binary number is `sum(b_j * 2^j)`. Using attention scores `-(k-j)*log(2)`, softmax weights are proportional to `2^(j-k)`, giving a weighted prefix sum. This directly extends the AdderBoard technique, replacing `log(10)` with `log(2)`.
- **Fixed mask**: Output decimal digit k attends to all binary input digits with appropriate ALiBi slopes.
- **MLP mod-10 + carry**: Same as addition -- extract the k-th decimal digit via mod 10.

**Key challenge**: The state is not a single carry bit but a multi-valued remainder. For binary-to-decimal, the decimal digit at position k is `floor(value / 10^k) mod 10`. The "carry" into position k is `floor(value / 10^k)`, which can be any non-negative integer, not just 0 or 1. However, the prefix sum already encodes this: `S_k = sum(b_j * 2^j) / 10^k`, and `floor(S_k) mod 10` is the digit. The MLP must extract the ones digit of `floor(S_k)`, which requires more complex thresholding than binary carry.

For binary-to-decimal with N binary digits: the maximum value is `2^N - 1`, which has `ceil(N * log10(2))` decimal digits. For N=10 binary digits, max value is 1023, needing 4 decimal digits. Sequence length: 10 + 1 + 4 = 15 tokens.

```python
def build_bin2dec_mask(n_bits=10, n_dec=4, max_seq=16):
    """Fixed mask for binary-to-decimal conversion.

    Head 0: Each decimal output attends to all binary input digits
            with ALiBi slopes -(k*log10 - j*log2) for weighted sum.
    Head 1: Self-attend (residual management).
    Head 2: Available for additional routing if needed.
    """
    M = torch.full((3, max_seq, max_seq), float('-inf'))
    sep_pos = n_bits  # SEP position

    for q in range(max_seq):
        if q <= sep_pos:
            # Input positions: self-attend only
            for h in range(3):
                M[h, q, q] = 0.0
        else:
            k = q - sep_pos - 1  # decimal digit index (0 = ones)
            # Head 0: attend to all binary digits with ALiBi slopes
            for j in range(n_bits):
                # Weight for bit j contributing to decimal digit k:
                # b_j * 2^j / 10^k -> score = j*log(2) - k*log(10)
                score = j * math.log(2) - k * math.log(10)
                M[0, q, j] = score
            # Head 1: self-attend
            M[1, q, q] = 0.0
            # Head 2: self-attend
            M[2, q, q] = 0.0
    return M
```

**Suitability**: 5/7
- Deterministic: Yes
- Short sequences: Yes (15-30 tokens depending on base sizes)
- Propagating state: Yes (remainder from higher-order digits)
- Known structure: Yes (positional numeral system conversion is well-defined)
- Small vocabulary: Yes (max(A, B) + separators)
- Weighted-sum state: Yes (prefix sum with `log(base)` slopes)
- Local computation: No (each output digit depends on ALL input digits)

**Feasibility assessment**: Medium-high. The ALiBi prefix sum generalizes naturally. The main challenge is that the MLP must extract `floor(S_k) mod 10` from a potentially large prefix sum, requiring more thresholding neurons than binary carry detection. Binary-to-decimal with 10-bit inputs is the best starting point.

**Estimated parameters**: 150-250 (similar to addition, possibly larger MLP for multi-valued state)

**Research angle**: How does state complexity affect parameter count? Binary carry is a 1-bit state (0 or 1). Base conversion has a multi-valued state (remainder mod B). This is a controlled experiment in state complexity: same ALiBi technique, same mask structure, but richer state. Does parameter count scale linearly with state range?

### 5. Binary Arithmetic (Addition, Subtraction, AND, OR, XOR)

**What**: Perform arithmetic and logical operations on binary strings.

**Architecture mapping**:
- Input: two binary strings (LSB-first), each N bits
- Output: result (N or N+1 bits depending on operation)
- Sequence: `a0..a_N SEP b0..b_N SEP2 r0..r_M`
- Vocab: {0, 1, SEP, SEP2} = 4 tokens

**Operations and their mask requirements**:

**Binary addition**: Direct adaptation of decimal addition. Replace `log(10)` with `log(2)` in ALiBi slopes. The carry is still a prefix sum: `carry_k = floor(sum((a_j + b_j) * 2^(j-k)))`. Same 3-head mask structure.

```python
# Only change from build_fixed_mask():
score = -(k - j) * math.log(2)  # was math.log(10)
```

**AND/OR**: No state propagation. Each output bit depends only on the corresponding input bits. Mask: Head 0 attends to a_k and b_k. MLP computes AND or OR via single ReLU threshold. Trivially simple.

**XOR**: Same input locality as AND/OR, but XOR requires 2 ReLU neurons per bit (as discussed in CRC section). Still no state propagation.

**Binary subtraction**: Equivalent to addition with borrow. Borrow propagation is similar to carry: `borrow_k = floor(sum((a_j - b_j) * 2^(j-k)))` when negative. Slightly more complex because the borrow condition depends on sign.

**Suitability**: 6/7 (for addition)
- Deterministic: Yes
- Short sequences: Yes (e.g., 16 + 1 + 16 + 1 + 17 = 51 tokens for 16-bit)
- Propagating state: Yes (carry for addition, none for AND/OR)
- Known structure: Yes
- Small vocabulary: Yes (4 tokens)
- Weighted-sum state: Yes (carry as prefix sum with log(2))
- Local computation: Yes (each bit depends on a_k, b_k, carry_k)

**Feasibility assessment**: Very high. Binary addition is the simplest possible adaptation of the existing system. The vocabulary shrinks from 12 to 4, saving 24 embedding + 32 head = 56 params. The ALiBi slopes change by one constant. This should converge with fewer parameters than decimal addition.

**Estimated parameters**: 80-120 (smaller vocab, simpler mod-2 vs mod-10)

**Research angle**: What is the absolute minimum parameter count for binary addition? With vocab 4 and mod-2 arithmetic (no multi-threshold needed in MLP), this may achieve the theoretical floor for a trained autoregressive transformer on a non-trivial sequence task with propagating state.

### 6. Prisoner's Dilemma / Game Theory

**What**: Given a history of moves in an iterated Prisoner's Dilemma, predict the next move according to a fixed strategy.

**Architecture mapping**:
- Input: interleaved history of moves: `p0 o0 p1 o1 ... p_n o_n` (player, opponent)
- Output: next move (C=cooperate, D=defect)
- Vocab: {C, D, SEP} = 3 tokens
- Sequence length: 2*N + 1 (history) + 1 (SEP) + 1 (prediction) for N rounds

**Strategies as state machines**:

| Strategy | State | Transition | Output |
|---|---|---|---|
| Tit-for-Tat (TFT) | last opponent move | s = o_{n-1} | C if s=C, D if s=D |
| Generous TFT | last opponent move | s = o_{n-1} | C if s=C, C w.p. 0.1 if s=D |
| Pavlov (Win-Stay-Lose-Switch) | last outcome | s = (p_{n-1}=o_{n-1}) | C if s=true, D if s=false |
| Grudger | ever_defected flag | s = OR(o_0, ..., o_n) | C if s=false, D if s=true |

**Techniques that transfer**:
- **Fixed mask**: Output position attends to relevant history positions. For TFT: attend to last opponent move only. For Grudger: attend to all opponent moves.
- **ALiBi slopes**: For Grudger, use steep slopes so any defection dominates the prefix sum.
- Residual cancellation: for clean MLP input.

**Key insight**: Each strategy is a tiny state machine with 1-2 states. The state transition is a simple function of recent history. TFT literally copies the last opponent move -- the model just needs to attend to position `n-1` and pass it through. Grudger requires detecting if any opponent move was D, which is a max/OR over the history -- expressible as a prefix sum with steep slopes where any D contribution dominates.

**Suitability**: 5/7
- Deterministic: Yes (for deterministic strategies; Generous TFT is stochastic, exclude it)
- Short sequences: Yes (~20-50 tokens for typical game lengths)
- Propagating state: Yes (Grudger, Pavlov) or No (TFT is memoryless w.r.t. state)
- Known structure: Yes (strategies are explicitly defined)
- Small vocabulary: Yes (3 tokens)
- Weighted-sum state: Partial (Grudger's OR can be approximated as max via steep prefix sum)
- Local computation: Yes (most strategies depend on last 1-2 moves + possibly a running flag)

**Feasibility assessment**: High for simple strategies (TFT, Pavlov). Medium for Grudger (requires OR-over-history via attention). The interesting extension is training one model that can implement multiple strategies based on a strategy token prefix.

**Estimated parameters**: 50-100 (tiny vocab, simple state machines)

**Research angle**: Can one model learn multiple strategies via different fixed masks? Or can a single mask with a strategy-selector token route to different behaviors? This connects to the broader question of how transformers represent finite automata, and whether the fixed-mask approach can encode parameterized families of state machines.

### 7. State Machine (DFA) Simulation

**What**: Given a DFA specification and an input string, produce the accept/reject decision or the full state trace.

**Architecture mapping** (state trace variant):
- Input: DFA encoded as transition table + input string
- Encoding: `state_0 input_0 state_1 input_1 ... state_n input_n SEP accept/reject`
- Alternative: separate DFA encoding prefix + input string + output states
- Vocab: state symbols (|Q|) + input symbols (|Sigma|) + SEP = |Q| + |Sigma| + 1

For a small DFA (3 states, binary alphabet):
- Vocab: {q0, q1, q2, a, b, SEP, ACC, REJ} = 8 tokens
- Input: `a b a b a` (5 symbols)
- Output: `q0 q1 q2 q1 q2 q1` (6 states including initial) or just `ACC/REJ`

**Techniques that transfer**:
- **Fixed mask**: Each output state position attends to (a) the previous state and (b) the current input symbol. This is a simple 2-position attention pattern.
- Residual cancellation: for clean MLP input.
- Parabolic LM head: for state decoding.

**Key challenge**: The state transition function `delta(q, a) -> q'` is an arbitrary lookup table, not a smooth arithmetic function. For 3 states and 2 input symbols, there are 6 entries in the transition table. The MLP must implement this lookup. With ReLU, each entry requires detecting a specific (state, input) pair and outputting the correct next state. This is `O(|Q| * |Sigma|)` neurons minimum.

For a DFA with |Q|=3, |Sigma|=2: 6 transitions, each needing ~2 ReLU neurons for detection + 1 for output = ~18 neurons. At d_model=3: 18*3 + 18 + 3*18 = 126 MLP params.

**Problem**: The DFA is part of the INPUT, not fixed. So the mask can't encode the transition table -- it changes per example. Two approaches:
1. **Fixed DFA, variable input**: The transition table is baked into the trained weights. Each model instance solves one specific DFA. The mask handles routing (attend to prev state + current input), the MLP learns the transition table.
2. **Variable DFA, variable input**: The DFA spec is part of the input sequence. The model must read the transition table from the input and apply it. This requires cross-attention between the DFA-spec prefix and the execution trace, which is substantially harder.

Option 1 is feasible and interesting. Option 2 is a longer-term research target.

**Suitability**: 5/7 (for fixed-DFA variant)
- Deterministic: Yes
- Short sequences: Yes (for small DFAs and short inputs)
- Propagating state: Yes (current DFA state propagates across input symbols)
- Known structure: Yes (DFA simulation is well-defined)
- Small vocabulary: Yes (|Q| + |Sigma| + separators)
- Weighted-sum state: No (state transitions are discrete lookup, not weighted sums)
- Local computation: Yes (next state depends only on current state + current input)

**Feasibility assessment**: Medium for fixed-DFA. The mask is simple (attend to prev state + current input), but the MLP must learn an arbitrary transition table. For a 3-state binary DFA, this is tractable. For larger DFAs, the MLP size grows quadratically.

**Estimated parameters**: 150-300 (depends on |Q| * |Sigma|)

**Research angle**: Can a tiny transformer simulate arbitrary small DFAs? This is the most general candidate because many problems reduce to DFA simulation (regex matching, protocol verification, simple parsers). If the fixed-mask approach can handle arbitrary 3-state DFAs, it establishes a general-purpose framework. The deeper question: is there a universal mask that works for ALL DFAs of a given size, with only the MLP weights changing?

### 8. Regex Matching

**What**: Given a regex pattern and a string, determine if the string matches the pattern (and optionally, the match positions).

**Architecture mapping**:
- Input: `pattern SEP string SEP2 match_result`
- For simple patterns: `a*b SEP aab SEP2 1` (match=1, no match=0)
- Vocab: alphabet + regex metacharacters (*, +, ?, ., |, (, )) + SEP + {0, 1}

**Relation to DFA simulation**: Any regex can be compiled to a DFA, and the matching problem reduces to DFA simulation. The compilation step (regex -> NFA -> DFA) is itself a complex transformation that would be hard to encode in a fixed mask. However, if we fix the regex and only vary the input string, this is exactly the fixed-DFA variant from Domain 7.

**Direct approach challenges**:
- Variable-length patterns make the mask structure input-dependent
- Backtracking (for non-DFA-compilable features like backreferences) requires multiple passes
- Metacharacter semantics (*, +, ?) are context-dependent

**Suitability**: 4/7
- Deterministic: Yes (for standard regex without ambiguity)
- Short sequences: Marginal (pattern + string + result)
- Propagating state: Yes (NFA state set propagates across string)
- Known structure: Yes (regex engines are well-understood)
- Small vocabulary: Marginal (alphabet + metacharacters can exceed 40)
- Weighted-sum state: No
- Local computation: No (match depends on entire pattern structure)

**Feasibility assessment**: Low for general regex. High for fixed-pattern matching (reduces to DFA simulation). Better to approach via Domain 7 (DFA) than to tackle regex directly.

**Estimated parameters**: 200-500 for fixed simple patterns; impractical for general regex

**Research angle**: Is there a simpler encoding than full DFA for common regex patterns? Some patterns (e.g., `a*b`, `(ab)+`) have regular structure that might allow a compact mask. The question is whether regex-specific structure can be exploited beyond what DFA simulation provides.

### 9. Sorting Networks

**What**: Given a sequence of N numbers, produce the sorted sequence.

**Architecture mapping**:
- Input: `x0 x1 ... x_{N-1} SEP` (N numbers)
- Output: sorted sequence `y0 y1 ... y_{N-1}`
- For small numbers (0-9): vocab = {0-9, SEP} = 11

**Sorting network structure**: A sorting network is a fixed sequence of compare-and-swap operations. For N=4 (bitonic sort):
- Layer 1: compare-swap (0,1), (2,3) -- 2 parallel comparisons
- Layer 2: compare-swap (0,2), (1,3)
- Layer 3: compare-swap (1,2)

Each compare-and-swap requires: (a) attending to both elements, (b) computing min and max, (c) outputting the correct one based on position. In a transformer, each layer can implement one round of parallel comparisons via attention heads.

**Key challenge**: Even the simplest non-trivial sort (N=4) requires 3 layers of comparisons. With 1 transformer layer, we can do at most 1 round of comparisons per head. A 3-head, 1-layer model could do 3 parallel comparisons, but sorting networks require sequential rounds (the output of round 1 feeds into round 2).

Multi-layer approach: a 3-layer transformer with 2 heads each could implement a 4-element sorting network, but parameter count grows linearly with layers: ~3 * (Q+K+V+O+MLP) = ~3 * 100 = 300+ params.

**Min/max in ReLU**:
```python
# min(a, b) = (a + b - |a - b|) / 2 = (a + b - ReLU(a-b) - ReLU(b-a)) / 2
# max(a, b) = (a + b + |a - b|) / 2
# Each compare-swap needs 4 ReLU neurons
```

**Suitability**: 3/7
- Deterministic: Yes
- Short sequences: Yes (for N <= 8)
- Propagating state: No (sorting network has fixed dataflow, no sequential state)
- Known structure: Yes (optimal sorting networks are known for small N)
- Small vocabulary: Yes
- Weighted-sum state: No (comparisons are non-linear, not weighted sums)
- Local computation: No (each output can depend on any input element)

**Feasibility assessment**: Low for 1-layer. The fundamental issue is that sorting inherently requires O(log^2 N) sequential rounds of comparisons (for sorting networks) or O(N log N) sequential comparisons (for comparison sorts). A 1-layer transformer can only do 1 round. Even for N=4, we need 3 layers minimum.

**Estimated parameters**: 300+ (multi-layer mandatory for N >= 4)

**Research angle**: What is the minimum transformer depth for sorting N elements? This is a clean theoretical question: sorting networks provide a lower bound on depth (3 for N=4, 5 for N=8), and each transformer layer can simulate one round. The question is whether transformer layers are more or less powerful than sorting network rounds (they have MLP non-linearity, which might help).

### 10. DNA Codon Translation

**What**: Given a DNA sequence (groups of 3 nucleotides), produce the corresponding amino acid sequence.

**Architecture mapping**:
- Input: DNA nucleotides in reading frame: `A T G C A G ... SEP`
- Output: amino acid sequence: `M Q ...` (Met, Gln, ...)
- Vocab: {A, C, G, T} + 20 amino acids + Stop + SEP = 26 tokens
- Each output position depends on exactly 3 input positions (the codon)

**Techniques that transfer**:
- **Fixed mask**: Output position k attends to input positions 3k, 3k+1, 3k+2. Simple, sparse, fixed-stride.
- Parabolic LM head: for amino acid decoding.

**Computation**: The genetic code is a 64-entry lookup table (4^3 codons -> 20 amino acids + stop). The MLP must implement this lookup. Each codon is a 3-digit base-4 number (0-63), and the output is one of 21 categories.

With d_model=3 and 3 attention heads, each head can attend to one nucleotide of the codon. The attention output is a 3-dimensional vector encoding the codon identity. The MLP must then map this 3D vector to the correct amino acid.

**MLP lookup table size**: 64 entries, each requiring ~2 ReLU neurons for detection (identify the specific codon) and 1 for output contribution. Naive: 128 neurons. But the genetic code has structure (degeneracy: multiple codons map to the same amino acid), so ~40-60 neurons might suffice with clever encoding.

**Suitability**: 4/7
- Deterministic: Yes
- Short sequences: Yes (typical gene: 100-1000 codons, but we can test on short sequences)
- Propagating state: No (each codon is independent)
- Known structure: Yes (standard genetic code is a fixed table)
- Small vocabulary: Yes (26 tokens)
- Weighted-sum state: No (lookup table, not weighted sum)
- Local computation: Yes (each output depends on exactly 3 adjacent inputs)

**Feasibility assessment**: Medium. The mask is trivial. The challenge is entirely in the MLP: can a small ReLU network implement a 64-entry lookup table? This is a well-studied problem in neural network theory. The minimum MLP size for a K-entry lookup with D-dimensional input is O(K) neurons in the worst case, but structured tables (like the genetic code with its degeneracy patterns) can be more compact.

**Estimated parameters**: 300-500 (dominated by MLP size for 64-entry lookup)

**Research angle**: Minimum parameters for a 64-entry lookup table in a ReLU network. The genetic code's degeneracy (e.g., all 4 codons ending in any nucleotide for Leu, Ser, Arg, etc.) provides natural structure that should reduce the required MLP size below the worst-case bound. This connects to the theory of Boolean function complexity in ReLU networks.

## Feasibility vs. Interest Matrix

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

**Quadrant analysis**:

| Quadrant | Domains | Strategy |
|---|---|---|
| High interest, high feasibility (top-right) | Base conversion, DFA simulation | Priority targets. Novel results with manageable effort. |
| High interest, low feasibility (top-left) | BPE tokenizer | Aspirational. Requires breakthroughs in mask design. |
| Low interest, high feasibility (bottom-right) | Binary arithmetic, Caesar/Vigenere | Validation targets. Prove generalization quickly. |
| Medium interest, medium feasibility (center) | CRC, Game theory, Sorting, Regex | Selective pursuit. CRC and Game theory offer the best ROI. |
| Low interest, low feasibility (bottom-left) | DNA codon | Skip unless MLP lookup theory is the specific goal. |

The sweet spot is the top-right quadrant. Base conversion extends the ALiBi technique to a new domain with richer state, and DFA simulation is the most general target. Binary arithmetic and Caesar are necessary validation steps to prove the approach generalizes before attempting harder targets.

## 3-Phase Implementation Roadmap

### Phase 1: Validation (1-2 weeks)

**Goal**: Prove the fixed-mask hybrid approach generalizes beyond base-10 addition.

**Target 1: Binary addition**

Minimal adaptation. Change one constant in `build_fixed_mask()`:

```python
# In build_fixed_mask(), Head 1 carry prefix sum:
score = -(k - j) * math.log(2)  # was math.log(10)
```

Update `generate_batch()` for binary inputs. Reduce vocab from 12 to 4. Expected parameter count: ~80-100 (smaller vocab, d_ff=4 may suffice since mod-2 is simpler than mod-10).

Success criterion: >= 99% accuracy on 16-bit binary addition with < 100 parameters.

**Target 2: Caesar cipher**

New mask (simpler than addition -- no carry head):

```python
def build_caesar_mask(max_seq=55):
    """Fixed mask for Caesar cipher.

    Format: key SEP plaintext SEP2 ciphertext
    key = single digit 0-25 at position 0
    plaintext positions: 2 to 2+N-1
    output positions: 2+N+1 onwards

    Head 0: attend to plaintext char at corresponding position
    Head 1: attend to key (position 0)
    Head 2: self-attend (residual management)
    """
    M = torch.full((3, max_seq, max_seq), float('-inf'))
    key_pos = 0
    sep1 = 1
    text_start = 2
    text_len = 26  # max plaintext length

    for q in range(max_seq):
        out_start = text_start + text_len + 1  # after SEP2
        if q < out_start:
            for h in range(3):
                M[h, q, q] = 0.0
        else:
            k = q - out_start  # output char index
            # Head 0: attend to corresponding plaintext char
            M[0, q, text_start + k] = 0.0
            # Head 1: attend to key
            M[1, q, key_pos] = 0.0
            # Head 2: self-attend
            M[2, q, q] = 0.0
    return M
```

MLP computes `(p + k) mod 26`. Parabolic LM head for letter decoding. Expected parameter count: ~80-120.

Success criterion: 100% accuracy on Caesar cipher for all 26 shifts and arbitrary plaintext.

**Deliverables**:
- `domains/binary_add.py` with mask, data gen, verification
- `domains/caesar.py` with mask, data gen, verification
- Comparative analysis: param count vs. addition for equivalent accuracy

### Phase 2: Novel Results (2-4 weeks)

**Goal**: Publishable results on non-arithmetic tasks.

**Target 3: Base conversion (binary-to-decimal)**

Extends ALiBi prefix sum with `log(2)` slopes instead of `log(10)`. Output digit k must extract `floor(value / 10^k) mod 10`, where value is the binary input's decimal equivalent. The MLP needs more neurons than addition because the "carry" is multi-valued (0-9 remainder, not 0/1).

Implementation: new mask with mixed slopes (`log(2)` for binary input weighting, `log(10)` for decimal output positioning). Training may require larger d_ff (8-12) to handle the multi-valued state extraction.

Success criterion: >= 99% accuracy on 10-bit binary to 4-digit decimal conversion.

**Target 4: Game theory strategies**

Train separate models for TFT, Pavlov, and Grudger. Each has a different mask:
- TFT: attend to last opponent move only
- Pavlov: attend to last own move and last opponent move
- Grudger: attend to all opponent moves with steep ALiBi slopes

Then attempt a unified model with a strategy-selector token that routes to different behaviors via the mask.

Success criterion: 100% accuracy for TFT and Pavlov; >= 99% for Grudger.

**Target 5: CRC-4**

Tackle the XOR-in-ReLU challenge. Start with CRC-4 on 4-bit messages (4 input + 4 output = 8 tokens + separators). The polynomial division requires ~16 XOR operations, each needing 2 ReLU neurons = 32 neurons minimum. Test whether training can find a more efficient encoding.

Success criterion: >= 99% accuracy on CRC-4 for all 4-bit messages.

**Deliverables**:
- `domains/base_convert.py`, `domains/game_theory.py`, `domains/crc.py`
- Parameter count comparison across all domains
- Analysis of which techniques transfer and which don't

### Phase 3: Ambitious Targets (1-3 months)

**Goal**: Establish "tiny purpose-built transformers" as a research direction.

**Target 6: DFA simulation (fixed DFA)**

Pick a specific 3-state binary DFA (e.g., "divisibility by 3" on binary strings). Design the mask, train the MLP to learn the transition table. Then test on increasingly complex DFAs (4-state, 5-state, ternary alphabet).

The key question: is there a universal mask structure that works for all DFAs of a given size? If so, only the MLP weights change per DFA, and we have a general-purpose tiny-transformer compiler.

**Target 7: Multi-operand addition**

Extend from 2-operand to 3-operand addition: `A + B + C`. The carry can now be 0, 1, or 2 (since max digit sum is 9+9+9=27). The mask needs additional routing for the third operand, and the MLP needs to handle carry values > 1.

**Target 8: BPE tokenizer (if DFA approach works)**

BPE merging can be viewed as a sequence of DFA-like state transitions. If the DFA approach from Target 6 scales, attempt to encode BPE merge rules as a multi-layer DFA simulation where each layer performs one round of merges.

**Deliverables**:
- `domains/dfa.py` with configurable DFA specification
- Analysis: parameter count vs. DFA complexity (|Q|, |Sigma|)
- Paper draft: "Fixed-Mask Transformers for Structured Computation"

## Codebase Refactoring Proposal

To support multi-domain experimentation, refactor the current single-purpose training infrastructure into a domain-agnostic framework.

### Directory Structure

```
domains/
  __init__.py          -- registry and discovery
  base.py              -- abstract Domain class
  addition.py          -- current AdderBoard (refactored from train_adder.py)
  binary_add.py        -- binary addition
  caesar.py            -- Caesar cipher
  base_convert.py      -- base conversion (binary-to-decimal, etc.)
  game_theory.py       -- Prisoner's Dilemma strategies
  crc.py               -- CRC-4/8 checksum
  dfa.py               -- DFA simulation
```

### Abstract Domain Class

```python
# domains/base.py
from abc import ABC, abstractmethod
import torch

class Domain(ABC):
    """Abstract base for a tiny-transformer domain.

    Each domain defines:
    - The fixed attention mask (routing topology)
    - Training data generation
    - Input/output encoding
    - Verification harness
    """

    @abstractmethod
    def name(self) -> str:
        """Human-readable domain name."""
        ...

    @abstractmethod
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        ...

    @abstractmethod
    def max_seq_len(self) -> int:
        """Maximum sequence length (input + separators + output)."""
        ...

    @abstractmethod
    def output_start(self) -> int:
        """Index of first output token in the sequence."""
        ...

    @abstractmethod
    def output_len(self) -> int:
        """Number of output tokens."""
        ...

    @abstractmethod
    def build_mask(self, n_heads: int, max_seq: int) -> torch.Tensor:
        """Build fixed attention mask. Shape: (n_heads, max_seq, max_seq).

        Returns a tensor where:
        - 0.0 means "attend to this position"
        - float('-inf') means "do not attend"
        - Intermediate values (e.g., ALiBi slopes) encode weighted attention
        """
        ...

    @abstractmethod
    def generate_batch(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Generate a training batch. Shape: (batch_size, max_seq_len).

        Returns token sequences including input, separators, and ground truth output.
        """
        ...

    @abstractmethod
    def encode(self, *args) -> list[int]:
        """Encode problem instance to token sequence (input portion only)."""
        ...

    @abstractmethod
    def decode(self, tokens: list[int]) -> object:
        """Decode output tokens to human-readable result."""
        ...

    @abstractmethod
    def verify(self, model: torch.nn.Module, n_tests: int = 10000,
               seed: int = 2025) -> float:
        """Run verification suite. Returns accuracy (0.0 to 1.0).

        Must use autoregressive generation (not teacher forcing).
        """
        ...

    def loss_positions(self, seq_len: int) -> slice:
        """Positions to compute loss on (default: output positions).

        Override if loss should be computed on a subset of output positions.
        """
        return slice(self.output_start(), self.output_start() + self.output_len())
```

### Domain Registration

```python
# domains/__init__.py
from domains.base import Domain

_REGISTRY: dict[str, type[Domain]] = {}

def register(name: str):
    def decorator(cls):
        _REGISTRY[name] = cls
        return cls
    return decorator

def get_domain(name: str, **kwargs) -> Domain:
    return _REGISTRY[name](**kwargs)

def list_domains() -> list[str]:
    return list(_REGISTRY.keys())
```

### Unified Training Script

The existing `train_adder.py` would be refactored into a domain-agnostic `train.py`:

```bash
# Train binary addition
uv run python train.py --domain binary_add --config mask-w6 --lr 0.01

# Train Caesar cipher
uv run python train.py --domain caesar --config mask-w4 --lr 0.01

# Train base conversion
uv run python train.py --domain base_convert --config mask-w12 --lr 0.01
```

The training loop, optimizer setup, two-phase schedule, and export logic are domain-independent. Only the mask, data generation, and verification change per domain.

### Migration Path

1. Extract `Domain` ABC and registration into `domains/base.py`
2. Wrap existing `build_fixed_mask()` and `generate_batch()` into `domains/addition.py`
3. Verify the refactored addition domain reproduces existing results
4. Add new domains incrementally, each with its own mask and data generator
5. Keep `train_adder.py` as-is for backward compatibility; new `train.py` uses the domain framework

## Open Questions

**Is the fixed-mask approach fundamentally limited to problems with "carry-like" state propagation?**

The approach requires that we know the attention routing pattern in advance. For problems where the routing depends on the input (e.g., "attend to the matching bracket" in bracket-balancing), the mask cannot be pre-computed. The approach works best when the routing is determined by position alone, not by content. This limits it to problems where each output position has a fixed set of relevant input positions.

However, "carry-like" state is not strictly required. Caesar cipher has no state propagation at all, and it should still work with a fixed mask. The mask's role is routing, not state computation. The more precise limitation is: the problem must have position-determined routing.

**Can the ALiBi prefix sum trick generalize to non-power-weighted state?**

The trick works because `softmax(-(k-j)*log(B))` produces weights proportional to `B^(j-k)`, which computes a base-B weighted sum. For non-geometric weightings (e.g., linear weights, Fibonacci weights), the ALiBi scores would need to be `log(w_j)` for arbitrary weights `w_j`. This works as long as all weights are positive (softmax outputs are always positive) and the weights decay fast enough that the sum converges.

The key constraint: softmax weights are always positive and sum to 1. This means the prefix sum is always a convex combination. If the desired computation requires negative weights or weights summing to something other than 1, the ALiBi trick alone won't work -- but O_proj scaling can compensate for the normalization.

**What is the theoretical minimum model size for each domain?**

This decomposes into three questions:
1. How many bits of routing information does the mask need? (Determines mask complexity, but masks have 0 learnable params.)
2. How many neurons does the MLP need to implement the per-position computation? (Determined by the Boolean/arithmetic complexity of the output function.)
3. How many embedding dimensions are needed to represent the relevant input features? (Determined by the information content per token.)

For binary addition: the per-position computation is `(a + b + carry) mod 2` and `(a + b + carry) / 2`. This is 3 Boolean functions, each requiring O(1) neurons. The embedding needs to distinguish 4 tokens (2 bits). Theoretical minimum: ~30-50 params.

For DFA simulation: the per-position computation is a lookup table of size |Q| * |Sigma|. Each entry needs O(1) neurons for detection and O(1) for output. Theoretical minimum: O(|Q| * |Sigma| * d_model) params.

**Is there a meta-learning approach: train a mask generator from domain specification?**

Given a formal specification of a domain (e.g., a grammar, a state machine diagram, or a set of equations), can we automatically generate the fixed mask? This would transform the approach from "hand-code a mask per domain" to "specify the domain, get a mask."

For DFA-based domains, this is straightforward: the mask structure is always "attend to previous state + current input," and only the MLP weights change. For arithmetic domains, the mask depends on the base and number of operands, but follows a pattern (ALiBi slopes = log(base)).

A mask generator could be as simple as:
```python
def generate_mask(domain_type, **params):
    if domain_type == "positional_arithmetic":
        return build_alibi_mask(base=params["base"], n_operands=params["n_operands"])
    elif domain_type == "dfa":
        return build_dfa_mask(n_states=params["n_states"])
    elif domain_type == "local_lookup":
        return build_local_mask(window=params["window_size"])
```

The deeper question is whether there's a learned mask generator -- a neural network that takes a domain specification and outputs a mask. This would be a form of program synthesis: learning to design attention patterns. This is speculative but connects to recent work on learned optimizers and architecture search.
