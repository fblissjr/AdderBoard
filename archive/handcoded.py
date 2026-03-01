"""
Hand-coded transformer adder (Track 1).

Architecture: 1L decoder, d_model=4, 2 heads (d_head=2), ff=4
Input: reversed digits LSB-first: a0..a9 SEP b0..b9 SEP (22 tokens)
Output: 11 sum digits LSB-first, autoregressive

Vocab: 0-9 digits, 10=SEP
Total sequence during generation of s_i: 22+i tokens.
Logits taken from last position (21+i) to predict s_i.

Key math:
- Head 1 gathers A_i + B_i into dim 1 via fixed attention bias
- Head 2 gathers A_{i-1} + B_{i-1} into dim 2 via fixed attention bias
- Current token (s_{i-1}) value lives in dim 0 from embedding
- MLP computes: carry = step(dim2 - dim0 >= 9), then result = (dim1 + carry) mod 10
- Parabolic LM head decodes result to digit
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


NUM_DIGITS = 10
VOCAB_SIZE = 11  # 0-9 + separator
SEP_TOKEN = 10
D_MODEL = 4
N_HEADS = 2
D_HEAD = D_MODEL // N_HEADS  # 2
D_FF = 4
MAX_SEQ = 34  # 22 input + 11 output + 1 buffer


class HandCodedAdder(nn.Module):
    def __init__(self):
        super().__init__()

        self.embed = nn.Embedding(VOCAB_SIZE, D_MODEL).to(torch.float64)
        self.q_proj = nn.Linear(D_MODEL, D_MODEL, bias=False).to(torch.float64)
        self.k_proj = nn.Linear(D_MODEL, D_MODEL, bias=False).to(torch.float64)
        self.v_proj = nn.Linear(D_MODEL, D_MODEL, bias=False).to(torch.float64)
        self.o_proj = nn.Linear(D_MODEL, D_MODEL, bias=False).to(torch.float64)
        self.mlp_up = nn.Linear(D_MODEL, D_FF, bias=True).to(torch.float64)
        self.mlp_down = nn.Linear(D_FF, D_MODEL, bias=False).to(torch.float64)
        self.lm_head = nn.Linear(D_MODEL, VOCAB_SIZE, bias=True).to(torch.float64)

        # Fixed attention bias (0 params) - encodes which positions to attend to
        self._build_attention_bias()
        self._set_weights()

    def _build_attention_bias(self):
        """Build fixed attention bias matrix for both heads.

        For output position p = 21+i (predicting s_i):
          Head 0: attend to pos i (A_i) and pos 11+i (B_i) with equal weight
          Head 1: attend to pos i-1 (A_{i-1}) and pos 10+i (B_{i-1})
                  When i=0, head 1 attends to nothing useful (no carry for first digit)
        """
        M = torch.full((N_HEADS, MAX_SEQ, MAX_SEQ), float('-inf'), dtype=torch.float64)

        for i in range(11):  # 11 output digits
            p = 21 + i  # query position (last token when predicting s_i)

            # Head 0: attend to A_i and B_i
            if i < 10:
                M[0, p, i] = 0.0       # A_i
                M[0, p, 11 + i] = 0.0  # B_i
            else:
                # i=10: no A_10 or B_10, sum digit is just carry
                # attend to self (token is s_9)
                M[0, p, p] = 0.0

            # Head 1: attend to A_{i-1} and B_{i-1} for carry detection
            if i > 0 and i <= 10:
                M[1, p, i - 1] = 0.0       # A_{i-1}
                M[1, p, 10 + i] = 0.0      # B_{i-1}
            else:
                # i=0: no carry needed, attend to self
                M[1, p, p] = 0.0

        # Causal mask for all positions (input positions attend to selves)
        for p in range(MAX_SEQ):
            # Ensure at least self-attention for non-output positions
            if M[0, p].max() == float('-inf'):
                M[0, p, p] = 0.0
            if M[1, p].max() == float('-inf'):
                M[1, p, p] = 0.0

        self.register_buffer('attn_bias', M)

    def _set_weights(self):
        with torch.no_grad():
            # --- Embedding ---
            # dim 0: digit value, dims 1-3: zero
            emb = torch.zeros(VOCAB_SIZE, D_MODEL, dtype=torch.float64)
            for d in range(10):
                emb[d, 0] = float(d)
            self.embed.weight.copy_(emb)

            # --- Attention projections ---
            # Q and K are zero (rely entirely on fixed bias for routing)
            self.q_proj.weight.zero_()
            self.k_proj.weight.zero_()

            # V projection: extract digit value (dim 0) into both heads
            # Head 0 (dims 0-1): put digit value in head dim 0
            # Head 1 (dims 2-3): put digit value in head dim 0
            V = torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64)
            V[0, 0] = 1.0  # head 0, dim 0 <- input dim 0 (digit value)
            V[2, 0] = 1.0  # head 1, dim 0 <- input dim 0 (digit value)
            self.v_proj.weight.copy_(V)

            # O projection: route head outputs to model dims
            # Head 0 output (dims 0-1) -> model dim 1 (current pair sum)
            # Head 1 output (dims 2-3) -> model dim 2 (prev pair sum)
            O = torch.zeros(D_MODEL, D_MODEL, dtype=torch.float64)
            O[1, 0] = 1.0  # model dim 1 <- head 0, dim 0
            O[2, 2] = 1.0  # model dim 2 <- head 1, dim 0
            self.o_proj.weight.copy_(O)

            # After attention + residual at output position p=21+i:
            # dim 0 = embed(current_token) = s_{i-1} value (or 0 for SEP)
            # dim 1 = 0.5*(A_i + B_i)  [avg from softmax over 2 positions]
            # dim 2 = 0.5*(A_{i-1} + B_{i-1})  [avg from softmax]
            # dim 3 = 0

            # --- MLP ---
            # Need to compute:
            #   carry = step(2*dim2 - dim0 >= 9)  [diff = A+B-S, but dim2 is half]
            #   Actually dim2 = 0.5*(A_{i-1}+B_{i-1}), dim0 = s_{i-1}
            #   diff = A_{i-1}+B_{i-1} - s_{i-1} = 2*dim2 - dim0
            #   carry = 1 iff diff >= 9
            #
            #   raw_sum = 2*dim1 + carry  [dim1 is half the pair sum]
            #   result = raw_sum mod 10
            #
            # Using 4 ReLU neurons:
            #   n0 = ReLU(2*(2*dim2 - dim0) - 17)  = ReLU(4*dim2 - 2*dim0 - 17)
            #   n1 = ReLU(2*(2*dim2 - dim0) - 19)  = ReLU(4*dim2 - 2*dim0 - 19)
            #     carry = n0 - n1 (exact 0/1 for integer inputs)
            #   n2 = ReLU(2*(2*dim1 + carry_signal) - 19)
            #   n3 = ReLU(2*(2*dim1 + carry_signal) - 20)
            #     But carry_signal depends on n0,n1... can't do in 1 layer.
            #
            # Problem: single-layer MLP can't chain carry -> mod10.
            # Solution: encode the sum value directly and let LM head handle it.
            #
            # Alternative: compute (2*dim1 + carry) and use WRAPPED parabolic head
            # that handles values 0-19.
            #
            # Simplest: MLP outputs carry into dim 3. Then dim 1 has half-sum,
            # dim 3 has carry. LM head combines: logit[c] peaks at (2*dim1+dim3) mod 10.
            #
            # For the LM head to do this, we need the hidden state to contain
            # the raw sum = 2*dim1 + carry.
            # Let's have MLP put carry into dim 3, and a separate step to compute raw_sum.
            #
            # Actually, let's rethink. Use MLP to:
            #   1. Compute carry (0 or 1) from dims 0, 2
            #   2. Add carry to 2*dim1 to get raw_sum
            #   3. Compute raw_sum mod 10
            # This requires chaining, which 1-layer MLP can't do.
            #
            # NEW APPROACH: Make MLP output the carry, add it to dim1 via residual
            # tricks, and use a "double parabolic" LM head that handles 0-19.
            #
            # SIMPLEST APPROACH: Make dim 1 hold the full sum (not half).
            # If attention outputs un-averaged values, we get full sum.
            # But softmax normalizes...
            #
            # Fix: scale V projection by 2 so after averaging we get the full sum.

            # REVISED V projection: scale by 2
            V[0, 0] = 2.0  # head 0 gets 2x digit value
            V[2, 0] = 2.0  # head 1 gets 2x digit value
            self.v_proj.weight.copy_(V)

            # Now after attention:
            # dim 1 = A_i + B_i (full sum, range 0-18)
            # dim 2 = A_{i-1} + B_{i-1} (full sum, range 0-18)
            # dim 0 = s_{i-1} (digit value, range 0-9)

            # MLP: 4 neurons to compute carry
            # diff = dim2 - dim0 is in {-1, 0, 9, 10}
            # carry = 1 iff diff >= 9
            # Using clamped step: carry = ReLU(2*(diff) - 17) - ReLU(2*(diff) - 19)
            # = ReLU(2*dim2 - 2*dim0 - 17) - ReLU(2*dim2 - 2*dim0 - 19)
            #
            # Check: diff=-1: ReLU(-19)-ReLU(-21) = 0
            #         diff=0: ReLU(-17)-ReLU(-19) = 0
            #         diff=9: ReLU(1)-ReLU(-1) = 1
            #         diff=10: ReLU(3)-ReLU(1) = 2  WRONG!

            # Fix: use thresholds 15 and 17 (between 0 and 9):
            # carry = ReLU(2*diff - 15) - ReLU(2*diff - 17)
            # diff=-1: ReLU(-17)-ReLU(-19) = 0
            # diff=0: ReLU(-15)-ReLU(-17) = 0
            # diff=9: ReLU(3)-ReLU(1) = 2  WRONG again

            # Correct thresholds: need step that's 0 for diff<=0, 1 for diff>=9
            # Gap is 9, so use: carry = ReLU(diff - 0.5) - ReLU(diff - 1.5)
            # diff=-1: 0-0 = 0, diff=0: 0-0 = 0
            # diff=9: 8.5-7.5 = 1, diff=10: 9.5-8.5 = 1  YES!
            #
            # n0 = ReLU(dim2 - dim0 - 0.5)
            # n1 = ReLU(dim2 - dim0 - 1.5)
            # carry = n0 - n1
            #
            # For mod-10 of (dim1 + carry):
            # sum = dim1 + carry, range 0-19
            # mod10 = sum - 10*step(sum >= 10)
            # step(sum>=10) = ReLU(sum - 9.5) - ReLU(sum - 10.5) (same trick)
            # n2 = ReLU(dim1 + (n0-n1) - 9.5)
            # But n0,n1 are intermediate -- can't chain in 1-layer MLP.
            #
            # SOLUTION: Use 2 passes or restructure.
            # OR: output carry in a separate dim, let LM head do the rest.
            #
            # LM HEAD approach:
            # After MLP, hidden state has:
            #   dim 0 = s_{i-1} (unchanged by MLP if we zero those weights)
            #   dim 1 = A_i + B_i (unchanged)
            #   dim 2 = A_{i-1} + B_{i-1} (unchanged)
            #   dim 3 = carry (computed by MLP)
            #
            # The LM head needs to decode (dim1 + dim3) mod 10:
            # raw = dim1 + dim3 (range 0-19)
            # target digit c = raw mod 10
            #
            # Double parabolic: logit[c] = -(raw - c)^2 + -(raw - 10 - c)^2
            # = -2*raw^2 + 2*(2c+10)*raw - c^2 - (10+c)^2
            # Dropping terms constant in c:
            # = (4c+20)*raw - c^2 - (10+c)^2
            # = (4c+20)*raw - 2c^2 - 20c - 100
            # Taking derivative w.r.t. c: 4*raw - 4c - 20 = 0 => c = raw - 5
            # That doesn't give mod 10...
            #
            # Actually, the sum of two parabolas centered at c and c+10:
            # -(x-c)^2 + -(x-10-c)^2
            # The argmax over c is NOT c = x mod 10 in general.
            #
            # Better: use max instead of sum. But linear head can't do max.
            #
            # CORRECT approach: compute mod-10 in the MLP itself.
            # Need 2 layers or a wider MLP.
            #
            # Let me use 8 neurons in the MLP to do everything:
            pass

            # COMPLETE REDESIGN with ff=8:
            # Neurons 0-1: carry detection
            # Neurons 2-3: compute raw_sum = dim1 + carry, put in dim 3
            # ... but single layer can't chain.
            #
            # FINAL APPROACH: Accept that 1-layer MLP can't chain operations.
            # Use the embedding + residual to help.
            #
            # KEY INSIGHT: We don't need to compute mod-10 explicitly!
            # Use a lookup-style LM head that's periodic.
            #
            # For each candidate digit c in 0-9:
            # logit[c] = -min((x-c)^2, (x-10-c)^2)
            #          = max(-(x-c)^2, -(x-10-c)^2)
            #
            # This CAN'T be done with a linear head.
            #
            # ACTUAL SOLUTION: The MLP computes carry. Then the hidden state
            # value fed to LM head is dim1 + dim3 = A_i+B_i+carry (range 0-19).
            # We DON'T need mod-10 if we use the right LM head.
            #
            # For x in [0,19], we want argmax_c logit[c] = x mod 10.
            # Use: logit[c] = cos(2*pi*(x-c)/10) (periodic!)
            # But cosine isn't linear in x...
            #
            # Or use float64 precision trick:
            # logit[c] = -(((x - c) mod 10) - 0)^2 ... can't compute mod in linear.
            #
            # I think the only clean solution with 1 layer is to have the MLP
            # compute the FINAL digit value directly (0-9) into dim 3,
            # using enough neurons.
            #
            # MLP with 8 neurons doing both carry and mod10:
            # Group A (carry): n0 = ReLU(dim2 - dim0 - 0.5)
            #                  n1 = ReLU(dim2 - dim0 - 1.5)
            # Group B (mod10 if no carry): n2 = ReLU(2*dim1 - 19)
            #                              n3 = ReLU(2*dim1 - 20)
            # Group C (mod10 if carry): n4 = ReLU(2*dim1 + 2 - 19) = ReLU(2*dim1 - 17)
            #                           n5 = ReLU(2*dim1 + 2 - 20) = ReLU(2*dim1 - 18)
            # But how to select between B and C based on carry? Can't with linear down proj.
            #
            # THIS IS FUNDAMENTALLY A 2-LAYER PROBLEM.
            # With 1 attention layer and 1 MLP, we can't chain carry -> mod10.
            # All successful hand-coded models either:
            # a) Use 2 layers (alexlitz: 2L)
            # b) Use the autoregressive loop to propagate carry implicitly

            # APPROACH B: Use the autoregressive carry propagation!
            # The carry is encoded in the PREVIOUS output token.
            # S_{i-1} = (A_{i-1} + B_{i-1} + C_{i-2}) mod 10
            # C_{i-1} = 1 iff diff = A_{i-1}+B_{i-1} - S_{i-1} >= 9
            #          (because diff is in {-1, 0, 9, 10})
            #
            # So the model computes at each step:
            # 1. carry = step(A_{i-1}+B_{i-1} - S_{i-1} - 0.5) clamped to {0,1}
            # 2. result = (A_i + B_i + carry) mod 10
            #
            # And we need BOTH in one MLP layer.
            # With d_ff=8, we have 8 neurons. We can do it by pre-computing
            # ALL combinations:
            #
            # For carry=0: result = (A_i+B_i) mod 10
            #   n2 = ReLU(2*dim1 - 19), n3 = ReLU(2*dim1 - 20)
            #   mod10_no_carry = dim1 - 10*(n2-n3)
            #
            # For carry=1: result = (A_i+B_i+1) mod 10
            #   n4 = ReLU(2*(dim1+1) - 19) = ReLU(2*dim1 - 17)
            #   n5 = ReLU(2*(dim1+1) - 20) = ReLU(2*dim1 - 18)
            #   mod10_with_carry = (dim1+1) - 10*(n4-n5)
            #
            # Then output = carry * mod10_with_carry + (1-carry) * mod10_no_carry
            # But this multiplication of carry * mod10 is nonlinear!
            #
            # UGHHH. OK I think for a WORKING model (not minimal params),
            # the easiest path is 2 layers. Layer 1 computes carry.
            # Layer 2 uses carry + current pair to compute output digit.
            #
            # But the plan says 1 layer. Let me think more...
            #
            # WAIT. There IS a way with 1 layer if we use the right encoding.
            # Instead of computing mod-10 explicitly, output the RAW SUM
            # (dim1 + carry, range 0-19) and use a LM head that does modular decode.
            #
            # LM head for modular decode:
            # We need: for input x in {0,...,19}, argmax_c W[c]*x + b[c] = x mod 10
            #
            # This means for any x and any wrong digit c' != x mod 10:
            #   W[x mod 10]*x + b[x mod 10] > W[c']*x + b[c']
            #
            # Can a linear function discriminate this? Let's check x=5 vs x=15:
            # Both should output 5. So we need:
            #   W[5]*5 + b[5] > W[c]*5 + b[c] for c != 5
            #   W[5]*15 + b[5] > W[c]*15 + b[c] for c != 5
            #
            # From the first: (W[5]-W[c])*5 > b[c]-b[5]
            # From the second: (W[5]-W[c])*15 > b[c]-b[5]
            #
            # Both must hold, so (W[5]-W[c]) must have the same sign for 5 and 15,
            # which it does (it's the same value). This works as long as
            # (W[5]-W[c]) > 0, i.e., W[5] > W[c] for all c != 5.
            # But then for x=0 and x=10 (both should output 0):
            #   W[0]*0 + b[0] > W[5]*0 + b[5] => b[0] > b[5]
            #   W[0]*10 + b[0] > W[5]*10 + b[5] => 10*(W[0]-W[5]) > b[5]-b[0]
            # If b[0] > b[5], then b[5]-b[0] < 0, so we need 10*(W[0]-W[5]) > negative.
            # This allows W[0] < W[5]. But earlier we needed W[5] > W[c] for all c.
            # So W[0] < W[5] and W[5] > W[0] -- consistent.
            # But for x=0: W[0]*0+b[0] > W[5]*0+b[5] means b[0] > b[5].
            # And for x=5: W[5]*5+b[5] > W[0]*5+b[0] means 5*(W[5]-W[0]) > b[0]-b[5].
            # Let's say W[5]-W[0] = w, b[0]-b[5] = d > 0.
            # Then: 5*w > d and also for x=15: 15*w + (b[5]-b[0]) = 15w - d > 0 iff w > d/15.
            # And for x=10: 10*(W[0]-W[5]) = -10w > b[5]-b[0] = -d iff -10w > -d iff 10w < d.
            # So we need: 5w > d AND 10w < d.
            # 5w > d and 10w < d means 5w > d > 10w, i.e., 5w > 10w, i.e., w < 0.
            # But then 5w > d with w < 0 means d < 5w < 0, contradicting d > 0.
            #
            # CONCLUSION: A linear LM head CANNOT do modular-10 decode for input range [0,19].
            # This is mathematically impossible.
            #
            # Therefore, the model MUST compute (sum mod 10) before the LM head.
            # And this requires chaining carry + mod10, which needs 2 layers.
            #
            # REVISED PLAN: Use 2 layers.
            # Layer 1: attention gathers digits, MLP computes carry into a free dim.
            # Layer 2: attention reads carry from current position (self-attend),
            #          MLP computes (pair_sum + carry) mod 10.
            pass

    def forward(self, x):
        pass


# Placeholder - this approach needs 2 layers.
# Switching to a clean 2-layer implementation.
def build_model():
    return None, {"name": "WIP", "params": 0}


def add(model, a, b):
    return 0
