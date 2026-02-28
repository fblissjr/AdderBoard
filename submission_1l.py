"""
1-layer hand-coded transformer for 10-digit addition.

Architecture: 1L decoder, d_model=3, 3 heads (d_head=1), ff=4
Input: reversed digits LSB-first: a0..a9 SEP b0..b9 SEP (22 tokens)
Output: 11 sum digits LSB-first (autoregressive)

Vocab: 0-9 digits, 10=SEP

Key techniques (all derived from first principles):

1. ALiBi prefix sum for carry detection:
   Head 1 attends to all preceding digit pairs with scores -(k-j)*log(10),
   producing weights proportional to 10^(j-k). The resulting prefix sum
   S_k = sum_{j<k} (A_j+B_j) * 10^(j-k) encodes the carry as floor(S_k).
   This puts carry detection in ATTENTION, not the MLP, enabling 1 layer.

2. Residual cancellation via dedicated head:
   Head 2 attends only to self with V = -token_value. Combined with the
   residual connection (+token_value), they cancel exactly. The MLP sees
   clean (A+B) regardless of what token was previously generated.

3. e^80 softmax anchoring:
   The separator token gets attention score 80.0. After softmax, the
   denominator is ~e^80. O_proj multiplies by e^80 to recover exact scalar
   values. Error margin: ~3.6e-35 vs 1e-11 threshold = 24 orders of safety.

4. 2-hinge ReLU step function:
   ReLU(x + eps1) - ReLU(x + eps2) = (eps1 - eps2) when x > 0, else 0.
   Scaled by 1/(eps1 - eps2), this gives an exact 0/1 step function for
   any positive input, regardless of magnitude. Used for carry extraction
   and mod-10 overflow detection.

5. Parabolic LM head:
   logit[c] = 2c*x - c^2 = -(c-x)^2 + x^2 peaks at c = x = correct digit.
   This sidesteps the impossibility of linear mod-10 decode: mod-10 is
   computed in the MLP BEFORE the LM head sees it.

33 unique parameter values. float64 throughout.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB = 11   # 0-9 + SEP(10)
SEP = 10
D = 3        # d_model
N_HEADS = 3  # d_head = 1 each
FF = 4       # MLP intermediate dim
MAX_SEQ = 34 # 22 input + 11 output + 1 buffer


class TransformerBlock(nn.Module):
    """Standard single-layer transformer block: multihead attention + MLP."""

    def __init__(self, d_model, n_heads, ff_dim):
        super().__init__()
        self.n_heads = n_heads
        self.d_head = d_model // n_heads

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.mlp_up = nn.Linear(d_model, ff_dim, bias=True)
        self.mlp_down = nn.Linear(ff_dim, d_model, bias=False)

    def forward(self, x, mask):
        B, L, _D = x.shape
        q = self.q_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(x).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        scores = q @ k.transpose(-2, -1) + mask
        attn = F.softmax(scores, dim=-1)

        out = (attn @ v).transpose(1, 2).reshape(B, L, _D)
        out = self.o_proj(out)

        x = x + out
        x = x + self.mlp_down(F.relu(self.mlp_up(x)))
        return x


class OneLayerAdder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D).double()
        self.block = TransformerBlock(D, N_HEADS, FF).double()
        self.lm_head = nn.Linear(D, VOCAB, bias=True).double()

        self._build_mask()
        self._set_weights()

    def _build_mask(self):
        """Fixed per-head attention routing (registered buffer, 0 params).

        Three heads, three jobs:
        - Head 0: Gather A_k + B_k for current output digit k
        - Head 1: Compute carry via ALiBi base-10 prefix sum over all j < k
        - Head 2: Cancel residual connection (self-attend with V = -token)
        """
        M = torch.full((N_HEADS, MAX_SEQ, MAX_SEQ), float('-inf'), dtype=torch.float64)

        for q in range(MAX_SEQ):
            if q < 21:
                # Prompt positions (0-20): self-attend on all heads
                M[0, q, q] = 0.0
                M[1, q, q] = 0.0
                M[2, q, q] = 0.0
            else:
                # Output positions (21+): k = q - 21 is the digit index
                k = q - 21

                # Head 2: self-attend for residual cancellation
                M[2, q, q] = 0.0

                # Head 0: digit pair A_k + B_k
                # Score 0 on the digit positions, score 80 on separator (anchor)
                if k < 10:
                    M[0, q, k] = 0.0           # A_k
                    M[0, q, 11 + k] = 0.0      # B_k
                M[0, q, 10] = 80.0              # softmax anchor (SEP)

                # Head 1: ALiBi prefix sum for carry into position k
                # Score -(k-j)*log(10) on preceding digit positions
                # After softmax: weight on (A_j,B_j) proportional to 10^(j-k)
                # Prefix sum S_k = sum_{j<k} (A_j+B_j) * 10^(j-k)
                # floor(S_k) = carry into position k (always 0 or 1)
                for j in range(min(k, 10)):
                    score = -(k - j) * math.log(10)
                    M[1, q, j] = score           # A_j
                    M[1, q, 11 + j] = score      # B_j
                M[1, q, 10] = 80.0               # softmax anchor

        self.register_buffer('M', M)

    def _set_weights(self):
        """Analytically derive and set all weights."""
        with torch.no_grad():
            # === EMBEDDING ===
            # dim 0 = digit value (0-9). SEP = [0,0,0].
            emb = torch.zeros(VOCAB, D, dtype=torch.float64)
            for d in range(10):
                emb[d, 0] = float(d)
            self.embed.weight.copy_(emb)

            # === Q, K: all zero (routing comes from fixed mask) ===
            self.block.q_proj.weight.copy_(torch.zeros(D, D, dtype=torch.float64))
            self.block.k_proj.weight.copy_(torch.zeros(D, D, dtype=torch.float64))

            # === V: each head extracts one scalar from dim 0 ===
            # Head 0: +digit_value (for pair sum)
            # Head 1: +digit_value (for prefix sum)
            # Head 2: -digit_value (for residual cancellation)
            self.block.v_proj.weight.copy_(torch.tensor([
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ], dtype=torch.float64))

            # === O: scale by e^80 to undo softmax normalization ===
            # dim 0 = e^80 * head0 + 1.0 * head2
            #        = e^80 * (A+B)/(2+e^80) + (-token_value)
            #        approx (A+B) - token_value
            # After residual (+token_value): dim 0 = A+B. Clean.
            #
            # dim 1 = e^80 * head1 approx prefix_sum S_k
            # dim 2 = 0 (unused)
            exp80 = math.exp(80.0)
            self.block.o_proj.weight.copy_(torch.tensor([
                [exp80, 0.0, 1.0],
                [0.0, exp80, 0.0],
                [0.0, 0.0, 0.0],
            ], dtype=torch.float64))

            # === MLP: carry extraction + mod-10 ===
            #
            # After attention+residual:
            #   dim 0 = A_k + B_k (range 0-18, exact)
            #   dim 1 = S_k (prefix sum, carry = floor(S_k) in {0, 1})
            #
            # 2-hinge step function for exact 0/1 extraction:
            #   Paired neurons with epsilon offset: eps1=1e-11, eps2=0.5e-11
            #   ReLU(x + eps1) - ReLU(x + eps2) = 0.5e-11 when x > 0
            #   Scaled by 2e11 -> 1.0. Scaled by 2e12 -> 10.0.
            #
            # n0,n1: carry indicator (dim1 >= 1?)
            # n2,n3: overflow indicator (dim0 + dim1 >= 10?)
            self.block.mlp_up.weight.copy_(torch.tensor([
                [0.0, 1.0, 0.0],   # n0: dim1
                [0.0, 1.0, 0.0],   # n1: dim1
                [1.0, 1.0, 0.0],   # n2: dim0 + dim1
                [1.0, 1.0, 0.0],   # n3: dim0 + dim1
            ], dtype=torch.float64))

            self.block.mlp_up.bias.copy_(torch.tensor([
                -1.0 + 1e-11,      # n0: threshold at dim1 = 1
                -1.0 + 0.5e-11,    # n1: slightly different epsilon
                -10.0 + 1e-11,     # n2: threshold at sum = 10
                -10.0 + 0.5e-11,   # n3: slightly different epsilon
            ], dtype=torch.float64))

            # dim0_delta = 2e11*(n0-n1) - 2e12*(n2-n3)
            #            = +1 if carry_in, -10 if total >= 10
            # After residual: dim0 = (A+B) + carry - 10*overflow
            #                       = (A+B+carry) mod 10
            self.block.mlp_down.weight.copy_(torch.tensor([
                [2e11, -2e11, -2e12, 2e12],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ], dtype=torch.float64))

            # === LM HEAD: parabolic decode ===
            # logit[c] = 2c * dim0 - c^2 = -(c - dim0)^2 + dim0^2
            # Maximum at c = dim0 = (A+B+carry) mod 10 = correct digit.
            W_lm = torch.zeros(VOCAB, D, dtype=torch.float64)
            B_lm = torch.zeros(VOCAB, dtype=torch.float64)
            for c in range(10):
                W_lm[c, 0] = 2.0 * float(c)
                B_lm[c] = -float(c * c)
            B_lm[SEP] = -2e12   # never predict SEP (shared value with mlp_down)
            self.lm_head.weight.copy_(W_lm)
            self.lm_head.bias.copy_(B_lm)

    def forward(self, x):
        B, L = x.shape
        h = self.embed(x)
        mask = self.M[:, :L, :L].unsqueeze(0)
        h = self.block(h, mask)
        return self.lm_head(h)


def build_model():
    model = OneLayerAdder()
    model.forward  # ensure model is built

    # Count unique parameter values
    all_vals = set()
    for p in model.parameters():
        all_vals.update(p.data.flatten().tolist())
    unique_count = len(all_vals)

    metadata = {
        "name": "Hand-Coded 1L Adder (Claude)",
        "author": "Claude",
        "params": unique_count,
        "architecture": "1L decoder, d=3, 3h (d_head=1), ff=4",
        "tricks": [
            "ALiBi prefix sum for carry-in-attention",
            "Residual cancellation via dedicated head",
            "e^80 softmax anchoring",
            "2-hinge ReLU exact step function",
            "Parabolic LM head decode",
            "float64 precision",
        ],
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    a_digits = [(a // (10**i)) % 10 for i in range(10)]
    b_digits = [(b // (10**i)) % 10 for i in range(10)]
    seq = a_digits + [SEP] + b_digits + [SEP]

    device = next(model.parameters()).device

    with torch.no_grad():
        for _ in range(11):
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(x)
            next_token = logits[0, -1].argmax().item()
            seq.append(next_token)

    out_digits = seq[22:33]
    result = sum(d * (10**i) for i, d in enumerate(out_digits))
    return result
