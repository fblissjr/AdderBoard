"""
Hand-coded 2-layer transformer adder.

Architecture: 2L decoder, d_model=4, 1 head (d_head=4), ff=4 per layer
Input: reversed digits LSB-first: a0..a9 SEP b0..b9 SEP (22 tokens)
       Vocab: 0-9 digits, 10=SEP
Output: 11 sum digits LSB-first, autoregressive

Layer 1: Attention gathers digit values. MLP computes carry (0 or 1).
Layer 2: Self-attention passes carry through. MLP computes (sum + carry) mod 10.
LM Head: Parabolic decode on the mod-10 result.

All weights are float64. Fixed attention biases are 0-param buffers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB_SIZE = 11  # 0-9 + separator
SEP = 10
D = 4   # d_model
FF = 4  # ff width per layer
MAX_SEQ = 34


class HandCodedAdder2L(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB_SIZE, D).double()

        # Layer 1
        self.v1 = nn.Linear(D, D, bias=False).double()
        self.o1 = nn.Linear(D, D, bias=False).double()
        self.ff1_up = nn.Linear(D, FF, bias=True).double()
        self.ff1_down = nn.Linear(FF, D, bias=False).double()

        # Layer 2
        self.ff2_up = nn.Linear(D, FF, bias=True).double()
        self.ff2_down = nn.Linear(FF, D, bias=False).double()

        # LM head
        self.lm_head = nn.Linear(D, VOCAB_SIZE, bias=True).double()

        self._build_biases()
        self._set_weights()

    def _build_biases(self):
        """Fixed attention routing (0 parameters).

        Layer 1 attention (1 head): from output pos p=21+i, attend to:
          A_i at position i, and B_i at position 11+i (equal weight).
          For i=10 (MSB of sum): attend only to sep tokens (value 0).

        Also from output pos p=21+i, the RESIDUAL connection preserves
        the embedding of the current token (s_{i-1} for i>0, SEP for i=0).

        Layer 2 attention: uniform self-attention (just passes through).
        We implement it as identity (attend only to self).
        """
        # Layer 1 bias
        M1 = torch.full((1, MAX_SEQ, MAX_SEQ), float('-inf'), dtype=torch.float64)
        for i in range(11):
            p = 21 + i
            if i < 10:
                M1[0, p, i] = 0.0       # A_i
                M1[0, p, 11 + i] = 0.0  # B_i
            else:
                # i=10: attend to position 10 (SEP, value 0) for zero contribution
                M1[0, p, 10] = 0.0
                M1[0, p, 21] = 0.0  # second SEP

        # Input positions: self-attend
        for p in range(22):
            M1[0, p, p] = 0.0

        self.register_buffer('bias1', M1)

        # Layer 2 bias: self-attend only
        M2 = torch.full((1, MAX_SEQ, MAX_SEQ), float('-inf'), dtype=torch.float64)
        for p in range(MAX_SEQ):
            M2[0, p, p] = 0.0
        self.register_buffer('bias2', M2)

    def _set_weights(self):
        with torch.no_grad():
            # === EMBEDDING ===
            # dim 0: digit value (0-9 for digits, 0 for SEP)
            # dim 1: will hold A_i + B_i after layer 1 attention
            # dim 2: will hold carry after layer 1 MLP
            # dim 3: will hold final result after layer 2 MLP
            emb = torch.zeros(VOCAB_SIZE, D, dtype=torch.float64)
            for d in range(10):
                emb[d, 0] = float(d)
            # SEP maps to [0, 0, 0, 0]
            self.embed.weight.copy_(emb)

            # === LAYER 1 ATTENTION ===
            # Q, K are implicit zero (fixed bias routes everything).
            # V extracts digit value and scales by 2 (to undo softmax averaging).
            # V: input dim 0 -> output dim 1
            V1 = torch.zeros(D, D, dtype=torch.float64)
            V1[1, 0] = 2.0  # output dim 1 = 2 * input dim 0
            self.v1.weight.copy_(V1)

            # O: identity-like, passes dim 1 through
            O1 = torch.zeros(D, D, dtype=torch.float64)
            O1[1, 1] = 1.0
            self.o1.weight.copy_(O1)

            # After L1 attention + residual at output position 21+i:
            # dim 0 = s_{i-1} (current token value, 0-9; or 0 for i=0)
            # dim 1 = A_i + B_i (0-18)
            # dim 2 = 0
            # dim 3 = 0

            # === LAYER 1 MLP: carry detection ===
            # diff = A_{i-1} + B_{i-1} - s_{i-1}
            # But we don't have A_{i-1}+B_{i-1} directly!
            # dim 1 has the CURRENT pair sum, not the previous one.
            #
            # PROBLEM: We need the PREVIOUS pair sum for carry detection,
            # but attention gathered the CURRENT pair sum.
            #
            # REVISED APPROACH: Head gathers BOTH current and previous pair info.
            # Use d_head > 1 or 2 heads.
            #
            # Actually, with 1 head attending to 2 positions (A_i, B_i),
            # we can only get one pair sum. To get both current AND previous,
            # we need 2 heads or attention to 4 positions.
            #
            # Let me use 2 heads with d_head=2.
            pass

        # This approach has a fundamental routing problem with 1 head.
        # Need to restructure. See handcoded_v3.py.

    def forward(self, x):
        pass


def build_model():
    return None, {"name": "WIP", "params": 0}


def add(model, a, b):
    return 0
