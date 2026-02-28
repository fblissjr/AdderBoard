"""
Hand-coded 2-layer transformer for 10-digit addition.

Architecture: 2L decoder, d_model=5, 2 heads (d_head=2, +1 padding), ff=4
Input: reversed digits LSB-first: a0..a9 SEP b0..b9 SEP  (22 tokens)
Output: 11 sum digits LSB-first (autoregressive)

Vocab: 0-9 digits, 10=SEP

Layer 1 attention (2 heads):
  Head 0: gathers A_i + B_i (current digit pair sum)
  Head 1: gathers A_{i-1} + B_{i-1} (previous pair sum, for carry)
Layer 1 MLP: computes carry = step(prev_sum - current_token_value >= 9)
  Stores carry in dim 4.

Layer 2 attention: self-attend (identity, just passes data through)
Layer 2 MLP: computes (current_sum + carry) mod 10, stores in dim 3.

LM head: parabolic decode on dim 3.

float64 throughout. Fixed attention biases are registered buffers (0 params).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


VOCAB = 11   # 0-9 + SEP(10)
SEP = 10
D = 5        # d_model
MAX_SEQ = 34 # 22 input + 11 output + 1 buffer


class HandCodedAdder(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, D).double()

        # Layer 1: attention + MLP
        self.v1 = nn.Linear(D, 4, bias=False).double()   # V proj (2 heads x d_head=2)
        self.o1 = nn.Linear(4, D, bias=False).double()    # O proj
        self.ff1_up = nn.Linear(D, 4, bias=True).double()
        self.ff1_down = nn.Linear(4, D, bias=False).double()

        # Layer 2: MLP only (attention is self-attend = identity)
        self.ff2_up = nn.Linear(D, 4, bias=True).double()
        self.ff2_down = nn.Linear(4, D, bias=False).double()

        # LM head
        self.lm_head = nn.Linear(D, VOCAB, bias=True).double()

        self._build_masks()
        self._set_weights()

    def _build_masks(self):
        """Build fixed attention routing masks (0 params)."""
        M = torch.full((2, MAX_SEQ, MAX_SEQ), float('-inf'), dtype=torch.float64)

        for i in range(11):
            p = 21 + i  # output position when predicting s_i

            # Head 0: attend to A_i and B_i
            if i < 10:
                M[0, p, i] = 0.0       # A_i at position i
                M[0, p, 11 + i] = 0.0  # B_i at position 11+i
            else:
                M[0, p, 10] = 0.0  # SEP (value=0)

            # Head 1: attend to A_{i-1} and B_{i-1}
            if i > 0:
                M[1, p, i - 1] = 0.0           # A_{i-1}
                M[1, p, 11 + i - 1] = 0.0      # B_{i-1}
            else:
                M[1, p, 10] = 0.0  # SEP for i=0

        # All other positions: self-attend
        for p in range(MAX_SEQ):
            if M[0, p].max() == float('-inf'):
                M[0, p, p] = 0.0
            if M[1, p].max() == float('-inf'):
                M[1, p, p] = 0.0

        self.register_buffer('mask1', M)

    def _set_weights(self):
        with torch.no_grad():
            # === EMBEDDING ===
            emb = torch.zeros(VOCAB, D, dtype=torch.float64)
            for d in range(10):
                emb[d, 0] = float(d)
            self.embed.weight.copy_(emb)

            # === LAYER 1 ATTENTION ===
            # V: extract digit value * 2 (to cancel softmax averaging over 2 positions)
            V = torch.zeros(4, D, dtype=torch.float64)
            V[0, 0] = 2.0  # head0 dim0 <- 2 * digit_value
            V[2, 0] = 2.0  # head1 dim0 <- 2 * digit_value
            self.v1.weight.copy_(V)

            # O: route to model dims
            O = torch.zeros(D, 4, dtype=torch.float64)
            O[1, 0] = 1.0  # dim1 <- head0 (current pair sum)
            O[2, 2] = 1.0  # dim2 <- head1 (previous pair sum)
            self.o1.weight.copy_(O)

            # After L1 attention + residual:
            # dim0 = token_value, dim1 = A_i+B_i, dim2 = A_{i-1}+B_{i-1}

            # === LAYER 1 MLP: carry detection ===
            # carry = step(dim2 - dim0 >= 9)
            # = ReLU(dim2 - dim0 - 0.5) - ReLU(dim2 - dim0 - 1.5)
            W_up1 = torch.zeros(4, D, dtype=torch.float64)
            b_up1 = torch.zeros(4, dtype=torch.float64)
            W_up1[0, 0] = -1.0;  W_up1[0, 2] = 1.0;  b_up1[0] = -0.5
            W_up1[1, 0] = -1.0;  W_up1[1, 2] = 1.0;  b_up1[1] = -1.5
            self.ff1_up.weight.copy_(W_up1)
            self.ff1_up.bias.copy_(b_up1)

            W_down1 = torch.zeros(D, 4, dtype=torch.float64)
            W_down1[4, 0] = 1.0;  W_down1[4, 1] = -1.0  # carry = n0 - n1
            self.ff1_down.weight.copy_(W_down1)

            # After L1: dim4 = carry (0 or 1)

            # === LAYER 2 MLP: (dim1 + dim4) mod 10 ===
            # S = dim1 + dim4, range [0, 19]
            # n0 = ReLU(S + 0.5)       [always positive, = S + 0.5]
            # n1 = ReLU(2S - 19)       [positive when S >= 10]
            # n2 = ReLU(2S - 20)       [positive when S >= 11]
            # step(S>=10) = n1 - n2
            # result = n0 - 10*n1 + 10*n2 = S+0.5 - 10*step = (S mod 10) + 0.5
            W_up2 = torch.zeros(4, D, dtype=torch.float64)
            b_up2 = torch.zeros(4, dtype=torch.float64)
            W_up2[0, 1] = 1.0;  W_up2[0, 4] = 1.0;  b_up2[0] = 0.5
            W_up2[1, 1] = 2.0;  W_up2[1, 4] = 2.0;  b_up2[1] = -19.0
            W_up2[2, 1] = 2.0;  W_up2[2, 4] = 2.0;  b_up2[2] = -20.0
            self.ff2_up.weight.copy_(W_up2)
            self.ff2_up.bias.copy_(b_up2)

            W_down2 = torch.zeros(D, 4, dtype=torch.float64)
            W_down2[3, 0] = 1.0;  W_down2[3, 1] = -10.0;  W_down2[3, 2] = 10.0
            self.ff2_down.weight.copy_(W_down2)

            # After L2: dim3 = (A_i + B_i + carry) mod 10 + 0.5

            # === LM HEAD: parabolic decode on dim 3 ===
            # logit[c] = -(dim3 - c - 0.5)^2 = 2(c+0.5)*dim3 - (c+0.5)^2 + const
            W_lm = torch.zeros(VOCAB, D, dtype=torch.float64)
            B_lm = torch.zeros(VOCAB, dtype=torch.float64)
            for c in range(10):
                W_lm[c, 3] = 2.0 * (c + 0.5)
                B_lm[c] = -(c + 0.5) ** 2
            B_lm[10] = -1e6  # never predict SEP
            self.lm_head.weight.copy_(W_lm)
            self.lm_head.bias.copy_(B_lm)

    def forward(self, x):
        B, L = x.shape
        h = self.embed(x)

        # Layer 1 attention (fixed routing, Q/K = 0)
        mask = self.mask1[:, :L, :L]
        v = self.v1(h).view(B, L, 2, 2).transpose(1, 2)
        scores = mask.unsqueeze(0).expand(B, -1, -1, -1)
        attn = F.softmax(scores, dim=-1)
        attn_out = (attn @ v).transpose(1, 2).reshape(B, L, 4)
        h = h + self.o1(attn_out)

        # Layer 1 MLP
        h = h + self.ff1_down(F.relu(self.ff1_up(h)))

        # Layer 2 (MLP only, attention = identity)
        h = h + self.ff2_down(F.relu(self.ff2_up(h)))

        return self.lm_head(h)


def build_model():
    model = HandCodedAdder()
    model.eval()
    total = sum(p.numel() for p in model.parameters())
    metadata = {
        "name": "Hand-Coded 2L Adder",
        "author": "Claude",
        "params": total,
        "architecture": "2L decoder, d=5, 2h, ff=4",
        "tricks": [
            "Fixed attention routing (0-param bias buffers)",
            "Clamped ReLU step for carry detection",
            "Two-hinge ReLU for mod-10",
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
