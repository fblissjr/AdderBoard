#!/usr/bin/env python3
"""Train a transformer adder from scratch with reversed digits.

Key insights combined from both approaches:
- Reversed digits (LSB-first): carry propagation becomes naturally causal
- Multi-head attention: allows specialization (digit extraction vs carry detection)
- Teacher forcing + CE loss: smooth gradient signal
- Curriculum learning: 1-digit -> 10-digit progression
- MLP biases: enable threshold detection for carry logic

Configurable architectures via --config flag.
"""

import argparse
import math
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# ==========================================
# ARCHITECTURE CONFIGS
# ==========================================
CONFIGS = {
    # === FIXED MASK (hand-coded routing + trainable weights) ===
    # No anchor: simplest, most gradient-friendly. ~130 trainable params.
    "mask-none": {"d": 3, "n_heads": 3, "d_ff": 4, "mlp_bias": True, "tie_embed": False,
                  "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "slopes", "mask_anchor": None},
    # Anchor=5: moderate normalization (e^5=148). Carry prefix sum recoverable.
    "mask-a5": {"d": 3, "n_heads": 3, "d_ff": 4, "mlp_bias": True, "tie_embed": False,
                "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "slopes", "mask_anchor": 5.0},
    # Anchor=10: strong normalization (e^10=22026). Near-exact prefix sum.
    "mask-a10": {"d": 3, "n_heads": 3, "d_ff": 4, "mlp_bias": True, "tie_embed": False,
                 "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "slopes", "mask_anchor": 10.0},
    # Structural mask: 0/-inf only, no slopes. Q@K must learn attention weights.
    "mask-even": {"d": 3, "n_heads": 3, "d_ff": 4, "mlp_bias": True, "tie_embed": False,
                  "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "even", "mask_anchor": None},
    # No norms: raw magnitude preserved through network (matches hand-coded model)
    "mask-nonorm": {"d": 3, "n_heads": 3, "d_ff": 4, "mlp_bias": True, "tie_embed": False,
                    "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "slopes",
                    "mask_anchor": None, "skip_norm": True},
    # Warm init: embed + V initialized to hand-coded structure
    "mask-warm": {"d": 3, "n_heads": 3, "d_ff": 4, "mlp_bias": True, "tie_embed": False,
                  "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "slopes",
                  "mask_anchor": None, "warm_init": True},
    # Warm init + larger MLP (8 neurons for more carry/mod-10 capacity)
    "mask-warm8": {"d": 3, "n_heads": 3, "d_ff": 8, "mlp_bias": True, "tie_embed": False,
                   "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "slopes",
                   "mask_anchor": None, "warm_init": True},
    # Medium models: d=3 with larger MLP (competitive with 311p trained leader)
    "mask-ff12": {"d": 3, "n_heads": 3, "d_ff": 12, "mlp_bias": True, "tie_embed": False,
                  "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "slopes",
                  "mask_anchor": None},  # 204 params
    "mask-ff16": {"d": 3, "n_heads": 3, "d_ff": 16, "mlp_bias": True, "tie_embed": False,
                  "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "slopes",
                  "mask_anchor": None},  # 232 params
    # Warm-init versions (embed + V from hand-coded solution)
    "mask-w4": {"d": 3, "n_heads": 3, "d_ff": 4, "mlp_bias": True, "tie_embed": False,
                "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "slopes",
                "mask_anchor": None, "warm_init": True},  # 148 params
    "mask-w5": {"d": 3, "n_heads": 3, "d_ff": 5, "mlp_bias": True, "tie_embed": False,
                "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "slopes",
                "mask_anchor": None, "warm_init": True},  # 155 params
    "mask-w6": {"d": 3, "n_heads": 3, "d_ff": 6, "mlp_bias": True, "tie_embed": False,
                "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "slopes",
                "mask_anchor": None, "warm_init": True},  # 162 params
    "mask-w8": {"d": 3, "n_heads": 3, "d_ff": 8, "mlp_bias": True, "tie_embed": False,
                "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "slopes",
                "mask_anchor": None, "warm_init": True},  # 176 params
    "mask-w10": {"d": 3, "n_heads": 3, "d_ff": 10, "mlp_bias": True, "tie_embed": False,
                 "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "slopes",
                 "mask_anchor": None, "warm_init": True},  # 190 params
    "mask-w12": {"d": 3, "n_heads": 3, "d_ff": 12, "mlp_bias": True, "tie_embed": False,
                 "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "slopes",
                 "mask_anchor": None, "warm_init": True},  # 204 params
    "mask-w16": {"d": 3, "n_heads": 3, "d_ff": 16, "mlp_bias": True, "tie_embed": False,
                 "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "slopes",
                 "mask_anchor": None, "warm_init": True},  # 232 params
    # Larger model with mask (proof of concept)
    "mask-large": {"d": 6, "n_heads": 3, "d_ff": 12, "mlp_bias": True, "tie_embed": False,
                   "use_alibi": False, "steep_alibi": False, "use_fixed_mask": "slopes",
                   "mask_anchor": None},  # 456 params

    # === ALIBI CONFIGS (previous experiments) ===
    "steep-3h": {"d": 3, "n_heads": 3, "d_ff": 4, "mlp_bias": True, "tie_embed": False, "use_alibi": True, "steep_alibi": True},
    "steep-med": {"d": 4, "n_heads": 2, "d_ff": 8, "mlp_bias": True, "tie_embed": False, "use_alibi": True, "steep_alibi": True},
    "steep-large": {"d": 6, "n_heads": 3, "d_ff": 12, "mlp_bias": True, "tie_embed": False, "use_alibi": True, "steep_alibi": True},
    "alibi-3h": {"d": 3, "n_heads": 3, "d_ff": 4, "mlp_bias": True, "tie_embed": False, "use_alibi": True, "steep_alibi": False},
    "alibi-large": {"d": 6, "n_heads": 3, "d_ff": 12, "mlp_bias": True, "tie_embed": False, "use_alibi": True, "steep_alibi": False},
    # Sinusoidal PE
    "sin-large": {"d": 6, "n_heads": 3, "d_ff": 12, "mlp_bias": True, "tie_embed": False, "use_alibi": False, "steep_alibi": False},
}


# ==========================================
# MASKS AND POSITIONAL ENCODING
# ==========================================
class RMSNormFree(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


def build_fixed_mask(max_seq=35, mask_type="slopes", anchor=None):
    """Build fixed attention routing mask from hand-coded 1L solution.

    3 heads with specialized routing:
    - Head 0: Digit pair (A_k, B_k) extraction
    - Head 1: Carry prefix sum (ALiBi slopes or even weighting)
    - Head 2: Self-attend for residual management

    mask_type: "slopes" = ALiBi log(10) decay in Head 1 (proven algorithm)
               "even" = all allowed positions get score 0 (Q@K learns weights)
    anchor: Score for SEP position in heads 0,1. None = 0.0 (no anchor).
    """
    M = torch.full((3, max_seq, max_seq), float('-inf'))
    sep_score = anchor if anchor is not None else 0.0

    for q in range(max_seq):
        if q < 21:
            # Prompt positions: self-attend only (all heads)
            for h in range(3):
                M[h, q, q] = 0.0
        else:
            k = q - 21  # output digit index (0 = ones, 10 = carry-out)

            # Head 0: digit pair A_k + B_k, plus SEP
            if k < 10:
                M[0, q, k] = 0.0           # A_k
                M[0, q, 11 + k] = 0.0      # B_k
            M[0, q, 10] = sep_score         # SEP (always, needed for k=10)

            # Head 1: carry prefix sum over preceding digit positions
            if k == 0:
                # First output digit: no carry, only SEP available
                M[1, q, 10] = sep_score
            else:
                for j in range(min(k, 10)):
                    if mask_type == "slopes":
                        score = -(k - j) * math.log(10)
                    else:  # "even"
                        score = 0.0
                    M[1, q, j] = score           # A_j
                    M[1, q, 11 + j] = score      # B_j
                # No SEP in Head 1 for k > 0 (avoids diluting prefix sum)

            # Head 2: self-attend only (residual management)
            M[2, q, q] = 0.0

    return M


def build_alibi_bias(n_heads, max_len=35, steep=False):
    """Build ALiBi attention bias matrix. 0 learnable params.

    Standard ALiBi slopes are geometric: 2^(-8/n * (i+1)).
    Steep mode uses slopes tuned for addition: [log(10), 0.5, 0.01].
    """
    if steep and n_heads == 3:
        # Slopes tuned for addition task:
        # Head 0: steep decay (carry computation needs nearby digits)
        # Head 1: medium decay (digit pair extraction)
        # Head 2: gentle decay (global context)
        slopes = torch.tensor([math.log(10), 0.5, 0.05])
    elif steep and n_heads == 2:
        slopes = torch.tensor([math.log(10), 0.1])
    else:
        # Standard ALiBi slopes
        slopes = torch.tensor([2.0 ** (-8.0 / n_heads * (i + 1)) for i in range(n_heads)])

    positions = torch.arange(max_len)
    dist = positions.unsqueeze(0) - positions.unsqueeze(1)

    alibi = -slopes.unsqueeze(1).unsqueeze(2) * dist.unsqueeze(0).float()

    causal_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
    alibi.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))

    return alibi


# ==========================================
# MODEL
# ==========================================
class TrainableAdder(nn.Module):
    def __init__(self, d=3, n_heads=3, d_ff=4, mlp_bias=True, tie_embed=False,
                 use_alibi=True, steep_alibi=False, use_fixed_mask=None, mask_anchor=None,
                 skip_norm=False, warm_init=False):
        super().__init__()
        self.d = d
        self.n_heads = n_heads
        self.d_head = d // n_heads
        self.tie_embed = tie_embed
        self.use_alibi = use_alibi
        self.steep_alibi = steep_alibi
        self.use_fixed_mask = use_fixed_mask
        self.skip_norm = skip_norm

        self.embed = nn.Embedding(12, d)

        # Multi-head attention
        self.q_proj = nn.Linear(d, d, bias=False)
        self.k_proj = nn.Linear(d, d, bias=False)
        self.v_proj = nn.Linear(d, d, bias=False)
        self.o_proj = nn.Linear(d, d, bias=False)

        # MLP
        self.mlp_up = nn.Linear(d, d_ff, bias=mlp_bias)
        self.mlp_down = nn.Linear(d_ff, d, bias=False)

        # Pre-norm (skipped when skip_norm=True for mask-based models)
        if not skip_norm:
            self.norm1 = RMSNormFree()
            self.norm2 = RMSNormFree()
            self.norm_f = RMSNormFree()

        # Output head
        if not tie_embed:
            self.lm_head = nn.Linear(d, 12, bias=True)
        else:
            self.lm_head = None

        if use_fixed_mask:
            # Fixed mask: 0-param attention routing from hand-coded solution
            # Requires n_heads=3; mask encodes which positions to attend to
            assert n_heads == 3, f"Fixed mask requires 3 heads, got {n_heads}"
            self.register_buffer("fixed_mask", build_fixed_mask(
                max_seq=35, mask_type=use_fixed_mask, anchor=mask_anchor))
            # Init Q,K near zero so mask dominates initially
            nn.init.normal_(self.q_proj.weight, std=0.01)
            nn.init.normal_(self.k_proj.weight, std=0.01)
        elif use_alibi:
            # ALiBi: 0-param positional bias in attention
            self.register_buffer("alibi", build_alibi_bias(n_heads, steep=steep_alibi))
        else:
            # Sinusoidal PE
            pe = torch.zeros(35, d)
            for pos in range(35):
                for i in range(0, d, 2):
                    pe[pos, i] = math.sin(pos / (10000 ** (i / d)))
                    if i + 1 < d:
                        pe[pos, i + 1] = math.cos(pos / (10000 ** (i / d)))
            self.register_buffer("pe", pe)

        if warm_init and d == 3 and n_heads == 3:
            self._warm_init()

    def _warm_init(self):
        """Initialize embed and V to match hand-coded structure.

        Embed: digit -> [digit_value, 0, 0], SEP tokens -> [0, 0, 0]
        V: heads 0,1 extract dim 0 (+1), head 2 extracts dim 0 (-1)
        This gives the model the right input representation from the start.
        """
        with torch.no_grad():
            emb = torch.zeros(12, 3)
            for d in range(10):
                emb[d, 0] = float(d)
            # SEP (10) and SEP2 (11) stay at [0, 0, 0]
            self.embed.weight.copy_(emb)

            # V: each head extracts digit value from dim 0
            # Head 0: +digit (for pair sum)
            # Head 1: +digit (for prefix sum)
            # Head 2: -digit (for residual cancellation)
            self.v_proj.weight.copy_(torch.tensor([
                [1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0],
            ]))

    def forward(self, x):
        B, L = x.shape

        if self.use_fixed_mask or self.use_alibi:
            h = self.embed(x)  # No PE added to embeddings
        else:
            h = self.embed(x) + self.pe[:L].unsqueeze(0)

        # Multi-head causal attention
        hn = h if self.skip_norm else self.norm1(h)
        q = self.q_proj(hn).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        k = self.k_proj(hn).view(B, L, self.n_heads, self.d_head).transpose(1, 2)
        v = self.v_proj(hn).view(B, L, self.n_heads, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(max(1e-5, float(self.d_head)))

        if self.use_fixed_mask:
            # Fixed mask handles both routing AND causal masking (-inf blocks)
            scores = scores + self.fixed_mask[:, :L, :L].unsqueeze(0)
        elif self.use_alibi:
            # Add ALiBi bias (includes causal mask)
            scores = scores + self.alibi[:, :L, :L].unsqueeze(0)
        else:
            # Standard causal mask
            causal = torch.tril(torch.ones(L, L, device=x.device)) == 0
            scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        attn_out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(B, L, self.d)
        h = h + self.o_proj(attn_out)

        # MLP with ReLU
        hn2 = h if self.skip_norm else self.norm2(h)
        h = h + self.mlp_down(F.relu(self.mlp_up(hn2)))

        # Output
        h_out = h if self.skip_norm else self.norm_f(h)
        if self.lm_head is not None:
            return self.lm_head(h_out)
        else:
            return F.linear(h_out, self.embed.weight)


# ==========================================
# DATA GENERATION (REVERSED DIGITS)
# ==========================================
def generate_batch(batch_size, max_digits, device):
    """Generate training batch with mixed-level sampling.

    70% of samples use max_digits, 30% use random lower levels.
    This prevents the model from learning level-specific patterns.
    """
    # Mix difficulty levels: 70% at max_digits, 30% at lower levels
    digits_per_sample = torch.full((batch_size,), max_digits, dtype=torch.long)
    if max_digits > 1:
        n_mixed = int(batch_size * 0.3)
        digits_per_sample[:n_mixed] = torch.randint(1, max_digits + 1, (n_mixed,))

    seqs = []
    for i in range(batch_size):
        d = digits_per_sample[i].item()
        a_val = torch.randint(0, 10**d, (1,)).item()
        b_val = torch.randint(0, 10**d, (1,)).item()

        # Inject carries: 20% chance of all-9s
        if torch.rand(1).item() < 0.2:
            a_val = 10**d - 1
        if torch.rand(1).item() < 0.2:
            b_val = 10**d - 1

        # Zero-pad to 10 digits and REVERSE (LSB first)
        sa = f"{a_val:010d}"[::-1]
        sb = f"{b_val:010d}"[::-1]
        ans = f"{a_val + b_val:011d}"[::-1]

        seq = (
            [int(c) for c in sa]
            + [10]
            + [int(c) for c in sb]
            + [11]
            + [int(c) for c in ans]
        )
        seqs.append(seq)

    return torch.tensor(seqs, device=device)


# ==========================================
# TRAINING
# ==========================================
def train(config_name="3h", lr=0.003, batch_size=512, max_steps=200000, all_levels=False):
    cfg = CONFIGS[config_name]
    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )

    model = TrainableAdder(**cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_steps, eta_min=lr * 0.01)

    mode = "all-levels (no curriculum)" if all_levels else "curriculum"
    print(f"Config: {config_name} = {cfg}")
    print(f"Parameters: {n_params}")
    print(f"Device: {device}")
    print(f"LR: {lr}, Batch: {batch_size}, Mode: {mode}")
    print("Reversed digits (LSB-first), teacher forcing")
    print()

    level = 10 if all_levels else 1
    step = 0
    moving_acc = 0.0
    best_acc = 0.0
    t0 = time.time()

    while level <= 10:
        model.train()
        seq = generate_batch(batch_size=batch_size, max_digits=level, device=device)

        x = seq[:, :-1]
        y = seq[:, 1:]

        logits = model(x)

        # CE loss on answer digits only (positions 21-32)
        loss = F.cross_entropy(
            logits[:, 21:32].reshape(-1, 12), y[:, 21:32].reshape(-1)
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        # Strict sequence accuracy
        preds = logits[:, 21:32].argmax(dim=-1)
        correct = (preds == y[:, 21:32]).all(dim=1).float().mean().item()
        moving_acc = 0.95 * moving_acc + 0.05 * correct

        if step % 200 == 0:
            elapsed = time.time() - t0
            cur_lr = optimizer.param_groups[0]['lr']
            print(
                f"  Lvl {level:<2} | Step {step:<6} | "
                f"Loss {loss.item():.4f} | "
                f"Batch {correct * 100:5.1f}% | "
                f"Avg {moving_acc * 100:5.1f}% | "
                f"LR {cur_lr:.5f} | "
                f"{elapsed:.1f}s"
            )
            sys.stdout.flush()

        if moving_acc > best_acc:
            best_acc = moving_acc

        # Curriculum upgrade at 90% (only for curriculum mode, not all-levels)
        if not all_levels and moving_acc > 0.90:
            elapsed = time.time() - t0
            if level == 10:
                print(f"\n  Reached 90% on 10-digit at step {step} ({elapsed:.1f}s). Continuing to grok...")
                # Don't break -- keep training for grokking
                moving_acc = 0.0  # Reset to track further progress
            else:
                level += 1
                moving_acc = 0.0
                print(f"\n  --- Upgrading to {level}-digit (step {step}, {elapsed:.1f}s) ---\n")

        step += 1
        if step > max_steps:
            print(f"\n  Step limit reached at level {level}")
            break

    elapsed = time.time() - t0
    print(f"\nTraining: {elapsed:.1f}s, {step} steps, best avg acc {best_acc:.1%}")

    # Autoregressive eval
    print("\nAutoregressive eval (500 random 10-digit pairs)...")
    model.eval()
    correct_count = 0
    total = 500
    rng = torch.Generator().manual_seed(42)
    for _ in range(total):
        a_val = torch.randint(0, 10**10, (1,), generator=rng).item()
        b_val = torch.randint(0, 10**10, (1,), generator=rng).item()
        expected = a_val + b_val

        sa = f"{a_val:010d}"[::-1]
        sb = f"{b_val:010d}"[::-1]
        seq = [int(c) for c in sa] + [10] + [int(c) for c in sb] + [11]

        with torch.no_grad():
            for _ in range(11):
                xt = torch.tensor([seq], dtype=torch.long, device=device)
                out = model(xt)
                next_token = out[0, -1].argmax().item()
                seq.append(next_token)

        c_digits = seq[22:33]
        result = sum(d * (10**i) for i, d in enumerate(c_digits))
        if result == expected:
            correct_count += 1

    acc = correct_count / total
    print(f"Autoregressive accuracy: {correct_count}/{total} = {acc:.1%}")

    if acc > 0.95:
        export_submission(model, config_name, n_params)
    else:
        print(f"\nAccuracy {acc:.1%} < 95%. Saving checkpoint.")
        torch.save(model.state_dict(), f"trained_{config_name}.pt")

    return model, acc


# ==========================================
# EXPORT
# ==========================================
def export_submission(model, config_name, n_params, filename=None):
    if filename is None:
        filename = f"submission_trained_{config_name}.py"
    print(f"\nExporting to {filename}...")

    sd = model.state_dict()
    # Only export learnable weights (skip buffers like alibi, pe)
    weight_lines = []
    learnable_keys = {n for n, _ in model.named_parameters()}
    for k, v in sd.items():
        if k in learnable_keys and v.numel() > 0:
            flat = v.flatten().tolist()
            weight_lines.append(
                f"    '{k}': torch.tensor({flat}, dtype=torch.float32).reshape({tuple(v.shape)})"
            )
    weights_block = "WEIGHTS = {\n" + ",\n".join(weight_lines) + "\n}\n"

    d = model.d
    n_heads = model.n_heads
    d_ff = model.mlp_up.out_features
    has_bias = model.mlp_up.bias is not None
    has_lm = model.lm_head is not None
    use_alibi = model.use_alibi
    use_fixed_mask = model.use_fixed_mask

    # Generate helper function code
    helper_fn = ""
    if use_fixed_mask:
        helper_fn = f"""
def build_fixed_mask(max_seq=35):
    M = torch.full((3, max_seq, max_seq), float('-inf'))
    sep_score = 0.0
    for q in range(max_seq):
        if q < 21:
            for h in range(3):
                M[h, q, q] = 0.0
        else:
            k = q - 21
            if k < 10:
                M[0, q, k] = 0.0
                M[0, q, 11 + k] = 0.0
            M[0, q, 10] = sep_score
            if k == 0:
                M[1, q, 10] = sep_score
            else:
                for j in range(min(k, 10)):
                    score = -(k - j) * math.log(10)
                    M[1, q, j] = score
                    M[1, q, 11 + j] = score
            M[2, q, q] = 0.0
    return M

"""
    elif use_alibi:
        helper_fn = """
def build_alibi_bias(n_heads, max_len=35):
    slopes = torch.tensor([2.0 ** (-8.0 / n_heads * (i + 1)) for i in range(n_heads)])
    positions = torch.arange(max_len)
    dist = positions.unsqueeze(0) - positions.unsqueeze(1)
    alibi = -slopes.unsqueeze(1).unsqueeze(2) * dist.unsqueeze(0).float()
    causal_mask = torch.triu(torch.ones(max_len, max_len), diagonal=1).bool()
    alibi.masked_fill_(causal_mask.unsqueeze(0), float("-inf"))
    return alibi

"""

    pe_init = ""
    if use_fixed_mask:
        pe_init = "        self.register_buffer('fixed_mask', build_fixed_mask())"
    elif use_alibi:
        pe_init = f"        self.register_buffer('alibi', build_alibi_bias({n_heads}))"
    else:
        pe_init = f"""        pe = torch.zeros(35, {d})
        for pos in range(35):
            for i in range(0, {d}, 2):
                pe[pos, i] = math.sin(pos / (10000 ** (i / {d})))
                if i + 1 < {d}:
                    pe[pos, i + 1] = math.cos(pos / (10000 ** (i / {d})))
        self.register_buffer('pe', pe)"""

    embed_line = "self.embed(x)" if (use_alibi or use_fixed_mask) else "self.embed(x) + self.pe[:L].unsqueeze(0)"

    if use_fixed_mask:
        attn_mask_code = "        scores = scores + self.fixed_mask[:, :L, :L].unsqueeze(0)"
    elif use_alibi:
        attn_mask_code = f"        scores = scores + self.alibi[:, :L, :L].unsqueeze(0)"
    else:
        attn_mask_code = """        causal = torch.tril(torch.ones(L, L, device=x.device)) == 0
        scores = scores.masked_fill(causal.unsqueeze(0).unsqueeze(0), float("-inf"))"""

    lm_head_init = f"        self.lm_head = nn.Linear({d}, 12, bias=True)" if has_lm else ""
    output_line = "self.lm_head(h_norm)" if has_lm else "F.linear(h_norm, self.embed.weight)"

    tricks = ["Reversed LSB-First", "Teacher Forcing", "RMSNorm"]
    if use_alibi:
        tricks.append("ALiBi")
    if use_fixed_mask:
        tricks.append("Fixed Mask (hand-coded routing)")

    code = f'''"""Trained {n_params}-parameter transformer adder.

Trained from scratch: AdamW + reversed digits (LSB-first).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormFree(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)

{helper_fn}
class TrainableAdder(nn.Module):
    def __init__(self):
        super().__init__()
        self.d = {d}
        self.n_heads = {n_heads}
        self.d_head = {d // n_heads}
        self.embed = nn.Embedding(12, {d})
        self.q_proj = nn.Linear({d}, {d}, bias=False)
        self.k_proj = nn.Linear({d}, {d}, bias=False)
        self.v_proj = nn.Linear({d}, {d}, bias=False)
        self.o_proj = nn.Linear({d}, {d}, bias=False)
        self.mlp_up = nn.Linear({d}, {d_ff}, bias={has_bias})
        self.mlp_down = nn.Linear({d_ff}, {d}, bias=False)
        self.norm1 = RMSNormFree()
        self.norm2 = RMSNormFree()
        self.norm_f = RMSNormFree()
{lm_head_init}
{pe_init}

    def forward(self, x):
        B, L = x.shape
        h = {embed_line}
        hn = self.norm1(h)
        q = self.q_proj(hn).view(B, L, {n_heads}, {d // n_heads}).transpose(1, 2)
        k = self.k_proj(hn).view(B, L, {n_heads}, {d // n_heads}).transpose(1, 2)
        v = self.v_proj(hn).view(B, L, {n_heads}, {d // n_heads}).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt({float(d // n_heads)})
{attn_mask_code}
        attn_out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(B, L, {d})
        h = h + self.o_proj(attn_out)
        hn2 = self.norm2(h)
        h = h + self.mlp_down(F.relu(self.mlp_up(hn2)))
        h_norm = self.norm_f(h)
        return {output_line}


{weights_block}

def build_model():
    model = TrainableAdder()
    model.load_state_dict(WEIGHTS, strict=False)
    model.eval()
    metadata = {{
        "name": "{n_params}-Parameter Trained Adder",
        "author": "fblissjr",
        "params": {n_params},
        "architecture": "1L d={d} {n_heads}h ff={d_ff} reversed-digits{'  ALiBi' if use_alibi else ''}",
        "tricks": {tricks},
    }}
    return model, metadata


def add(model, a: int, b: int) -> int:
    sa = f"{{a:010d}}"[::-1]
    sb = f"{{b:010d}}"[::-1]
    seq = [int(c) for c in sa] + [10] + [int(c) for c in sb] + [11]
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        for _ in range(11):
            x = torch.tensor([seq], dtype=torch.long, device=device)
            logits = model(x)
            seq.append(logits[0, -1].argmax().item())
    c_digits = seq[22:33]
    return sum(d * (10**i) for i, d in enumerate(c_digits))
'''

    with open(filename, "w") as f:
        f.write(code)
    print(f"Exported! Run: uv run python verify.py {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="mask-warm", choices=list(CONFIGS.keys()))
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-steps", type=int, default=200000)
    parser.add_argument("--all-levels", action="store_true",
                        help="Train on all digit levels from start (no curriculum)")
    args = parser.parse_args()

    train(args.config, args.lr, args.batch_size, args.max_steps, args.all_levels)
