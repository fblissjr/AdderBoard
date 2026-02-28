"""
Training script for Track 2: Trained transformer adder.

Architecture: 1L decoder, d_model=7, n_head=1, d_ff=14
Input format: "0000000005+0000000007=" (zero-padded MSB-first, 10 digits each)
Output format: reversed sum digits + EOS (LSD-first for carry alignment)

Curriculum: 3 phases
  Phase 1: 1-3 digit numbers, 2K steps
  Phase 2: 1-6 digit numbers, 5K steps
  Phase 3: 1-10 digit numbers, 20K+ steps
"""

import math
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F


# --- Vocabulary ---
# 0-9: digits, 10: '+', 11: '=', 12: PAD, 13: EOS
VOCAB_SIZE = 14
TOK_PLUS = 10
TOK_EQ = 11
TOK_PAD = 12
TOK_EOS = 13


def encode_pair(a: int, b: int):
    """Encode a+b into input tokens and target tokens.

    Input: 10 digits of a (MSB) + '+' + 10 digits of b (MSB) + '='
    Target: reversed sum digits (LSD first) + EOS
    """
    a_str = f"{a:010d}"
    b_str = f"{b:010d}"

    input_toks = [int(c) for c in a_str] + [TOK_PLUS] + [int(c) for c in b_str] + [TOK_EQ]

    s = a + b
    s_str = f"{s:011d}"  # up to 11 digits (max 19999999998)
    target_toks = [int(c) for c in reversed(s_str)] + [TOK_EOS]

    return input_toks, target_toks


def make_batch(batch_size: int, max_digits: int, rng: random.Random):
    """Generate a batch of random addition pairs."""
    max_val = 10**max_digits - 1

    input_seqs = []
    target_seqs = []
    for _ in range(batch_size):
        a = rng.randint(0, max_val)
        b = rng.randint(0, max_val)
        inp, tgt = encode_pair(a, b)
        input_seqs.append(inp)
        target_seqs.append(tgt)

    # Full sequence: input + target (teacher forcing)
    # We predict target positions only
    full_seqs = []
    for inp, tgt in zip(input_seqs, target_seqs):
        full_seqs.append(inp + tgt)

    x = torch.tensor(full_seqs, dtype=torch.long)

    return x, len(input_seqs[0])  # return prompt_len for masking


# --- Model ---


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x):
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, d_ff: int):
        super().__init__()
        self.n_head = n_head
        self.d_head = d_model // n_head

        self.ln1 = RMSNorm(d_model)
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.o_proj = nn.Linear(d_model, d_model, bias=False)

        self.ln2 = RMSNorm(d_model)
        self.ff_up = nn.Linear(d_model, d_ff, bias=False)
        self.ff_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x, mask):
        B, L, D = x.shape
        h = self.ln1(x)

        q = self.q_proj(h).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        k = self.k_proj(h).view(B, L, self.n_head, self.d_head).transpose(1, 2)
        v = self.v_proj(h).view(B, L, self.n_head, self.d_head).transpose(1, 2)

        scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.d_head)
        scores = scores + mask
        attn = F.softmax(scores, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, L, D)
        out = self.o_proj(out)
        x = x + out

        h = self.ln2(x)
        x = x + self.ff_down(F.gelu(self.ff_up(h)))
        return x


class AdderTransformer(nn.Module):
    def __init__(self, d_model=7, n_head=1, d_ff=14, max_len=36):
        super().__init__()
        self.d_model = d_model
        self.embed = nn.Embedding(VOCAB_SIZE, d_model)
        self.pos_embed = nn.Embedding(max_len, d_model)
        self.block = TransformerBlock(d_model, n_head, d_ff)
        self.ln_f = RMSNorm(d_model)
        # Tie lm_head to embedding
        self.lm_head_bias = nn.Parameter(torch.zeros(VOCAB_SIZE))

    def forward(self, x):
        B, L = x.shape
        pos = torch.arange(L, device=x.device).unsqueeze(0)
        h = self.embed(x) + self.pos_embed(pos)

        mask = torch.triu(torch.full((L, L), float('-inf'), device=x.device), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L)

        h = self.block(h, mask)
        h = self.ln_f(h)

        # Tied weights: logits = h @ embed.weight.T + bias
        logits = F.linear(h, self.embed.weight, self.lm_head_bias)
        return logits

    def count_params(self):
        return sum(p.numel() for p in self.parameters())


# --- Training ---


def evaluate_model(model, max_digits=10, n_samples=200, rng=None):
    """Evaluate model on random 10-digit addition pairs."""
    model.eval()
    device = next(model.parameters()).device
    max_val = 10**max_digits - 1
    correct = 0

    with torch.no_grad():
        for _ in range(n_samples):
            a = rng.randint(0, max_val)
            b = rng.randint(0, max_val)
            expected = a + b

            inp, _ = encode_pair(a, b)
            seq = list(inp)

            for _ in range(12):  # 11 digits + EOS
                x = torch.tensor([seq], dtype=torch.long, device=device)
                logits = model(x)
                next_tok = logits[0, -1].argmax().item()
                seq.append(next_tok)
                if next_tok == TOK_EOS:
                    break

            # Decode: output tokens after the '=' are reversed digits
            out_toks = seq[len(inp):]
            # Remove EOS if present
            if out_toks and out_toks[-1] == TOK_EOS:
                out_toks = out_toks[:-1]

            # Reversed digits -> number
            result = 0
            for i, d in enumerate(out_toks):
                if 0 <= d <= 9:
                    result += d * (10 ** i)

            if result == expected:
                correct += 1

    return correct / n_samples


def train():
    device = torch.device('cpu')
    model = AdderTransformer(d_model=7, n_head=1, d_ff=14, max_len=36).to(device)

    print(f"Model parameters: {model.count_params()}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-2, weight_decay=0.01)

    # Curriculum phases: (max_digits, num_steps)
    phases = [
        (3, 2000),
        (6, 5000),
        (10, 20000),
    ]

    rng = random.Random(42)
    batch_size = 512
    step = 0
    best_acc = 0.0

    for phase_idx, (max_digits, num_steps) in enumerate(phases):
        print(f"\n--- Phase {phase_idx+1}: max_digits={max_digits}, steps={num_steps} ---")
        phase_start = time.time()

        for phase_step in range(num_steps):
            model.train()
            x, prompt_len = make_batch(batch_size, max_digits, rng)
            x = x.to(device)

            logits = model(x)

            # Shift: predict next token
            # logits[:, :-1] predicts x[:, 1:]
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = x[:, 1:].contiguous().clone()

            # Mask prompt positions (don't train on input tokens)
            # prompt_len tokens of input, we mask positions 0..prompt_len-2 in shifted labels
            shift_labels[:, :prompt_len - 1] = -100

            loss = F.cross_entropy(
                shift_logits.view(-1, VOCAB_SIZE),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1

            if step % 500 == 0:
                elapsed = time.time() - phase_start
                acc = evaluate_model(model, max_digits=10, n_samples=200, rng=random.Random(999))
                print(f"  step {step}, loss={loss.item():.4f}, "
                      f"10-digit acc={acc:.1%}, time={elapsed:.0f}s")
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), 'best_model.pt')
                    print(f"  ** New best: {acc:.1%} **")

                if acc >= 0.995:
                    print(f"  Reached {acc:.1%} -- stopping early!")
                    break

        # Reduce LR for next phase
        for pg in optimizer.param_groups:
            pg['lr'] *= 0.5

    # Extended training if needed
    if best_acc < 0.99:
        print(f"\n--- Extended training (grokking phase) ---")
        for pg in optimizer.param_groups:
            pg['lr'] = 1e-3

        for ext_step in range(80000):
            model.train()
            x, prompt_len = make_batch(batch_size, 10, rng)
            x = x.to(device)

            logits = model(x)
            shift_logits = logits[:, :-1].contiguous()
            shift_labels = x[:, 1:].contiguous().clone()
            shift_labels[:, :prompt_len - 1] = -100

            loss = F.cross_entropy(
                shift_logits.view(-1, VOCAB_SIZE),
                shift_labels.view(-1),
                ignore_index=-100,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            if step % 1000 == 0:
                acc = evaluate_model(model, max_digits=10, n_samples=200, rng=random.Random(999))
                print(f"  step {step}, loss={loss.item():.4f}, acc={acc:.1%}")
                if acc > best_acc:
                    best_acc = acc
                    torch.save(model.state_dict(), 'best_model.pt')
                    print(f"  ** New best: {acc:.1%} **")
                if acc >= 0.995:
                    print(f"  Reached {acc:.1%} -- stopping!")
                    break

    print(f"\nTraining complete. Best accuracy: {best_acc:.1%}")
    print(f"Total steps: {step}")


if __name__ == '__main__':
    train()
