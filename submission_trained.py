"""Trained 162-parameter transformer adder.

Trained from scratch: AdamW + reversed digits (LSB-first).
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNormFree(nn.Module):
    def forward(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)


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


class TrainableAdder(nn.Module):
    def __init__(self):
        super().__init__()
        self.d = 3
        self.n_heads = 3
        self.d_head = 1
        self.embed = nn.Embedding(12, 3)
        self.q_proj = nn.Linear(3, 3, bias=False)
        self.k_proj = nn.Linear(3, 3, bias=False)
        self.v_proj = nn.Linear(3, 3, bias=False)
        self.o_proj = nn.Linear(3, 3, bias=False)
        self.mlp_up = nn.Linear(3, 6, bias=True)
        self.mlp_down = nn.Linear(6, 3, bias=False)
        self.norm1 = RMSNormFree()
        self.norm2 = RMSNormFree()
        self.norm_f = RMSNormFree()
        self.lm_head = nn.Linear(3, 12, bias=True)
        self.register_buffer('fixed_mask', build_fixed_mask())

    def forward(self, x):
        B, L = x.shape
        h = self.embed(x)
        hn = self.norm1(h)
        q = self.q_proj(hn).view(B, L, 3, 1).transpose(1, 2)
        k = self.k_proj(hn).view(B, L, 3, 1).transpose(1, 2)
        v = self.v_proj(hn).view(B, L, 3, 1).transpose(1, 2)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(1.0)
        scores = scores + self.fixed_mask[:, :L, :L].unsqueeze(0)
        attn_out = (F.softmax(scores, dim=-1) @ v).transpose(1, 2).reshape(B, L, 3)
        h = h + self.o_proj(attn_out)
        hn2 = self.norm2(h)
        h = h + self.mlp_down(F.relu(self.mlp_up(hn2)))
        h_norm = self.norm_f(h)
        return self.lm_head(h_norm)


WEIGHTS = {
    'embed.weight': torch.tensor([0.25090765953063965, 0.05607781931757927, 0.014350642450153828, 0.25568366050720215, 0.061973936855793, 0.03368891030550003, 0.22687062621116638, -0.021619923412799835, -0.12829327583312988, 0.23255746066570282, -0.013515417464077473, -0.1048656776547432, 0.237169548869133, -0.007034439593553543, -0.08395978808403015, 0.24121685326099396, -0.0007155911880545318, -0.06470540165901184, 0.24503976106643677, 0.0044853524304926395, -0.045980218797922134, 0.24866829812526703, 0.009805049747228622, -0.027725325897336006, 0.25293901562690735, 0.015447382815182209, -0.009664037264883518, 0.25674968957901, 0.01990293152630329, 0.007876046933233738, 0.5379047393798828, 0.4266173541545868, -0.06282509863376617, 0.17659910023212433, 0.2717416286468506, -0.09145602583885193], dtype=torch.float32).reshape((12, 3)),
    'q_proj.weight': torch.tensor([0.17339178919792175, 0.058474231511354446, -0.053214773535728455, -0.13797833025455475, 0.6283875703811646, -0.23575830459594727, 7.943783566588536e-05, -6.719836528645828e-05, 0.00011687590449582785], dtype=torch.float32).reshape((3, 3)),
    'k_proj.weight': torch.tensor([0.15214289724826813, -0.12405259162187576, 0.3198838233947754, 0.008299211971461773, 1.472868800163269, -1.8613412380218506, 0.00013436820881906897, 7.06532591721043e-05, 9.372107888339087e-05], dtype=torch.float32).reshape((3, 3)),
    'v_proj.weight': torch.tensor([-0.003932340070605278, 0.2978977560997009, 0.8117371201515198, 0.10671859234571457, -0.5053172707557678, 0.23289620876312256, -0.02941049262881279, -0.0513446219265461, -0.37107759714126587], dtype=torch.float32).reshape((3, 3)),
    'o_proj.weight': torch.tensor([-0.2406831830739975, -0.13967576622962952, 0.06593146920204163, 0.8646777272224426, 0.5566954016685486, 0.18276150524616241, -0.1575557142496109, -0.08548810333013535, 0.3722631633281708], dtype=torch.float32).reshape((3, 3)),
    'mlp_up.weight': torch.tensor([0.370336651802063, -0.43292850255966187, 0.11066335439682007, 0.38288429379463196, 0.2593104839324951, 0.03875504806637764, 0.06975443661212921, -0.6498867273330688, 0.37541577219963074, 0.337531715631485, -0.36362871527671814, 0.2349330484867096, -0.3799830675125122, 0.12681032717227936, -0.008131003938615322, -0.0005042640259489417, 0.00066546187736094, 8.310584962600842e-05], dtype=torch.float32).reshape((6, 3)),
    'mlp_up.bias': torch.tensor([0.19872090220451355, -0.17853491008281708, -0.1054936945438385, -0.17419807612895966, 0.6443207263946533, -0.0014440951636061072], dtype=torch.float32).reshape((6,)),
    'mlp_down.weight': torch.tensor([0.06072515994310379, -0.3430235683917999, -0.11973056197166443, -0.3678515553474426, -0.08072155714035034, -0.000690398970618844, 0.15042263269424438, -0.1792101114988327, 0.32240474224090576, -0.2101515382528305, -0.3099268972873688, -0.0014447529101744294, 0.4824496805667877, -0.5149084329605103, -0.5278264284133911, -0.26022404432296753, 0.8636730313301086, -0.0003682384267449379], dtype=torch.float32).reshape((3, 6)),
    'lm_head.weight': torch.tensor([-16.51704216003418, 5.105659484863281, -5.354624271392822, -18.288270950317383, 0.433194637298584, 8.126330375671387, -10.866294860839844, -3.1336021423339844, 16.521377563476562, 5.973390102386475, -2.6708831787109375, 23.040987014770508, 6.406068801879883, -15.605461120605469, 7.811237812042236, 8.023282051086426, -12.535080909729004, -3.7050631046295166, 10.940877914428711, -2.529020309448242, -8.695941925048828, 14.088859558105469, 13.352066040039062, -3.265382766723633, 5.868876934051514, 14.531843185424805, -14.462445259094238, -5.269567012786865, 9.736734390258789, -12.945984840393066, 0.11707933992147446, 0.22455792129039764, 0.23162391781806946, 0.1091575026512146, 0.22191984951496124, 0.2391137182712555], dtype=torch.float32).reshape((12, 3)),
    'lm_head.bias': torch.tensor([-2.526684045791626, -1.2569266557693481, 1.2658332586288452, 3.891775131225586, 0.3549794554710388, 3.7392425537109375, 3.0429461002349854, -1.9760853052139282, -6.187828063964844, -0.8619504570960999, -0.2859551012516022, -0.28772875666618347], dtype=torch.float32).reshape((12,))
}


def build_model():
    model = TrainableAdder()
    model.load_state_dict(WEIGHTS, strict=False)
    model.eval()
    metadata = {
        "name": "162-Parameter Trained Adder",
        "author": "fblissjr",
        "params": 162,
        "architecture": "1L d=3 3h ff=6 reversed-digits",
        "tricks": ['Reversed LSB-First', 'Teacher Forcing', 'RMSNorm', 'Fixed Mask (hand-coded routing)'],
    }
    return model, metadata


def add(model, a: int, b: int) -> int:
    sa = f"{a:010d}"[::-1]
    sb = f"{b:010d}"[::-1]
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
