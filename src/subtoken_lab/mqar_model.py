"""Variable-length models; the original experiment remains byte-for-byte intact."""
import math
import torch
from torch import nn
from .model import Subtoken

METHODS = ('mlp', 'pool', 'nonlinear', 'attention')


class NonlinearPool(Subtoken):
    def __init__(self, d=32, r=16, k=4):
        super().__init__(d, r, k, attention=False)

    def forward(self, x):
        z = self.project(x).unflatten(-1, (self.k, self.r))
        return self.out(torch.nn.functional.gelu(z).mean(-2))


class RecallModel(nn.Module):
    def __init__(self, method, d=32):
        super().__init__()
        if method not in METHODS or d not in (32, 48):
            raise ValueError((method, d))
        # Create ALL shared weights before architecture-dependent draws.
        self.embed = nn.Embedding(41, d, padding_idx=0)
        self.seq = nn.ModuleList([nn.MultiheadAttention(d, 4, dropout=0, batch_first=True) for _ in range(2)])
        self.n1 = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.n2 = nn.ModuleList([nn.LayerNorm(d) for _ in range(2)])
        self.norm, self.head = nn.LayerNorm(d), nn.Linear(d, 41)
        r, hidden = d // 2, 5 * d // 4 + 1
        self.mix = nn.ModuleList([
            nn.Sequential(nn.Linear(d, hidden), nn.GELU(), nn.Linear(hidden, d)) if method == 'mlp'
            else NonlinearPool(d, r) if method == 'nonlinear'
            else Subtoken(d, r, attention=method == 'attention') for _ in range(2)])
        self.d = d

    def forward(self, tokens, padding_mask=None):
        length = tokens.shape[1]
        positions = torch.arange(length, device=tokens.device, dtype=self.embed.weight.dtype)[:, None]
        frequency = torch.exp(torch.arange(0, self.d, 2, device=tokens.device) * (-math.log(10000.0) / self.d))
        phase = positions * frequency
        pos = torch.stack((phase.sin(), phase.cos()), dim=-1).flatten(-2)
        h = self.embed(tokens) + pos
        padding = tokens.eq(0) if padding_mask is None else padding_mask
        causal = torch.ones(length, length, device=tokens.device, dtype=torch.bool).triu(1)
        for seq, n1, n2, mix in zip(self.seq, self.n1, self.n2, self.mix):
            y = n1(h)
            h = h + seq(y, y, y, attn_mask=causal, key_padding_mask=padding, need_weights=False)[0]
            h = h + mix(n2(h))
        return self.head(self.norm(h))
