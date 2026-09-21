import math
import torch
from torch import nn


class Subtoken(nn.Module):
    def __init__(self, d=32, r=16, k=4, attention=True):
        super().__init__()
        self.r, self.k, self.attention = r, k, attention
        self.project = nn.Linear(d, k * r)
        self.out = nn.Linear(r, d)

    def forward(self, x):
        z = self.project(x).unflatten(-1, (self.k, self.r))
        if self.attention:
            a = (z @ z.transpose(-1, -2) / math.sqrt(self.r)).softmax(-1)
            z = a @ z
        return self.out(z.mean(-2))


class Block(nn.Module):
    def __init__(self, method):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(32), nn.LayerNorm(32)
        self.seq = nn.MultiheadAttention(32, 4, dropout=0, batch_first=True)
        self.mix = (nn.Sequential(nn.Linear(32, 41), nn.GELU(), nn.Linear(41, 32))
                    if method == 'mlp' else Subtoken(attention=method == 'attention'))

    def forward(self, x):
        y = self.n1(x)
        mask = torch.ones(x.shape[1], x.shape[1], dtype=torch.bool, device=x.device).triu(1)
        x = x + self.seq(y, y, y, attn_mask=mask, need_weights=False)[0]
        return x + self.mix(self.n2(x))


class Model(nn.Module):
    def __init__(self, method):
        super().__init__()
        if method not in ('attention', 'pool', 'mlp'):
            raise ValueError(method)
        self.embed = nn.Embedding(19, 32)
        self.pos = nn.Embedding(16, 32)
        self.blocks = nn.Sequential(Block(method), Block(method))
        self.norm = nn.LayerNorm(32)
        self.head = nn.Linear(32, 19)

    def forward(self, x):
        h = self.embed(x) + self.pos(torch.arange(x.shape[1], device=x.device))
        return self.head(self.norm(self.blocks(h)))
