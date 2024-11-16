import math

import torch
import torch.nn as nn
from einops import rearrange, repeat


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=500, dropout=0.1, batch_first=False):
        super(PositionalEncoding, self).__init__()

        self.batch_first = batch_first

        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        factor = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * factor)
        pe[:, 1::2] = torch.cos(position * factor)
        pe = pe.unsqueeze(0).transpose(0, 1)

        self.register_buffer("pe", pe)

    def forward(self, x):
        pe = self.pe.permute(1, 0, 2)[:, :x.shape[1]] if self.batch_first else self.pe[:x.shape[0]]
        return self.dropout(x + pe)


class TimeEmbedding(nn.Module):
    def __init__(self, dim):
        super(TimeEmbedding, self).__init__()

        self.register_buffer('factor', torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim)))

    def forward(self, x):
        emb_sin = torch.sin(x[:, None] * self.factor[None, :])
        emb_cos = torch.cos(x[:, None] * self.factor[None, :])
        return torch.stack((emb_sin, emb_cos), dim=-1).flatten(1, 2)


class RotaryPositionEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super(RotaryPositionEmbedding, self).__init__()

        self.register_buffer("freqs", 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim)))

        self.cache = {}

    def rotate_queries_or_keys(self, t, seq_dim=-2):
        seq_len = t.shape[seq_dim]
        freqs = self.forward(torch.arange(seq_len, device=t.device), seq_len)
        return apply_rotary_emb(freqs, t)

    def forward(self, t, cache_key=None):
        if cache_key is not None and cache_key in self.cache:
            return self.cache[cache_key]

        freqs = self.freqs
        freqs = torch.einsum("..., f -> ... f", t.type(freqs.dtype), freqs)
        freqs = repeat(freqs, "... n -> ... (n r)", r=2)

        if cache_key is not None:
            self.cache[cache_key] = freqs

        return freqs

def apply_rotary_emb(freqs, t, start_index=0):
    freqs = freqs.to(t)
    rot_dim = freqs.shape[-1]
    end_index = start_index + rot_dim
    assert(rot_dim <= t.shape[-1]), f"feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}"
    t_left, t, t_right = (t[..., :start_index], t[..., start_index: end_index], t[..., end_index:])
    t = (t * freqs.cos()) + (rotate_half(t) * freqs.sin())
    return torch.cat((t_left, t, t_right), dim=-1)

def rotate_half(x):
    x = rearrange(x, "... (d r) -> ... d r", r=2)
    x1, x2 = x.unbind(dim=-1)
    x = torch.stack((-x2, x1), dim=-1)
    return rearrange(x, "... d r -> ... (d r)")
