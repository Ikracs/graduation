import torch.nn as nn
import torch.nn.functional as F

from driving.audio2video.utils import modulate


class DenseFiLM(nn.Module):
    '''
    Feature-wise linear modulation (FiLM) generator.
    '''
    def __init__(self, dim):
        super(DenseFiLM, self).__init__()

        self.block = nn.Sequential(nn.Mish(), nn.Linear(dim, dim * 2))

    def forward(self, x):
        scale_shift = self.block(x).unsqueeze(1)
        return scale_shift.chunk(2, dim=-1)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, dim, nhead, hidden_dim=2048, dropout=0.1, batch_first=False, norm_first=True, rotary=None):
        super(TransformerEncoderLayer, self).__init__()

        self.attn = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=batch_first)
        self.dropout = nn.Dropout(dropout)

        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

        self.norm_first = norm_first
        self.rotary = rotary

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        x = src
        if self.norm_first:
            x = x + self.sa_block(self.norm1(x), src_mask, src_key_padding_mask)
            x = x + self.ff_block(self.norm2(x))
        else:
            x = self.norm1(x + self.sa_block(x, src_mask, src_key_padding_mask))
            x = self.norm2(x + self.ff_block(x))
        return x

    def sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.rotary is not None else x
        x = self.attn(qk, qk, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout(x)

    def ff_block(self, x):
        x = self.dropout1(F.gelu(self.linear1(x)))
        return self.dropout2(self.linear2(x))


class TransformerDecoderLayer(nn.Module):
    def __init__(self, dim, nhead, hidden_dim=2048, dropout=0.1, batch_first=False, norm_first=True, rotary=None):
        super(TransformerDecoderLayer, self).__init__()

        self.sa = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=batch_first)
        self.dropout1 = nn.Dropout(dropout)

        self.ca = nn.MultiheadAttention(dim, nhead, dropout=dropout, batch_first=batch_first)
        self.dropout2 = nn.Dropout(dropout)
        
        self.linear1 = nn.Linear(dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, dim)
        self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)

        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

        self.film1 = DenseFiLM(dim)
        self.film2 = DenseFiLM(dim)
        self.film3 = DenseFiLM(dim)

        self.norm_first = norm_first
        self.rotary = rotary

    def forward(self, tgt, mem, t, tgt_mask=None, mem_mask=None, tgt_key_padding_mask=None, mem_key_padding_mask=None):
        x = tgt
        if self.norm_first:
            x = x + modulate(self.sa_block(self.norm1(x), tgt_mask, tgt_key_padding_mask), *self.film1(t))
            x = x + modulate(self.ca_block(self.norm2(x), mem, mem_mask, mem_key_padding_mask), *self.film2(t))
            x = x + modulate(self.ff_block(self.norm3(x)), *self.film3(t))
        else:
            x = self.norm1(x + modulate(self.sa_block(x, tgt_mask, tgt_key_padding_mask), *self.film1(t)))
            x = self.norm2(x + modulate(self.ca_block(x, mem, mem_mask, mem_key_padding_mask)), *self.film2(t))
            x = self.norm3(x + modulate(self.ff_block(x), *self.film3(t)))
        return x

    def sa_block(self, x, attn_mask, key_padding_mask):
        qk = self.rotary.rotate_queries_or_keys(x) if self.rotary is not None else x
        x = self.sa(qk, qk, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout1(x)

    def ca_block(self, x, mem, attn_mask, key_padding_mask):
        q = self.rotary.rotate_queries_or_keys(x) if self.rotary is not None else x
        k = self.rotary.rotate_queries_or_keys(mem) if self.rotary is not None else mem
        x = self.ca(q, k, mem, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]
        return self.dropout2(x)

    def ff_block(self, x):
        x = self.dropout3(F.gelu(self.linear1(x)))
        return self.dropout4(self.linear2(x))
