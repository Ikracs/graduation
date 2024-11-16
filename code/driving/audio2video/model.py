import torch
import numpy as np
import torch.nn as nn

from driving.audio2video.modules import PositionalEncoding, TimeEmbedding, RotaryPositionEmbedding
from driving.audio2video.network import TransformerEncoderLayer, TransformerDecoderLayer


class PMTransformer(nn.Module):
    '''
    Pose-Motion Transformer.
    '''
    def __init__(self, dim, zdim, cond_dim, seq_n, pre_k, nhead=4, encoder_layers=4, decoder_layers=4, dropout=0.1, use_rotary=True):
        super(PMTransformer, self).__init__()

        self.rotary = None
        self.abs_pe = nn.Identity()

        if use_rotary:
            self.rotary = RotaryPositionEmbedding(dim=zdim)
        else:
            self.abs_pe = PositionalEncoding(
                dim=zdim,
                max_len=seq_n,
                dropout=dropout,
                batch_first=True
            )

        self.cond_proj = nn.Linear(cond_dim, zdim)
        self.null_cond = nn.Parameter(torch.randn(cond_dim))

        self.cond_encoder = nn.Sequential()
        for _ in range(encoder_layers):
            self.cond_encoder.append(
                TransformerEncoderLayer(
                    dim=zdim,
                    nhead=nhead,
                    hidden_dim=zdim * 4,
                    dropout=dropout,
                    batch_first=True,
                    rotary=self.rotary
                )
            )

        self.time_proj = nn.Sequential(
            TimeEmbedding(zdim),
            nn.Linear(zdim, 4 * zdim),
            nn.Mish(),
            nn.Linear(4 * zdim, zdim)
        )

        attn_mask = torch.cat((torch.zeros(pre_k + 1, dtype=torch.bool), torch.ones(seq_n - 1, dtype=torch.bool)))
        self.register_buffer('attn_mask', torch.stack([torch.roll(attn_mask, i) for i in range(seq_n)], dim=0))

        self.decoder = nn.ModuleList()
        for _ in range(decoder_layers):
            self.decoder.append(
                TransformerDecoderLayer(
                    dim=zdim,
                    nhead=nhead,
                    hidden_dim=zdim * 4,
                    dropout=dropout,
                    batch_first=True,
                    rotary=self.rotary
                )
            )

        self.final = nn.Linear(zdim, dim)

    def guided_forward(self, x, cond, time, padding_mask, guidance):
        uncond_out = self.forward(x, cond, time, padding_mask, cond_drop_prob=1.0)
        cond_out = self.forward(x, cond, time, padding_mask, cond_drop_prob=0.0)

        return uncond_out + (cond_out - uncond_out) * guidance

    def forward(self, x, cond, time, padding_mask=None, cond_drop_prob=0.0):
        padding_mask = torch.zeros_like(cond) if padding_mask is None else padding_mask
        cond_keep_mask = (torch.rand(x.shape[0]) > cond_drop_prob).to(x.device).unsqueeze(-1)
        keep_mask = torch.logical_and(cond_keep_mask, torch.logical_not(padding_mask))
        cond = torch.where(keep_mask.unsqueeze(-1), cond, self.null_cond.repeat(x.shape[0], 1, 1))

        cond_tokens = self.cond_proj(cond)
        cond_tokens = self.abs_pe(cond_tokens)
        cond_tokens = self.cond_encoder(cond_tokens)

        t = self.time_proj(time)

        x = self.abs_pe(x)
        for layer in self.decoder:
            x = layer(x, cond_tokens, t, mem_mask=self.attn_mask)

        return self.final(x)
