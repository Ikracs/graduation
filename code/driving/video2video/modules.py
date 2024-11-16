import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn.utils import spectral_norm

from driving.video2video.sync_batchnorm import SynchronizedBatchNorm2d as BatchNorm2d
from driving.video2video.sync_batchnorm import SynchronizedBatchNorm3d as BatchNorm3d


class ResBlock2d(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlock2d, self).__init__()
        self.learned_shortcut = (in_features != out_features)

        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        self.norm1 = BatchNorm2d(in_features, affine=True)
        self.norm2 = BatchNorm2d(in_features, affine=True)

        if self.learned_shortcut:
            self.skip = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
            self.norm = BatchNorm2d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.learned_shortcut:
            x = self.skip(self.norm(x))
        out += x
        return out


class ResBlock3d(nn.Module):
    def __init__(self, in_features, out_features):
        super(ResBlock3d, self).__init__()
        self.learned_shortcut = (in_features != out_features)

        self.conv1 = nn.Conv3d(in_features, in_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_features, out_features, kernel_size=3, padding=1)
        self.norm1 = BatchNorm3d(in_features, affine=True)
        self.norm2 = BatchNorm3d(in_features, affine=True)

        if self.learned_shortcut:
            self.skip = nn.Conv3d(in_features, out_features, kernel_size=1, bias=False)
            self.norm = BatchNorm3d(in_features, affine=True)

    def forward(self, x):
        out = self.norm1(x)
        out = F.relu(out)
        out = self.conv1(out)
        out = self.norm2(out)
        out = F.relu(out)
        out = self.conv2(out)
        if self.learned_shortcut:
            x = self.skip(self.norm(x))
        out += x
        return out


class UpBlock2d(nn.Module):
    def __init__(self, in_features, out_features):
        super(UpBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.up = nn.Upsample(scale_factor=2)

    def forward(self, x):
        out = self.conv(self.up(x))
        out = F.relu(self.norm(out))
        return out


class UpBlock3d(nn.Module):
    def __init__(self, in_features, out_features):
        super(UpBlock3d, self).__init__()
        self.conv = nn.Conv3d(in_features, out_features, kernel_size=3, padding=1)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.up = nn.Upsample(scale_factor=(1, 2, 2))

    def forward(self, x):
        out = self.conv(self.up(x))
        out = F.relu(self.norm(out))
        return out


class DownBlock2d(nn.Module):
    def __init__(self, in_features, out_features):
        super(DownBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        self.norm = BatchNorm2d(out_features, affine=True)
        self.pool = nn.AvgPool2d(kernel_size=2)

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(self.norm(out))
        out = self.pool(out)
        return out


class DownBlock3d(nn.Module):
    def __init__(self, in_features, out_features):
        super(DownBlock3d, self).__init__()
        self.conv = nn.Conv3d(in_features, out_features, kernel_size=3, padding=1)
        self.norm = BatchNorm3d(out_features, affine=True)
        self.pool = nn.AvgPool3d(kernel_size=(1, 2, 2))

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(self.norm(out))
        out = self.pool(out)
        return out


class SimpleBlock2d(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleBlock2d, self).__init__()
        self.conv = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        self.norm = BatchNorm2d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(self.norm(out))
        return out


class SimpleBlock3d(nn.Module):
    def __init__(self, in_features, out_features):
        super(SimpleBlock3d, self).__init__()
        self.conv = nn.Conv3d(in_features, out_features, kernel_size=3, padding=1)
        self.norm = BatchNorm3d(out_features, affine=True)

    def forward(self, x):
        out = self.conv(x)
        out = F.relu(self.norm(out))
        return out


class Encoder2D(nn.Module):
    def __init__(self, block_expansion, num_blocks, max_features):
        super(Encoder2D, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock2d(in_features, out_features))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder2D(nn.Module):
    def __init__(self, block_expansion, num_blocks, max_features):
        super(Decoder2D, self).__init__()

        up_blocks = []
        for i in range(num_blocks):
            in_features = (1 if i == 0 else 2) * min(max_features, block_expansion * (2 ** (num_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_blocks - i - 1)))
            up_blocks.append(UpBlock2d(in_features, out_features))
        self.up_blocks = nn.ModuleList(up_blocks)

        in_features = out_features + block_expansion
        self.conv = nn.Conv2d(in_features, block_expansion, kernel_size=3, padding=1)
        self.norm = BatchNorm2d(block_expansion, affine=True)

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            out = torch.cat([out, x.pop()], dim=1)
        out = self.conv(out)
        out = F.relu(self.norm(out))
        return out


class Hourglass2D(nn.Module):
    def __init__(self, in_features, num_blocks, max_features):
        super(Hourglass2D, self).__init__()
        self.encoder = Encoder2D(in_features, num_blocks, max_features)
        self.decoder = Decoder2D(in_features, num_blocks, max_features)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Encoder3D(nn.Module):
    def __init__(self, block_expansion, num_blocks, max_features):
        super(Encoder3D, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            down_blocks.append(DownBlock3d(in_features, out_features))
        self.down_blocks = nn.ModuleList(down_blocks)

    def forward(self, x):
        outs = [x]
        for down_block in self.down_blocks:
            outs.append(down_block(outs[-1]))
        return outs


class Decoder3D(nn.Module):
    def __init__(self, block_expansion, num_blocks, max_features):
        super(Decoder3D, self).__init__()

        up_blocks = []
        for i in range(num_blocks):
            in_features = (1 if i == 0 else 2) * min(max_features, block_expansion * (2 ** (num_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_blocks - i - 1)))
            up_blocks.append(UpBlock3d(in_features, out_features))
        self.up_blocks = nn.ModuleList(up_blocks)

        in_features = out_features + block_expansion
        self.conv = nn.Conv3d(in_features, block_expansion, kernel_size=3, padding=1)
        self.norm = BatchNorm3d(block_expansion, affine=True)

    def forward(self, x):
        out = x.pop()
        for up_block in self.up_blocks:
            out = up_block(out)
            out = torch.cat([out, x.pop()], dim=1)
        out = self.conv(out)
        out = F.relu(self.norm(out))
        return out


class Hourglass3D(nn.Module):
    def __init__(self, in_features, num_blocks, max_features):
        super(Hourglass3D, self).__init__()
        self.encoder = Encoder3D(in_features, num_blocks, max_features)
        self.decoder = Decoder3D(in_features, num_blocks, max_features)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class SPADE2D(nn.Module):
    def __init__(self, in_features, prior_channel):
        super(SPADE2D, self).__init__()

        self.norm = nn.InstanceNorm2d(in_features, affine=False)

        self.conv = nn.Conv2d(prior_channel, in_features // 2, kernel_size=3, padding=1)
        self.gamma = nn.Conv2d(in_features // 2, in_features, kernel_size=3, padding=1)
        self.beta = nn.Conv2d(in_features // 2, in_features, kernel_size=3, padding=1)

    def forward(self, x, prior):
        latent = F.interpolate(prior, size=x.shape[2:])
        latent = F.relu(self.conv(latent))
        gamma, beta = self.gamma(latent), self.beta(latent)
        out = self.norm(x) * (1 + gamma) + beta
        return out


class AdaIN3D(nn.Module):
    def __init__(self, in_features, style_channel):
        super(AdaIN3D, self).__init__()

        self.norm = nn.InstanceNorm3d(in_features, affine=False)

        self.middle = nn.Linear(style_channel, in_features // 2)
        self.gamma = nn.Linear(in_features // 2, in_features)
        self.beta = nn.Linear(in_features // 2, in_features)

    def forward(self, x, style):
        latent = F.relu(self.middle(style))
        gamma = self.gamma(latent)[:, :, None, None, None]
        beta = self.beta(latent)[:, :, None, None, None]
        out = self.norm(x) * (1 + gamma) + beta
        return out


class SPADEResBlock2d(nn.Module):
    def __init__(self, in_features, out_features, prior_channel, spectral=False):
        super(SPADEResBlock2d, self).__init__()
        self.learned_shortcut = (in_features != out_features)

        self.conv1 = nn.Conv2d(in_features, in_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(in_features, out_features, kernel_size=3, padding=1)
        self.norm1 = SPADE2D(in_features, prior_channel)
        self.norm2 = SPADE2D(in_features, prior_channel)

        if self.learned_shortcut:
            self.skip = nn.Conv2d(in_features, out_features, kernel_size=1, bias=False)
            self.norm = SPADE2D(in_features, prior_channel)

        if spectral:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)

            if self.learned_shortcut:
                self.skip = spectral_norm(self.skip)

    def forward(self, x, prior):
        out = self.norm1(x, prior)
        out = F.leaky_relu(out, 0.2)
        out = self.conv1(out)
        out = self.norm2(out, prior)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        if self.learned_shortcut:
            x = self.skip(self.norm(x, prior))
        out += x
        return out


class StyleResBlock3d(nn.Module):
    def __init__(self, in_features, out_features, style_channel, spectral=False):
        super(StyleResBlock3d, self).__init__()
        self.learned_shortcut = (in_features != out_features)

        self.conv1 = nn.Conv3d(in_features, in_features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(in_features, out_features, kernel_size=3, padding=1)
        self.norm1 = AdaIN3D(in_features, style_channel)
        self.norm2 = AdaIN3D(in_features, style_channel)

        if self.learned_shortcut:
            self.skip = nn.Conv3d(in_features, out_features, kernel_size=1, bias=False)
            self.norm = AdaIN3D(in_features, style_channel)

        if spectral:
            self.conv1 = spectral_norm(self.conv1)
            self.conv2 = spectral_norm(self.conv2)

            if self.learned_shortcut:
                self.skip = spectral_norm(self.skip)

    def forward(self, x, style):
        out = self.norm1(x, style)
        out = F.leaky_relu(out, 0.2)
        out = self.conv1(out)
        out = self.norm2(out, style)
        out = F.leaky_relu(out, 0.2)
        out = self.conv2(out)
        if self.learned_shortcut:
            x = self.skip(self.norm(x, style))
        out += x
        return out


class SelfAttention(nn.Module):
    def __init__(self, in_features, hidden_dim, num_heads):
        super(SelfAttention, self).__init__()

        self.num_heads = num_heads

        self.q = nn.Linear(in_features, hidden_dim, bias=False)
        self.k = nn.Linear(in_features, hidden_dim, bias=False)
        self.v = nn.Linear(in_features, hidden_dim, bias=False)

        self.out = nn.Linear(hidden_dim, in_features, bias=False)

    def forward(self, x):
        Q = self.q(x).unflatten(2, (self.num_heads, -1)).permute(0, 2, 1, 3)
        K = self.k(x).unflatten(2, (self.num_heads, -1)).permute(0, 2, 1, 3)
        V = self.v(x).unflatten(2, (self.num_heads, -1)).permute(0, 2, 1, 3)

        score_mat = torch.einsum('BHQD, BHKD -> BHQK', Q, K)
        score_mat = F.softmax(score_mat / math.sqrt(Q.shape[-1]), dim=-1)

        out = torch.einsum('BHQK, BHKD -> BHQD', score_mat, V)
        return self.out(out.permute(0, 2, 1, 3).flatten(2))


class SpatialAttnBlock(nn.Module):
    def __init__(self, in_features, hidden_dim, num_heads, expansion, dropout=0.1):
        super(SpatialAttnBlock, self).__init__()

        self.sa = SelfAttention(in_features, hidden_dim, num_heads)
        self.ff = nn.Sequential(
            nn.Linear(in_features, in_features * expansion),
            nn.GELU(),
            nn.Linear(in_features * expansion, in_features)
        )

        self.norm1 = nn.LayerNorm(in_features)
        self.norm2 = nn.LayerNorm(in_features)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        out = x.flatten(2).permute(0, 2, 1)
        out = self.dropout1(self.sa(self.norm1(out)) + out)
        out = self.dropout2(self.ff(self.norm2(out)) + out)
        return out.permute(0, 2, 1).unflatten(2, x.shape[2:])
