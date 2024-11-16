import torch.nn as nn
import torch.nn.functional as F


class DownBlock2d(nn.Module):
    def __init__(self, in_features, out_features, norm=False, kernel_size=4, pool=False, spectral=False):
        super(DownBlock2d, self).__init__()

        self.conv = nn.Conv2d(in_channels=in_features, out_channels=out_features, kernel_size=kernel_size)

        if spectral:
            self.conv = nn.utils.spectral_norm(self.conv)

        self.norm = nn.InstanceNorm2d(out_features, affine=True) if norm else nn.Identity()
        self.pool = nn.AvgPool2d((2, 2)) if pool else nn.Identity()

    def forward(self, x):
        out = self.conv(x)
        out = F.leaky_relu(self.norm(out), 0.2)
        out = self.pool(out)
        return out


class Discriminator(nn.Module):
    def __init__(self, num_channel, block_expansion, num_blocks, max_features, spectral=False):
        super(Discriminator, self).__init__()

        down_blocks = []
        for i in range(num_blocks):
            down_blocks.append(
                DownBlock2d(num_channel if i == 0 else min(max_features, block_expansion * (2 ** i)),
                            min(max_features, block_expansion * (2 ** (i + 1))),
                            norm=(i != 0), kernel_size=4, pool=(i != num_blocks - 1), spectral=spectral)
            )

        self.down_blocks = nn.ModuleList(down_blocks)
        self.conv = nn.Conv2d(self.down_blocks[-1].conv.out_channels, out_channels=1, kernel_size=1)

        if spectral:
            self.conv = nn.utils.spectral_norm(self.conv)

    def forward(self, x):
        out, feature_maps = x, []
        for down_block in self.down_blocks:
            out = down_block(out)
            feature_maps.append(out)
        prediction = self.conv(out)
        return feature_maps, prediction


class MultiScaleDiscriminator(nn.Module):
    def __init__(self, scales, **kwargs):
        super(MultiScaleDiscriminator, self).__init__()

        self.scales = scales
        self.discs = nn.ModuleDict({str(scale).replace('.', '-'): Discriminator(**kwargs) for scale in scales})

    def forward(self, x):
        out_dict = {}
        for scale, disc in self.discs.items():
            scale = str(scale).replace('-', '.')
            key = 'prediction_' + scale
            feature_maps, prediction_map = disc(x[key])
            out_dict['feature_maps_' + scale] = feature_maps
            out_dict['prediction_map_' + scale] = prediction_map
        return out_dict
