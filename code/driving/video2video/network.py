from collections import Iterable

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from driving.video2video.modules import SimpleBlock2d, DownBlock2d
from driving.video2video.modules import ResBlock3d, SPADEResBlock2d
from driving.video2video.modules import Hourglass3D, StyleResBlock3d
from driving.video2video.utils import bins2degree

from utils import apply_imagenet_normalization
from utils import MODEL_PATH


class VolumeEncoder(nn.Module):
    def __init__(self, img_channel, block_expansion, max_features, num_down_blocks, reshape_channel, num_resblocks):
        super(VolumeEncoder, self).__init__()

        self.conv = SimpleBlock2d(img_channel, block_expansion)

        self.down_blocks = nn.Sequential()
        for i in range(num_down_blocks):
            in_features = min(max_features, block_expansion * (2 ** i))
            out_features = min(max_features, block_expansion * (2 ** (i + 1)))
            self.down_blocks.append(DownBlock2d(in_features, out_features))

        self.neck = nn.Conv2d(in_channels=out_features, out_channels=max_features, kernel_size=1, stride=1)

        self.reshape = lambda x: x.unflatten(1, (reshape_channel, -1))

        self.resblocks = nn.Sequential()
        for _ in range(num_resblocks):
            self.resblocks.append(ResBlock3d(reshape_channel, reshape_channel))

    def forward(self, x):
        out = self.conv(x)
        out = self.down_blocks(out)
        out = self.neck(out)
        out = self.reshape(out)
        out = self.resblocks(out)
        return out


class SPADEDecoder(nn.Module):
    def __init__(self, img_channel, block_expansion, max_features, num_up_blocks, num_resblocks):
        super(SPADEDecoder, self).__init__()

        self.reshape = lambda x: x.flatten(1, 2)

        self.resblocks = nn.Sequential()
        for _ in range(num_resblocks):
            self.resblocks.append(SPADEResBlock2d(max_features, max_features, max_features, spectral=True))

        out_features = block_expansion * (2 ** num_up_blocks)
        self.neck = SimpleBlock2d(max_features, out_features)

        self.up = nn.Upsample(scale_factor=2)

        self.up_blocks = nn.Sequential()
        for i in range(num_up_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_up_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_up_blocks - i - 1)))
            self.up_blocks.append(SPADEResBlock2d(in_features, out_features, max_features, spectral=True))

        self.final = nn.Conv2d(out_features, img_channel, kernel_size=(3, 3), padding=1)

    def forward(self, x):
        out = x = self.reshape(x)
        for resblock in self.resblocks:
            out = resblock(out, x)
        out = self.neck(out)
        for up_block in self.up_blocks:
            out = up_block(self.up(out), x)
        out = F.leaky_relu(out, 0.2)
        out = self.final(out)
        return out.sigmoid()


class IdentityExtractor(nn.Module):
    def __init__(self, hidden_dim, backbone='resnet18d'):
        super(IdentityExtractor, self).__init__()
        self.network = timm.create_model(
            backbone, pretrained=True,
            pretrained_cfg_overlay=dict(file=MODEL_PATH[backbone]),
            num_classes=hidden_dim
        )

    def forward(self, x):
        return self.network(apply_imagenet_normalization(x))


class PMEstimator(nn.Module):
    '''
    Pose-Motion Estimator.
    '''
    def __init__(self, hidden_dim, angle_bins=None, backbone='resnet50d'):
        super(PMEstimator, self).__init__()
        self.backbone = timm.create_model(
            backbone, pretrained=True,
            pretrained_cfg_overlay=dict(file=MODEL_PATH[backbone]),
        )

        self.rotation = False if angle_bins is None else True

        if self.rotation:
            self.yaw = nn.Linear(self.backbone.fc.in_features, angle_bins)
            self.pitch = nn.Linear(self.backbone.fc.in_features, angle_bins)
            self.roll = nn.Linear(self.backbone.fc.in_features, angle_bins)

        self.trans = nn.Linear(self.backbone.fc.in_features, 2)
        self.scale = nn.Linear(self.backbone.fc.in_features, 1)

        self.motion = nn.Linear(self.backbone.fc.in_features, hidden_dim)

        self.backbone.fc = nn.Identity()

    def forward(self, x):
        feature = self.backbone(apply_imagenet_normalization(x))

        output_dict = {}

        output_dict['pose'] = {}

        if self.rotation:
            output_dict['pose']['yaw'] = bins2degree(self.yaw(feature))
            output_dict['pose']['pitch'] = bins2degree(self.pitch(feature))
            output_dict['pose']['roll'] = bins2degree(self.roll(feature))

        output_dict['pose']['trans'] = torch.tanh(self.trans(feature))
        output_dict['pose']['scale'] = torch.tanh(self.scale(feature))

        output_dict['motion'] = self.motion(feature)

        return output_dict


class MaskPredictor(nn.Module):
    def __init__(self, num_classes, in_features, num_blocks, max_features):
        super(MaskPredictor, self).__init__()

        self.hourglass = Hourglass3D(in_features, num_blocks, max_features)
        self.out = nn.Sequential(nn.Conv3d(in_features, num_classes, kernel_size=3, padding=1), nn.Softmax(dim=1))
    
    def forward(self, x):
        return self.out(self.hourglass(x))


class MotionGenerator(nn.Module):
    def __init__(self, block_expansion, num_blocks, max_features, hidden_dim, spatial_size):
        super(MotionGenerator, self).__init__()

        if isinstance(spatial_size, Iterable):
            assert(len(spatial_size) == 3)
            self.spatial_size = [s // (2 ** num_blocks) for s in spatial_size]
        else:
            self.spatial_size = [spatial_size // (2 ** num_blocks)] * 3

        out_features = min(max_features, block_expansion * (2 ** num_blocks))
        self.motion = nn.Parameter(torch.randn(1, out_features, *self.spatial_size))

        self.up = nn.Upsample(scale_factor=2)

        self.up_blocks = nn.Sequential()
        for i in range(num_blocks):
            in_features = min(max_features, block_expansion * (2 ** (num_blocks - i)))
            out_features = min(max_features, block_expansion * (2 ** (num_blocks - i - 1)))
            self.up_blocks.append(StyleResBlock3d(in_features, out_features, hidden_dim))

        self.final = nn.Conv3d(out_features, 3, kernel_size=1, stride=1)

    def forward(self, x):
        out = self.motion.repeat(x.shape[0], 1, 1, 1, 1)
        for up_block in self.up_blocks:
            out = up_block(self.up(out), x)
        out = F.leaky_relu(out, 0.2)
        out = torch.tanh(self.final(out))
        return out.permute(0, 2, 3, 4, 1)
