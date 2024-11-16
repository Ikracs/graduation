from collections import Iterable

import torch
import torch.nn as nn

from driving.video2video.modules import Hourglass2D

from driving.video2video.utils import make_coordinate_grid_2d
from driving.video2video.utils import resize_deformation_2d
from driving.video2video.utils import apply_deformation


class DenseMotionNetwork(nn.Module):
    def __init__(self, in_features, num_blocks, max_features, spatial_size):
        super(DenseMotionNetwork, self).__init__()

        if isinstance(spatial_size, Iterable):
            assert(len(spatial_size) == 2)
            self.spatial_size = spatial_size
        else:
            self.spatial_size = [spatial_size] * 2

        self.identity_grid = make_coordinate_grid_2d(self.spatial_size, dtype=torch.float32)

        self.hourglass = Hourglass2D(in_features, num_blocks, max_features)
        self.out = nn.Sequential(nn.Conv2d(in_features, 2, kernel_size=3, padding=1), nn.Tanh())

    def create_sparse_motion(self, feature, source_kps, driving_kps):
        B, K, H, W = *source_kps.shape[:-1], *self.spatial_size

        src_coords = source_kps.flatten(0, 1)
        dri_coords = driving_kps.flatten(0, 1)

        b = torch.arange(B, device=feature.device)
        b = b.unsqueeze(1).repeat(1, K).flatten()

        src_coords_h = (src_coords[:, 1].clamp(0, 1) * (H - 1)).round().type(torch.int64)
        src_coords_w = (src_coords[:, 0].clamp(0, 1) * (W - 1)).round().type(torch.int64)

        dri_coords_h = (dri_coords[:, 1].clamp(0, 1) * (H - 1)).round().type(torch.int64)
        dri_coords_w = (dri_coords[:, 0].clamp(0, 1) * (W - 1)).round().type(torch.int64)

        sparse_motion = self.identity_grid.to(device=feature.device, dtype=feature.dtype)
        sparse_motion = sparse_motion.unsqueeze(0).repeat(feature.shape[0], 1, 1, 1)

        sparse_motion[b, src_coords_h, src_coords_w] = torch.inf
        sparse_motion[b, dri_coords_h, dri_coords_w] = src_coords * 2 - 1

        return resize_deformation_2d(sparse_motion, feature.shape[2:])

    def forward(self, feature, source_kps, driving_kps):
        sparse_motion = self.create_sparse_motion(feature, source_kps, driving_kps)
        prediction = self.hourglass(apply_deformation(feature, sparse_motion))
        return self.out(prediction).permute(0, 2, 3, 1)
