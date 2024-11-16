import torch

from driving.video2video.utils import degrees2rotation
from driving.video2video.utils import make_coordinate_grid_like_3d
from driving.video2video.utils import stitch_deformation, apply_deformation

from driving.video2video.generator import SPADEGenerator


class PortraitGenerator(SPADEGenerator):
    @staticmethod
    def make_deformation(mask, source_motion, driving_motion, source_pose, driving_pose=None):
        merge = lambda x, y: torch.maximum(x, y)
        apply = lambda m, d: apply_deformation(torch.cat(m, dim=1), d).split(1, dim=1)

        make_rd = lambda d, r: torch.matmul(r[:, None, None, None], d[..., None]).squeeze(-1)
        make_td = lambda d, t: d + torch.cat((t, torch.zeros_like(t[:, 1:])), dim=1)[:, None, None, None]
        make_sd = lambda d, s: d * s[:, None, None, None]
        make_md = lambda d, m: d + m

        identity_grid = make_coordinate_grid_like_3d(mask)
        background = stitch_deformation(10, merge(mask[:, 0: 1], mask[:, 1: 2]), identity_grid)

        source_scale = source_pose['scale'] + 1.0
        source_rotation = degrees2rotation(yaw=source_pose['yaw'], pitch=source_pose['pitch'], roll=source_pose['roll'])
        source_trans = source_pose['trans']

        driving_pose = source_pose if driving_pose is None else driving_pose

        driving_scale_rec = 1 / (driving_pose['scale'] + 1.0)
        driving_rotation_inv = torch.inverse(degrees2rotation(yaw=driving_pose['yaw'], pitch=driving_pose['pitch'], roll=driving_pose['roll']))
        driving_trans_neg = -driving_pose['trans']

        head_mask, body_mask, _ = mask.split(1, dim=1)

        head_mask, body_mask = apply((head_mask, body_mask), make_td(identity_grid, driving_trans_neg))
        deformation_ = stitch_deformation(make_td(identity_grid, driving_trans_neg), merge(head_mask, body_mask), background)

        head_mask = apply_deformation(head_mask, make_rd(identity_grid, driving_rotation_inv))
        deformation_ = stitch_deformation(make_rd(deformation_, driving_rotation_inv), head_mask, deformation_)

        head_mask, body_mask = apply((head_mask, body_mask), make_sd(identity_grid, driving_scale_rec))
        deformation_ = stitch_deformation(make_sd(deformation_, driving_scale_rec), merge(head_mask, body_mask), deformation_)

        head_mask = apply_deformation(head_mask, make_md(identity_grid, source_motion - driving_motion))
        deformation_ = stitch_deformation(make_md(deformation_, source_motion - driving_motion), head_mask, deformation_)

        head_mask, body_mask = apply((head_mask, body_mask), make_sd(identity_grid, source_scale))
        deformation_ = stitch_deformation(make_sd(deformation_, source_scale), merge(head_mask, body_mask), deformation_)

        head_mask = apply_deformation(head_mask, make_rd(identity_grid, source_rotation))
        deformation_ = stitch_deformation(make_rd(deformation_, source_rotation), head_mask, deformation_)

        head_mask, body_mask = apply((head_mask, body_mask), make_td(identity_grid, source_trans))
        deformation_ = stitch_deformation(make_td(deformation_, source_trans), merge(head_mask, body_mask), deformation_)

        deformed_mask = torch.cat((head_mask, body_mask), dim=1)

        return deformation_, deformed_mask
