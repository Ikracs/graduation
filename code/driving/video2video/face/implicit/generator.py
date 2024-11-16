import torch

from driving.video2video.utils import degrees2rotation
from driving.video2video.utils import make_coordinate_grid_like_3d
from driving.video2video.utils import stitch_deformation, apply_deformation

from driving.video2video.generator import SPADEGenerator


class FaceGenerator(SPADEGenerator):
    @staticmethod
    def make_deformation(mask, source_motion, driving_motion, source_pose, driving_pose=None):
        identity_grid = make_coordinate_grid_like_3d(mask)
        background = stitch_deformation(10, mask[:, 0: 1], identity_grid)

        source_scale = source_pose['scale'] + 1.0
        source_rotation = degrees2rotation(yaw=source_pose['yaw'], pitch=source_pose['pitch'], roll=source_pose['roll'])
        source_trans = torch.cat((source_pose['trans'], torch.zeros_like(source_pose['trans'][:, 1:])), dim=1)

        driving_pose = source_pose if driving_pose is None else driving_pose

        driving_scale = driving_pose['scale'] + 1.0
        driving_rotation = degrees2rotation(yaw=driving_pose['yaw'], pitch=driving_pose['pitch'], roll=driving_pose['roll'])
        driving_trans = torch.cat((driving_pose['trans'], torch.zeros_like(driving_pose['trans'][:, 1:])), dim=1)

        mask = mask[:, 0: 1]

        deformation_ = identity_grid - driving_trans[:, None, None, None]
        deformation_ = torch.matmul(torch.inverse(driving_rotation)[:, None, None, None], deformation_[..., None]).squeeze(-1)
        deformation_ /= driving_scale[:, None, None, None]

        deformation_ = deformation_ + source_motion - driving_motion

        deformation_ *= source_scale[:, None, None, None]
        deformation_ = torch.matmul(source_rotation[:, None, None, None], deformation_[..., None]).squeeze(-1)
        deformation_ = deformation_ + source_trans[:, None, None, None]

        deformed_mask = apply_deformation(mask, deformation_)
        deformation_ = stitch_deformation(deformation_, deformed_mask, background)

        return deformation_, deformed_mask
