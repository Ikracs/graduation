import torch.nn as nn

from driving.video2video.network import VolumeEncoder, SPADEDecoder
from driving.video2video.network import IdentityExtractor, PMEstimator
from driving.video2video.network import MaskPredictor, MotionGenerator

from driving.video2video.utils import project_mask
from driving.video2video.utils import apply_deformation


class SPADEGenerator(nn.Module):
    def __init__(self, img_size, img_channel, block_expansion, max_features, num_down_blocks, num_resblocks,
                 reshape_channel, hidden_dim, mask_params, motion_params):
        super(SPADEGenerator, self).__init__()

        self.img_size = img_size
        self.img_channel = img_channel

        reshape_depth = max_features // reshape_channel
        spatial_size = [reshape_depth] + [img_size // (2 ** num_down_blocks)] * 2

        self.mask = MaskPredictor(in_features=reshape_channel, **mask_params)
        self.motion = MotionGenerator(**motion_params, hidden_dim=hidden_dim, spatial_size=spatial_size)

        self.id_extractor = IdentityExtractor(hidden_dim)
        self.pm_estimator = PMEstimator(hidden_dim, angle_bins=66)

        self.encoder = VolumeEncoder(img_channel, block_expansion, max_features, num_down_blocks, reshape_channel, num_resblocks)
        self.decoder = SPADEDecoder(img_channel, block_expansion, max_features, num_down_blocks, num_resblocks)

    def process_source(self, image):
        feature = self.encoder(image)
        mask = self.mask(feature)

        src_id = self.id_extractor(image)
        src_pm = self.pm_estimator(image)

        output_dict = {'feature': feature, 'mask': mask, 'id': src_id}
        output_dict.update(src_pm)

        return output_dict

    def process_driving(self, image):
        return self.pm_estimator(image)

    @staticmethod
    def make_deformation(mask, source_motion, driving_motion, source_pose, driving_pose, **kwargs):
        raise NotImplementedError

    def driving(self, source, driving_motion, driving_pose=None):
        deformation, deformed_mask = self.make_deformation(
            source['mask'],
            self.motion(source['id'] + source['motion']),
            self.motion(source['id'] + driving_motion),
            source['pose'], driving_pose
        )

        src_mask_2d = project_mask(source['mask'], self.img_size)
        dri_mask_2d = project_mask(deformed_mask, self.img_size)

        deformed_feature = apply_deformation(source['feature'], deformation)
        prediction = self.decoder(deformed_feature)

        output_dict = {}

        output_dict['feature'] = source['feature']
        output_dict['source_id'] = source['id']
        output_dict['source_pose'] = source['pose']
        output_dict['source_motion'] = source['motion']
        output_dict['driving_pose'] = driving_pose
        output_dict['driving_motion'] = driving_motion

        output_dict['source_mask'] = src_mask_2d
        output_dict['driving_mask'] = dri_mask_2d

        output_dict['deformation'] = deformation
        output_dict['prediction'] = prediction

        return output_dict

    def forward(self, source_image, driving_image):
        source = self.process_source(source_image)
        driving = self.process_driving(driving_image)
        return self.driving(source, driving['motion'], driving['pose'])
