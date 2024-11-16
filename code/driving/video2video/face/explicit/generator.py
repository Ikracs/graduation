import torch.nn as nn

from face_reconstruction.regression.utils import prepare_pipeline

from driving.video2video.face.explicit.network import DenseMotionNetwork

from driving.video2video.network import VolumeEncoder
from driving.video2video.network import SPADEDecoder
from driving.video2video.utils import apply_deformation


class FaceGenerator(nn.Module):
    def __init__(self, img_size, img_channel, block_expansion, max_features, num_down_blocks, num_resblocks, kp_params, motion_params):
        super(FaceGenerator, self).__init__()

        self.img_size = img_size
        self.img_channel = img_channel

        spatial_size = [img_size // (2 ** num_down_blocks)] * 2

        self.model, self.penet = prepare_pipeline(**kp_params, img_size=img_size)
        self.motion = DenseMotionNetwork(in_features=max_features, **motion_params, spatial_size=spatial_size)

        self.encoder = VolumeEncoder(img_channel, block_expansion, max_features, num_down_blocks, num_resblocks)
        self.decoder = SPADEDecoder(img_channel, block_expansion, max_features, num_down_blocks, num_resblocks)

    def extract_kps(self, coeffs):
        return self.model(coeffs, shape=True)['vs_proj'][:, :, :2] / (self.img_size - 1)

    def process_source(self, source_image):
        feature = self.encoder(source_image)
        coeffs = self.penet(source_image)
        return {'feature': feature, 'coeffs': coeffs}

    def process_driving(self, driving_image):
        return {'coeffs': self.penet(driving_image)}

    def driving(self, feature, source_kps, driving_kps):
        deformation = self.motion(feature, source_kps, driving_kps)
        deformed_feature = apply_deformation(feature, deformation)
        prediction = self.decoder(deformed_feature)
        return {'deformation': deformation, 'prediction': prediction}

    def forward(self, source_image, driving_image):
        source = self.process_source(source_image)
        driving = self.process_driving(driving_image)

        source['kps'] = self.extract_kps(source['coeffs'])
        driving['kps'] = self.extract_kps(driving['coeffs'])

        output_dict = self.driving(source['feature'], source['kps'], driving['kps'])
        output_dict.update({'source_coeffs': source['coeffs'], 'driving_coeffs': driving['coeffs']})

        return output_dict
