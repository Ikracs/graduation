import argparse
from omegaconf import OmegaConf

import torch

from driving.video2video.driver import Driver
from driving.video2video.face.explicit.generator import FaceGenerator


class FaceDriver(Driver):
    def load_model(self, model_params, checkpoint):
        generator = FaceGenerator(**dict(model_params))
        generator.load_state_dict(torch.load(checkpoint, map_location='cpu')['generator'])
        return generator

    def motion_transfer(self, source_coeffs, driving_coeffs, paste_back):
        if paste_back:
            coeffs = self.generator.model.split_coeffs(source_coeffs, return_dict=True)
            coeffs.update({'exp_coeff', self.generator.model.get_exp_coeff(driving_coeffs)})
        else:
            coeffs = self.generator.model.split_coeffs(driving_coeffs, return_dict=True)
            coeffs.update({'id_coeff', self.generator.model.get_id_coeff(source_coeffs)})
        return self.generator.model(self.generator.model.merge_coeffs(coeffs))['vs_proj']

    def process(self, source, driving, paste_back):
        source_kps = self.generator.extract_kps(source['coeffs'])
        driving_kps = self.motion_transfer(source['coeffs'], driving['coeffs'], paste_back)
        return self.generator.driving(source['feature'], source_kps, driving_kps)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run video2video driving generator.")
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
    parser.add_argument('--config', type=str, default='driving/video2video/face/explicit/config/default.yaml')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint load path')
    parser.add_argument('--input_image', type=str, default=None, help='input image path')
    parser.add_argument('--input_video', type=str, default=None, help='input video path')
    parser.add_argument('--output', type=str, default=None, help='output video path')
    parser.add_argument('--paste_back', action='store_true', default=None, help='paste result back')

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.train_params.checkpoint = args.checkpoint

    driver = FaceDriver(config, args.device)
    driver.run(args.input_image, args.input_video, args.output, args.paste_back)
