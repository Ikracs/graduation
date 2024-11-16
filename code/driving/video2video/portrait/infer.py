import argparse

import torch
from omegaconf import OmegaConf

from driving.video2video.portrait.generator import PortraitGenerator

from driving.video2video.driver import Driver


class PortraitDriver(Driver):
    def load_model(self, model_params, checkpoint):
        generator = PortraitGenerator(**dict(model_params))
        generator.load_state_dict(torch.load(checkpoint, map_location='cpu')['generator'])
        return generator

    def process(self, source, driving, paste_back):
        return self.generator.driving(source, driving['motion'], None if paste_back else driving['pose'])

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Run video2video driving generator.")
    parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
    parser.add_argument('--config', type=str, default='driving/video2video/portrait/config/default.yaml')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint load path')
    parser.add_argument('--input_image', type=str, default=None, help='input image path')
    parser.add_argument('--input_video', type=str, default=None, help='input video path')
    parser.add_argument('--output', type=str, default=None, help='output video path')
    parser.add_argument('--paste_back', action='store_true', default=None, help='paste result back')

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.train_params.checkpoint = args.checkpoint

    driver = PortraitDriver(config, args.device)
    driver.run(args.input_image, args.input_video, args.output, args.paste_back)
