import os
import argparse
from omegaconf import OmegaConf

import torch
from torchvision.utils import make_grid

from driving.video2video.face.implicit.data import FramesDataset
from driving.video2video.face.implicit.generator import FaceGenerator
from driving.video2video.discriminator import MultiScaleDiscriminator

from driving.video2video.trainer import Trainer

from utils import postprocess, save_image


class FaceDrivingTrainer(Trainer):
    def load_dataset(self, dataset_params):
        return FramesDataset(**dict(dataset_params), sampler=self.sampler)

    def load_model(self, model_params):
        generator = FaceGenerator(**dict(model_params.generator_params))
        discriminator = MultiScaleDiscriminator(**dict(model_params.discriminator_params))
        return generator, discriminator

    def save_visualization(self, epoch, x, generated, n=5):
        out = []

        out.append(postprocess(x['src_img']))
        out.append(postprocess((x['src_msk'] > 0.5).repeat(1, 3, 1, 1)))

        out.append(postprocess(x['dri_img']))
        out.append(postprocess((x['dri_msk'] > 0.5).repeat(1, 3, 1, 1)))

        if 'reconstruction' in generated.keys():
            out.append(postprocess(generated['reconstruction'].detach().cpu()))
            out.append(postprocess((generated['source_mask'] > 0.5).repeat(1, 3, 1, 1).detach().cpu()))

        out.append(postprocess(generated['prediction'].detach().cpu()))
        out.append(postprocess((generated['driving_mask'] > 0.5).repeat(1, 3, 1, 1).detach().cpu()))

        if 'cid_prediction' in generated.keys():
            out.append(postprocess(x['cid_img']))
            out.append(postprocess(generated['cid_prediction'].detach().cpu()))

        out = torch.cat(out, dim=3)[:n]
        out = make_grid(out, nrow=1, padding=0, normalize=False)
        out = save_image(f'{self.log_dir}/samples/{epoch:0>4d}.jpg', out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train implicit face driving model.")
    parser.add_argument("--device_ids", type=int, nargs='+', default=[0], help='available devices (seperate by space)')
    parser.add_argument('--config', type=str, default='driving/video2video/face/implicit/config/default.yaml', help='config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint load path')
    parser.add_argument("--exp", type=str, default='test', help="experiment name")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.train_params.checkpoint = args.checkpoint

    os.makedirs(f'driving/video2video/face/implicit/logs/{args.exp}/samples', exist_ok=True)
    OmegaConf.save(config, f'driving/video2video/face/implicit/logs/{args.exp}/config.yaml')

    torch.inverse(torch.ones((1,1), device=f'cuda:{args.device_ids[0]}'))

    trainer = FaceDrivingTrainer(config, f'driving/video2video/face/implicit/logs/{args.exp}', args.device_ids)
    trainer.run()
