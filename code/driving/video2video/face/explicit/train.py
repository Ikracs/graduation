import os
import copy
import argparse
from omegaconf import OmegaConf

import torch
from torchvision.utils import make_grid

from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from face_reconstruction.regression.utils import prepare_pipeline

from driving.video2video.trainer import Trainer
from driving.video2video.face.explicit.data import FramesDataset
from driving.video2video.face.explicit.generator import FaceGenerator
from driving.video2video.discriminator import MultiScaleDiscriminator
from driving.video2video.face.explicit.model import GeneratorFullModel
from driving.video2video.model import DiscriminatorFullModel
from driving.video2video.sync_batchnorm import DataParallelWithCallback
from driving.video2video.utils import DataSampler

from utils import postprocess, save_image
from utils import ExponentialMovingAverage


class FaceDrivingTrainer(Trainer):
    def __init__(self, cfg, log_dir, device_ids):
        self.log_dir = log_dir
        self.device_ids = device_ids

        self.train_params = cfg.train_params

        self.sampler = DataSampler(num_epochs=self.train_params.num_epochs)
        self.ema = ExponentialMovingAverage(beta=0.99)

        self.data_loader = DataLoader(
            self.load_dataset(cfg.dataset_parmas),
            batch_size=self.train_params.batch_size,
            shuffle=True, drop_last=True, num_workers=8, pin_memory=True
        )

        model, penet = prepare_pipeline(
            **dict(self.model_params.generator_params.kp_params),
            img_size=self.dataset_params.img_size, checkpoint='assets/0200-penet.pth.tar'
        )

        generator, discriminator = self.load_model(cfg.model_params)

        generator_full = GeneratorFullModel(penet, generator, discriminator, self.train_params)
        discriminator_full = DiscriminatorFullModel(generator, discriminator, self.train_params)

        self.gen_optimizer = AdamW(generator.parameters(), lr=self.train_params.generator_lr, betas=(0.5, 0.999))
        self.dis_optimizer = AdamW(discriminator.parameters(), lr=self.train_params.discriminator_lr, betas=(0.5, 0.999))

        self.gen_scheduler = CosineAnnealingLR(self.gen_optimizer, self.train_params.num_epochs // 2, eta_min=self.train_params.min_lr)
        self.dis_scheduler = CosineAnnealingLR(self.dis_optimizer, self.train_params.num_epochs // 2, eta_min=self.train_params.min_lr)

        self.model = model
        self.generator = generator
        self.discriminator = discriminator
        self.generator_full = DataParallelWithCallback(generator_full, self.device_ids)
        self.discriminator_full = DataParallelWithCallback(discriminator_full, self.device_ids)

        self.model.to(f'cuda:{device_ids[0]}')
        self.generator_full.to(f'cuda:{device_ids[0]}')
        self.discriminator_full.to(f'cuda:{device_ids[0]}')

        self.master_model = copy.deepcopy(self.generator)

        self.start_epoch = 1

        if self.train_params.checkpoint:
            print(f'loading checkpoint from {self.train_params.checkpoint}...')
            self.load_checkpoint(self.train_params.checkpoint)

    def load_dataset(self, dataset_params):
        return FramesDataset(**dict(dataset_params))

    def load_model(self, model_params):
        generator = FaceGenerator(**dict(model_params.generator_params))
        discriminator = MultiScaleDiscriminator(**dict(model_params.discriminator_params))
        return generator, discriminator

    def save_visualization(self, epoch, x, generated, n=5):
        out = []

        out.append(postprocess(x['src_img']))
        out.append(postprocess(self.model(generated['source_coeffs'], render=True)['render_img'].detach().cpu()))

        out.append(postprocess(x['dri_img']))
        out.append(postprocess(self.model(generated['driving_coeffs'], render=True)['render_img'].detach().cpu()))

        if 'reconstruction' in generated.keys():
            out.append(postprocess(generated['reconstruction'].detach().cpu()))

        out.append(postprocess(generated['prediction'].detach().cpu()))

        out = torch.cat(out, dim=3)[:n]
        out = make_grid(out, nrow=1, padding=0, normalize=False)
        out = save_image(f'{self.log_dir}/samples/{epoch:0>4d}.jpg', out)

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train explicit face driving model.")
    parser.add_argument("--device_ids", type=int, nargs='+', default=[0], help='available devices (seperate by space)')
    parser.add_argument('--config', type=str, default='driving/video2video/face/explicit/config/default.yaml', help='config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint load path')
    parser.add_argument("--exp", type=str, default='test', help="experiment name")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.train_params.checkpoint = args.checkpoint

    os.makedirs(f'driving/video2video/face/explicit/logs/{args.exp}/samples', exist_ok=True)
    OmegaConf.save(config, f'driving/video2video/face/explicit/logs/{args.exp}/config.yaml')

    torch.inverse(torch.ones((1,1), device=f'cuda:{args.device_ids[0]}'))

    trainer = FaceDrivingTrainer(config, f'driving/video2video/face/explicit/logs/{args.exp}', args.device_ids)
    trainer.run()
