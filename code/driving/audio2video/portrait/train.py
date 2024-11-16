import os
import argparse
from omegaconf import OmegaConf

from driving.audio2video.portrait.data import A2MDataset

from driving.audio2video.trainer import Trainer


class PortraitTrainer(Trainer):
    def load_dataset(self, dataset_params):
        return A2MDataset(**dict(dataset_params))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Train audio2motion driving model.")
    parser.add_argument("--device_ids", type=int, nargs='+', default=[0], help='available devices (seperate by space)')
    parser.add_argument('--config', type=str, default='driving/audio2video/portrait/config/default.yaml', help='config file')
    parser.add_argument('--checkpoint', type=str, default=None, help='checkpoint load path')
    parser.add_argument("--exp", type=str, default='test', help="experiment name")

    args = parser.parse_args()

    config = OmegaConf.load(args.config)
    config.train_params.checkpoint = args.checkpoint

    os.makedirs(f'driving/audio2video/portrait/logs/{args.exp}', exist_ok=True)
    OmegaConf.save(config, f'driving/audio2video/portrait/logs/{args.exp}/config.yaml')

    trainer = PortraitTrainer(config, f'driving/audio2video/portrait/logs/{args.exp}', args.device_ids)
    trainer.run()
