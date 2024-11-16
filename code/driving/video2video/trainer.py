import os
import copy

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from driving.video2video.model import GeneratorFullModel, DiscriminatorFullModel
from driving.video2video.sync_batchnorm import DataParallelWithCallback
from driving.video2video.utils import DataSampler

from utils import ExponentialMovingAverage


class Trainer:
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

        generator, discriminator = self.load_model(cfg.model_params)

        generator_full = GeneratorFullModel(generator, discriminator, self.train_params)
        discriminator_full = DiscriminatorFullModel(generator, discriminator, self.train_params)

        self.gen_optimizer = AdamW(generator.parameters(), lr=self.train_params.generator_lr, betas=(0.5, 0.999))
        self.dis_optimizer = AdamW(discriminator.parameters(), lr=self.train_params.discriminator_lr, betas=(0.5, 0.999))

        self.gen_scheduler = CosineAnnealingLR(self.gen_optimizer, self.train_params.num_epochs // 2, eta_min=self.train_params.min_lr)
        self.dis_scheduler = CosineAnnealingLR(self.dis_optimizer, self.train_params.num_epochs // 2, eta_min=self.train_params.min_lr)

        self.generator = generator
        self.discriminator = discriminator
        self.generator_full = DataParallelWithCallback(generator_full, self.device_ids)
        self.discriminator_full = DataParallelWithCallback(discriminator_full, self.device_ids)

        self.generator_full.to(f'cuda:{device_ids[0]}')
        self.discriminator_full.to(f'cuda:{device_ids[0]}')

        self.master_model = copy.deepcopy(self.generator)

        self.start_epoch = 1

        if self.train_params.checkpoint:
            print(f'loading checkpoint from {self.train_params.checkpoint}...')
            self.load_checkpoint(self.train_params.checkpoint)

    def load_dataset(self, dataset_params):
        raise NotImplementedError

    def load_model(self, model_params):
        raise NotImplementedError

    def save_checkpoint(self, epoch):
        checkpoint = {
            'generator': self.master_model.state_dict(),
            'discriminator': self.discriminator.state_dict(),
            'gen_optimizer': self.gen_optimizer.state_dict(),
            'dis_optimizer': self.dis_optimizer.state_dict(),
            'gen_scheduler': self.gen_scheduler.state_dict(),
            'dis_scheduler': self.dis_scheduler.state_dict()
        }
        torch.save(checkpoint, f'{self.log_dir}/{epoch:0>4d}-ckpt.pth.tar')

    def load_checkpoint(self, ckpt, model_only=False):
        self.start_epoch = int(os.path.basename(ckpt).split('-')[0])

        checkpoint = torch.load(ckpt, map_location='cpu')
        self.generator.load_state_dict(checkpoint['generator'])
        self.master_model.load_state_dict(checkpoint['generator'])
        self.discriminator.load_state_dict(checkpoint['discriminator'])

        if not model_only:
            self.gen_optimizer.load_state_dict(checkpoint['gen_optimizer'])
            self.dis_optimizer.load_state_dict(checkpoint['dis_optimizer'])
            self.gen_scheduler.load_state_dict(checkpoint['gen_scheduler'])
            self.dis_scheduler.load_state_dict(checkpoint['dis_scheduler'])

    def log(self, epoch, losses):
        prefix = f'[Epoch {epoch:0>4d}/{self.train_params.num_epochs:0>4d}] '
        description = ', '.join([f'{k}: {v:.4f}' for k, v in losses.items()])

        print(prefix + description)

        with open(f'{self.log_dir}/results.txt', 'a') as f:
            f.write(prefix + description + '\n')

        if epoch % self.train_params.checkpoint_freq == 0:
            self.save_checkpoint(epoch)

    def save_visualization(self, epoch, x, generated, n=5):
        raise NotImplementedError

    def run(self):
        print('Start Training...')

        for epoch in range(self.start_epoch, self.train_params.num_epochs + 1):

            for i_iter, x in enumerate(self.data_loader):
                gen_losses, generated = self.generator_full(x)
                loss = sum([v.mean() for v in gen_losses.values()])

                self.gen_optimizer.zero_grad()
                loss.backward()
                self.gen_optimizer.step()

                if self.train_params.loss_weights.generator != 0:
                    dis_losses = self.discriminator_full(x, generated)
                    loss = sum([v.mean() for v in dis_losses.values()])

                    self.dis_optimizer.zero_grad()
                    loss.backward()
                    self.dis_optimizer.step()
                else:
                    dis_losses = {}

                losses = {k: v.mean().item() for k, v in {**gen_losses, **dis_losses}.items()}

            self.log(epoch, losses)
            self.save_visualization(epoch, x, generated)

            self.sampler.step()

            self.gen_scheduler.step()
            self.dis_scheduler.step()

            self.ema.update_model(self.master_model, self.generator)
