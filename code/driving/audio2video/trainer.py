import os
import copy

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

from driving.audio2video.model import PMTransformer
from driving.audio2video.diffusion import PMDiffusion

from utils import ExponentialMovingAverage
from utils import AUDIO_FEAT_DIM


class Trainer:
    def __init__(self, cfg, log_dir, device_ids):
        self.log_dir = log_dir
        self.device_ids = device_ids

        self.train_params = cfg.train_params

        cfg.dataset_params.seq_n = self.train_params.seq_n
        cfg.dataset_params.pre_k = self.train_params.pre_k

        cfg.model_params.seq_n = self.train_params.seq_n
        cfg.model_params.pre_k = self.train_params.pre_k
        cfg.model_params.cond_dim = AUDIO_FEAT_DIM[cfg.dataset_params.audio_feat_type]

        self.ema = ExponentialMovingAverage(0.99)

        self.data_loader = DataLoader(
            self.load_dataset(cfg.dataset_params),
            batch_size=self.train_params.batch_size,
            shuffle=True, drop_last=True, num_workers=8, pin_memory=True
        )

        self.model = PMTransformer(**dict(cfg.model_params))
        self.diffusion = nn.DataParallel(PMDiffusion(self.model, **dict(cfg.diffusion_params)))

        self.optimizer = AdamW(self.model.parameters(), lr=self.train_params.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, self.train_params.num_epochs // 2, eta_min=self.train_params.min_lr)

        self.diffusion.to(f'cuda:{device_ids[0]}')

        self.master_model = copy.deepcopy(self.model)

        self.start_epoch = 1

        if self.train_params.checkpoint:
            print(f'loading checkpoint from {self.train_params.checkpoint}...')
            self.load_checkpoint(self.train_params.checkpoint)

    def load_dataset(self, dataset_params):
        raise NotImplementedError

    def save_checkpoint(self, epoch):
        checkpoint = {
            'network': self.master_model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }
        torch.save(checkpoint, f'{self.log_dir}/{epoch:0>4d}-ckpt.pth.tar')

    def load_checkpoint(self, ckpt, model_only=False):
        self.start_epoch = int(os.path.basename(ckpt).split('-')[0])

        checkpoint = torch.load(ckpt)
        self.model.load_state_dict(checkpoint['network'])
        self.master_model.load_state_dict(checkpoint['network'])

        if not model_only:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])

    def log(self, epoch, losses):
        prefix = f'[Epoch {epoch:0>4d}/{self.train_params.num_epochs:0>4d}] '
        description = ', '.join([f'{k}: {v:.4f}' for k, v in losses.items()])

        print(prefix + description)

        with open(f'{self.log_dir}/results.txt', 'a') as f:
            f.write(prefix + description + '\n')

        if epoch % self.train_params.checkpoint_freq == 0:
            self.save_checkpoint(epoch)

    def run(self):
        print('Start Training...')

        for epoch in range(self.start_epoch, self.train_params.num_epochs + 1):

            for i_iter, (cond, motion, padding_mask) in enumerate(self.data_loader):
                losses = self.diffusion(motion, cond, padding_mask)
                loss = sum([v for v in losses.values()])

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            self.log(epoch, losses)

            self.ema.update_model(self.master_model, self.model)

            self.scheduler.step()
