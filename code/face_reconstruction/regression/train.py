import os
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from face_reconstruction.regression.data import FramesDataset
from face_reconstruction.losses import get_lmk_weights_68, get_lmk_weights_478
from face_reconstruction.losses import landmark_loss, photo_loss, get_reg
from face_reconstruction.losses import identity_loss, id_consistent_loss
from face_reconstruction.losses import closure_loss, gaze_loss
from face_reconstruction.losses import EYE_CLOSURE_PAIRS, MP_EYE_CLOSURE_PAIRS
from face_reconstruction.losses import LIP_CLOSURE_PAIRS, MP_LIP_CLOSURE_PAIRS

from face_reconstruction.regression.utils import prepare_pipeline

from pretrained.face_parsing import BiSeNet
from pretrained.face_recognition import FaceNet

from utils import requires_grad
from utils import overlay_landmarks
from utils import postprocess, save_image

from utils import MP_LEYE, MP_REYE
from utils import CONNECTIVITY, MP_CONNECTIVITY


class Trainer:
    def __init__(self, cfg, log_dir, device):
        self.log_dir = log_dir
        self.device = device

        self.dataset_params = cfg.dataset_params
        self.train_params = cfg.train_params

        self.model, self.network = prepare_pipeline(
            model=cfg.model,
            expand_ratio=self.dataset_params.expand_ratio,
            img_size=self.dataset_params.img_size
        )

        self.model.to(device)
        self.network.to(device)

        self.data_loader = DataLoader(
            FramesDataset(lmk_type=self.model.lmk_type, **dict(self.dataset_params)),
            batch_size=self.train_params.batch_size // self.dataset_params.num_frames,
            shuffle=True, drop_last=True, num_workers=8, pin_memory=True
        )

        if self.train_params.loss_weights.photo != 0:
            self.bisenet = BiSeNet().to(device)
            requires_grad(self.bisenet, False)
            self.bisenet.eval()

        if self.train_params.loss_weights.identity != 0:
            self.facenet = FaceNet().to(device)
            requires_grad(self.facenet, False)
            self.facenet.eval()

        self.start_epoch = 1

        self.optimizer = AdamW(self.network.parameters(), self.train_params.lr)
        self.scheduler = CosineAnnealingLR(self.optimizer, self.train_params.num_epochs // 2, eta_min=self.train_params.min_lr)

        if self.model.lmk_type == '478':
            self.lmk_weights = get_lmk_weights_478().to(device).unsqueeze(0)
            if self.train_params.loss_weights.eye_closure != 0:
                self.eye_closure_pairs = torch.tensor(MP_EYE_CLOSURE_PAIRS, dtype=torch.int64, device=device)
            if self.train_params.loss_weights.lip_closure != 0:
                self.lip_closure_pairs = torch.tensor(MP_LIP_CLOSURE_PAIRS, dtype=torch.int64, device=device)
            if self.train_params.loss_weights.gaze != 0:
                self.l_eye = torch.tensor(MP_LEYE, dtype=torch.int64, device=device)
                self.r_eye = torch.tensor(MP_REYE, dtype=torch.int64, device=device)
                self.l_iris = torch.tensor([473], dtype=torch.int64, device=device)
                self.r_iris = torch.tensor([468], dtype=torch.int64, device=device)
            self.lmk_connectivity = MP_CONNECTIVITY
        elif self.model.lmk_type == '68':
            self.lmk_weights = get_lmk_weights_68().to(device).unsqueeze(0)
            if self.train_params.loss_weights.eye_closure != 0:
                self.eye_closure_pairs = torch.tensor(EYE_CLOSURE_PAIRS, dtype=torch.int64, device=device)
            if self.train_params.loss_weights.lip_closure != 0:
                self.lip_closure_pairs = torch.tensor(LIP_CLOSURE_PAIRS, dtype=torch.int64, device=device)
            self.lmk_connectivity = CONNECTIVITY
        else:
            raise NotImplementedError(f'Unknown landmark type {self.model.lmk_type}')

    def save_checkpoint(self, epoch):
        checkpoint = {
            'network': self.network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()
        }
        torch.save(checkpoint, f'{self.log_dir}/{epoch:0>4d}-penet.pth.tar')

    def load_checkpoint(self, ckpt):
        self.start_epoch = int(ckpt.split('-')[0])
        checkpoint = torch.load(ckpt, map_location=self.device)
        self.network.load_state_dict(checkpoint['network'])
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])

    def get_training_pbar(self):
        beg, end = self.start_epoch, self.train_params.num_epochs
        return tqdm(range(beg, end + 1), initial=beg, total=end, dynamic_ncols=True, disable=False)

    def log(self, epoch, losses):
        prefix = f'[Epoch {epoch:0>4d}/{self.train_params.num_epochs:0>4d}] '
        description = ', '.join([f'{k}: {v:.4f}' for k, v in losses.items()])

        with open(f'{self.log_dir}/results.txt', 'a') as f:
            f.write(prefix + description + '\n')

    def save_visualization(self, epoch, x, lms, pred_lms, render, n=5):
        img = postprocess(x.cpu())
        render_img = postprocess(render.detach().cpu())

        img_with_lms = overlay_landmarks(img, lms.cpu(), self.lmk_connectivity)
        img_with_pred_lms = overlay_landmarks(img, pred_lms.detach().cpu(), self.lmk_connectivity)

        out = [img, img_with_lms, img_with_pred_lms, render_img]
        out = torch.cat(out, dim=3)[torch.randperm(x.shape[0])[:n]]
        out = make_grid(out, nrow=1, padding=0, normalize=False)
        save_image(f'{self.log_dir}/samples/{epoch:0>4d}.png', out)

    def run(self):
        if self.train_params.checkpoint:
            print(f'loading checkpoint from {self.train_params.checkpoint}...')
            self.load_checkpoint(self.train_params.checkpoint)

        self.network.train()

        self.pbar = self.get_training_pbar()
 
        for epoch in self.pbar:
            for i_iter, (x, lms) in enumerate(self.data_loader):
                x = x.flatten(0, 1).to(self.device)
                lms = lms.flatten(0, 1).to(self.device)

                coeffs = self.network(x)

                pred = self.model(coeffs, render=True, shape=True)

                pred_img = pred['render_img'][:, :3]
                pred_mask = pred['render_img'][:, 3].detach() > 0

                losses = {}
                if self.train_params.loss_weights.landmark != 0:
                    value = landmark_loss(pred['lms_proj'], lms, self.lmk_weights)
                    losses['landmark'] = self.train_params.loss_weights.landmark * value

                if self.train_params.loss_weights.photo != 0:
                    mask = self.bisenet.get_face_mask(x).squeeze(1)
                    value = photo_loss(pred_img, x, pred_mask, mask)
                    losses['photo'] = self.train_params.loss_weights.photo * value

                if self.train_params.loss_weights.identity != 0:
                    overlay = pred_img * pred_mask + x * (1 - pred_mask)
                    value = identity_loss(self.facenet(overlay), self.facenet(x))
                    losses['identity'] = self.train_params.loss_weights.identity * value

                if self.train_params.loss_weights.id_regularization != 0:
                    value = get_reg(self.model.get_id_coeff(coeffs))
                    losses['id_regularization'] = self.train_params.loss_weights.id_regularization * value

                if self.train_params.loss_weights.exp_regularization != 0:
                    value = get_reg(self.model.get_exp_coeff(coeffs))
                    losses['exp_regularization'] = self.train_params.loss_weights.exp_regularization * value

                if self.train_params.loss_weights.tex_regularization != 0:
                    value = get_reg(self.model.get_tex_coeff(coeffs))
                    losses['tex_regularization'] = self.train_params.loss_weights.tex_regularization * value

                if self.train_params.loss_weights.id_consistency != 0:
                    value = id_consistent_loss(self.model.get_id_coeff(coeffs).unflatten(0, (-1, self.dataset_params.num_frames)))
                    losses['id_consistency'] = self.train_params.loss_weights.id_consistency * value

                if self.train_params.loss_weights.eye_closure != 0:
                    value = closure_loss(pred['lms_proj'], lms, self.eye_closure_pairs)
                    losses['eye_closure'] = self.train_params.loss_weights.eye_closure * value

                if self.train_params.loss_weights.lip_closure != 0:
                    value = closure_loss(pred['lms_proj'], lms, self.lip_closure_pairs)
                    losses['lip_closure'] = self.train_params.loss_weights.lip_closure * value

                if self.train_params.loss_weights.gaze != 0:
                    value = gaze_loss(pred['lms_proj'], lms, self.l_eye, self.l_iris)
                    value += gaze_loss(pred['lms_proj'], lms, self.r_eye, self.r_iris)
                    losses['gaze'] = self.train_params.loss_weights.gaze * value

                self.optimizer.zero_grad()
                total_loss = sum(losses.values())
                total_loss.backward()
                self.optimizer.step()

            self.scheduler.step()
            self.log(epoch, losses)

            self.save_visualization(epoch, x, lms, pred['lms_proj'], pred_img)

            if epoch % self.train_params.save_freq == 0:
                self.save_checkpoint(epoch)

if __name__ == '__main__':
    bisenet = argparse.ArgumentParser("Train Parameter Estimation Network.")
    bisenet.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
    bisenet.add_argument('--config', type=str, default='face_reconstruction/regression/config/default.yaml')
    bisenet.add_argument('--checkpoint', type=str, default=None, help='checkpoint load path')
    bisenet.add_argument("--exp", type=str, default='test', help="experiment name")

    args = bisenet.parse_args()

    config = OmegaConf.load(args.config)
    config.train_params.checkpoint = args.checkpoint

    os.makedirs(f'face_reconstruction/regression/logs/{args.exp}/samples', exist_ok=True)
    OmegaConf.save(config, f'face_reconstruction/regression/logs/{args.exp}/config.yaml')

    trainer = Trainer(config, f'face_reconstruction/regression/logs/{args.exp}', args.device)
    trainer.run()
