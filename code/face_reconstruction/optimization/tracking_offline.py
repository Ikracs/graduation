import os
import argparse
from tqdm import tqdm
from datetime import datetime
from omegaconf import OmegaConf
from threading import Thread

import torch
import numpy as np
import torchvision.transforms.functional as F
from torch.optim import Adam

from face_3dmm import load_face_model

from face_reconstruction.optimization.data import VideoReader, FrameFolder
from face_reconstruction.losses import get_lmk_weights_68, get_lmk_weights_478
from face_reconstruction.losses import landmark_loss, photo_loss, get_reg
from face_reconstruction.losses import closure_loss, closure_loss, gaze_loss
from face_reconstruction.losses import EYE_CLOSURE_PAIRS, MP_EYE_CLOSURE_PAIRS
from face_reconstruction.losses import LIP_CLOSURE_PAIRS, MP_LIP_CLOSURE_PAIRS

from utils import FaceAlignment
from utils import NoFaceDetectedException, NoFrameReturnedException
from utils import postprocess, save_image

from utils import VIDEO_EXT
from utils import MP_LEYE, MP_REYE


class Tracker:
    def __init__(self, video, out_dir, cfg, device):
        self.timestamp = str(datetime.now())[:-7].replace(' ', '-')

        self.out_dir = out_dir
        self.cfg = cfg
        self.device = device

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
            self.detector = FaceAlignment(detector='mediapipe')
        elif self.model.lmk_type == '68':
            self.lmk_weights = get_lmk_weights_68().to(device).unsqueeze(0)
            if self.train_params.loss_weights.eye_closure != 0:
                self.eye_closure_pairs = torch.tensor(EYE_CLOSURE_PAIRS, dtype=torch.int64, device=device)
            if self.train_params.loss_weights.lip_closure != 0:
                self.lip_closure_pairs = torch.tensor(LIP_CLOSURE_PAIRS, dtype=torch.int64, device=device)
            self.detector = FaceAlignment(detector='face_alignment')
        else:
            raise NotImplementedError(f'Unknown landmark type {self.model.lmk_type}')

        if os.path.isdir(video):
            self.reader = FrameFolder(video, cfg.img_size, cfg.expand_ratio)
        elif os.path.splitext(video)[-1] in VIDEO_EXT:
            self.reader = VideoReader(video, cfg.img_size, cfg.expand_ratio)
        else:
            raise NotImplementedError(f'Unsupported input {video}!')

        self.model = load_face_model(model=cfg.model, img_size=cfg.img_size).to(device)

    def build_optimizers(self, coeffs, first_frame=False):
        return Adam(coeffs, lr=1e-2, betas=(0.8, 0.95)) if first_frame else Adam(coeffs, lr=1e-3, betas=(0.5, 0.9))

    def init_coeffs(self, batch_size):
        coeffs = {k: v.requires_grad_(True) for k, v in self.model.get_default_coeffs(batch_size, self.device, return_dict=True)}
        coeffs['id_coeff'] = getattr(self, 'id_coeff', self.model.get_default_id_coeff(1, self.device).requires_grad_(True))
        coeffs['tex_coeff'] = getattr(self, 'tex_coeff', self.model.get_default_tex_coeff(1, self.device).requires_grad_(True))
        return coeffs

    def merge_coeffs(self, coeffs):
        updated_coeffs = coeffs.copy()
        updated_coeffs['id_coeff'] = coeffs['id_coeff'].repeat(coeffs['exp_coeff'].shape[0], 1)
        updated_coeffs['tex_coeff'] = coeffs['tex_coeff'].repeat(coeffs['exp_coeff'].shape[0], 1)
        return self.model.merge_coeffs(**updated_coeffs)

    def optimize_loop(self, img, lms, coeffs, optimizer, n_iter):
        for i_iter in range(n_iter):
            pred = self.model(self.merge_coeffs(**coeffs), render=True if self.cfg.loss_weights.photo != 0 else False)

            pred_img = pred['render_img'][:, :3]
            pred_mask = pred['render_img'][:, 3].detach() > 0

            losses = {}
            if self.cfg.loss_weights.landmark != 0:
                value = landmark_loss(pred['lms_proj'], lms, self.lmk_weights)
                losses['landmark'] = self.cfg.loss_weights.landmark * value

            if self.cfg.loss_weights.photo != 0:
                value = photo_loss(pred_img, img, pred_mask)
                losses['photo'] = self.cfg.loss_weights.photo * value

            if self.cfg.loss_weights.id_regularization != 0:
                value = get_reg(self.model.get_id_coeff(coeffs))
                losses['id_regularization'] = self.cfg.loss_weights.id_regularization * value

            if self.cfg.loss_weights.exp_regularization != 0:
                value = get_reg(self.model.get_exp_coeff(coeffs))
                losses['exp_regularization'] = self.cfg.loss_weights.exp_regularization * value

            if self.cfg.loss_weights.tex_regularization != 0:
                value = get_reg(self.model.get_tex_coeff(coeffs))
                losses['tex_regularization'] = self.cfg.loss_weights.tex_regularization * value

            if self.cfg.loss_weights.eye_closure != 0:
                value = closure_loss(pred['lms_proj'], lms, self.eye_closure_pairs)
                losses['eye_closure'] = self.cfg.loss_weights.eye_closure * value

            if self.cfg.loss_weights.mouth_closure != 0:
                value = closure_loss(pred['lms_proj'], lms, self.lip_closure_pairs)
                losses['mouth_closure'] = self.cfg.loss_weights.mouth_closure * value

            if self.model.lmk_type == '478' and self.cfg.loss_weights.gaze != 0:
                value = gaze_loss(pred['lms_proj'], lms, self.l_eye, self.l_iris)
                value += gaze_loss(pred['lms_proj'], lms, self.r_eye, self.r_iris)
                losses['gaze'] = self.cfg.loss_weights.gaze * value

            optimizer.zero_grad()
            total_loss = sum(losses.values())
            total_loss.backward()
            optimizer.step()

        return coeffs

    def fit_shared_coeffs(self):

        def sample_frames(n):
            align_list, lms_list = [], []
            fid = self.reader.get_frame_ptr()
            for i in np.random.permutation(self.reader.num_frames):
                if len(align_list) >= n: break

                try:
                    self.reader.set_frame_ptr(i)
                    aligned = self.reader.get()
                    lms = self.detector(aligned)
                except Exception as exception:
                    if isinstance(exception, NoFaceDetectedException):
                        with open(f'{self.timestamp}.error', 'a') as f:
                            f.write(f'No face detected in frame {i}.\n')
                    elif isinstance(exception, NoFrameReturnedException):
                        with open(f'{self.timestamp}.error', 'a') as f:
                            f.write(f'Broken frame {i}.\n')
                    else:
                        raise exception

                    self.reader.relocate_face()

                align_list.append(aligned)
                lms_list.append(lms)

            self.reader.set_frame_ptr(fid)
            return np.stack(align_list), np.stack(lms_list)

        aligned, lms = sample_frames(min(self.cfg.num_samples, self.reader.num_frames))
        lms = torch.tensor(lms, dtype=torch.float32).to(self.device)
        img = F.to_tensor(aligned).to(dtype=torch.float32, device=self.device)

        coeffs = self.init_coeffs(batch_size=aligned.shape[0])
        optimizer = self.build_optimizers(coeffs, first_frame=True)
        coeffs = self.optimize_loop(img, lms, coeffs, optimizer, self.cfg.n_iter_1)

        self.id_coeff = self.model.get_id_coeff(coeffs).detach()
        self.tex_coeff = self.model.get_tex_coeff(coeffs).detach()

    def smooth_coeffs(self, coeffs):
        if not hasattr(self, 'coeffs'):
            self.coeffs = []
        self.coeffs.append(coeffs)
        if len(self.coeffs) > self.args.smooth:
            self.coeffs.pop(0)
        return sum(self.coeffs) / len(self.coeffs)

    def save_results(self, frame_idx, frame, render, coeffs):
        if frame is not None:
            save_image(f'{self.out_dir}/frames/{str(frame_idx).zfill(6)}.png', postprocess(frame)[0])
        if render is not None:
            save_image(f'{self.out_dir}/render/{str(frame_idx).zfill(6)}.png', postprocess(render)[0])
        if coeffs is not None:
            np.save(f'{self.out_dir}/coeffs/{str(frame_idx).zfill(6)}.npy', coeffs[0].cpu().numpy())

    def run(self, save_render, save_coeffs, no_pbar=False):
        self.fit_shared_coeffs()

        first_frame = True
        coeffs = self.init_coeffs(batch_size=1)

        pbar = tqdm(range(self.reader.num_frames), dynamic_ncols=True, disable=no_pbar)
        for i in pbar:
            try:
                aligned = self.reader.get()
                lms = self.detector(aligned)
            except Exception as exception:
                if isinstance(exception, NoFaceDetectedException):
                    with open(f'{self.timestamp}.error', 'a') as f:
                        f.write(f'No face detected in frame {i}.\n')
                elif isinstance(exception, NoFrameReturnedException):
                    with open(f'{self.timestamp}.error', 'a') as f:
                        f.write(f'Broken frame {i}.\n')
                else:
                    raise exception

                first_frame = True
                self.reader.relocate_face()
                coeffs = self.init_coeffs(batch_size=1)
                continue

            lms = torch.tensor(lms[None, ...], dtype=torch.float32).to(self.device)
            img = F.to_tensor(aligned).to(dtype=torch.float32, device=self.device).unsqueeze(0)

            optimizer = self.build_optimizers(coeffs, first_frame=first_frame)
            coeffs = self.optimize_loop(img, lms, coeffs, optimizer, self.cfg.n_iter_1 if first_frame else self.cfg.n_iter_2)

            first_frame = False

            with torch.no_grad():
                out_coeffs = self.smooth_coeffs(self.merge_coeffs(**coeffs))
                pred = self.model(out_coeffs, render=save_render)

            thread = Thread(target=self.save_results, kwargs={
                'frame_idx': i, 'frame': img,
                'render': pred['render_img'][:, :3] if save_render else None,
                'coeffs': out_coeffs if save_coeffs else None
            })
            thread.start()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Face face_reconstruction tracker")

    parser.add_argument('--device', type=str, default='cuda:0', help='cuda or cpu')
    parser.add_argument('--config', type=str, default='config/default.yaml', help='config path')
    parser.add_argument('--video', type=str, required=True, help='input video path')
    parser.add_argument('--output', type=str, required=True, help='output directory')
    parser.add_argument('--save_frames', action='store_true', help='save aligned frame')
    parser.add_argument('--save_render', action='store_true', help='save render image')
    parser.add_argument('--save_coeffs', action='store_true', help='save 3DMM coefficients')
    parser.add_argument('--no_pbar', action='store_true', help='run without processing bar')

    args = parser.parse_args()
    config = OmegaConf.load(args.config)

    os.makedirs(args.output, exist_ok=True)
    if args.save_frames: os.makedirs(f'{args.output}/frames', exist_ok=True)
    if args.save_render: os.makedirs(f'{args.output}/render', exist_ok=True)
    if args.save_coeffs: os.makedirs(f'{args.output}/coeffs', exist_ok=True)

    tracker = Tracker(args.video, args.output, config, args.device)
    tracker.run(args.save_render, args.save_coeffs, args.no_pbar)
