import random
from collections import Iterable

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

from utils import apply_imagenet_normalization

def make_coordinate_grid_2d(spatial_size, dtype):
    h, w = spatial_size
    x = torch.arange(w).type(dtype)
    y = torch.arange(h).type(dtype)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)

    yy = y.view(-1, 1).repeat(1, w)
    xx = x.view(1, -1).repeat(h, 1)

    meshed = torch.cat([xx.unsqueeze_(2), yy.unsqueeze_(2)], 2)
    return meshed

def make_coordinate_grid_like_2d(feature):
    coords = make_coordinate_grid_2d(feature.shape[2:], feature.dtype)
    coords = coords.unsqueeze(0).repeat(feature.shape[0], 1, 1, 1)
    return coords.to(feature.device)

def make_coordinate_grid_3d(spatial_size, dtype):
    d, h, w = spatial_size
    x = torch.arange(w).type(dtype)
    y = torch.arange(h).type(dtype)
    z = torch.arange(d).type(dtype)

    x = (2 * (x / (w - 1)) - 1)
    y = (2 * (y / (h - 1)) - 1)
    z = (2 * (z / (d - 1)) - 1)
   
    yy = y.view(1, -1, 1).repeat(d, 1, w)
    xx = x.view(1, 1, -1).repeat(d, h, 1)
    zz = z.view(-1, 1, 1).repeat(1, h, w)

    meshed = torch.cat([xx.unsqueeze_(3), yy.unsqueeze_(3), zz.unsqueeze_(3)], 3)
    return meshed

def make_coordinate_grid_like_3d(feature):
    coords = make_coordinate_grid_3d(feature.shape[2:], feature.dtype)
    coords = coords.unsqueeze(0).repeat(feature.shape[0], 1, 1, 1, 1)
    return coords.to(feature.device)

def resize_deformation_3d(motion, spatial_size):
    spatial_size = [spatial_size] * 3  if not isinstance(spatial_size, Iterable) else spatial_size
    return F.interpolate(motion.permute(0, 4, 1, 2, 3), size=spatial_size, mode='trilinear', align_corners=False).permute(0, 2, 3, 4, 1)

def resize_deformation_2d(motion, spatial_size):
    spatial_size = [spatial_size] * 2  if not isinstance(spatial_size, Iterable) else spatial_size
    return F.interpolate(motion.permute(0, 3, 1, 2), size=spatial_size, mode='bilinear', align_corners=False).permute(0, 2, 3, 1)

def apply_deformation(feature, motion):
    return F.grid_sample(feature, motion, mode='bilinear', align_corners=False)

def stitch_deformation(foreground, foreground_mask, background, threshold=0.5):
    mask = foreground_mask.squeeze(1)[..., None] > threshold
    return foreground * mask + background * torch.logical_not(mask)

def bins2degree(logits):
    angle_bins = torch.arange(66).to(dtype=logits.dtype, device=logits.device)
    degrees = F.softmax(logits, dim=-1) * angle_bins
    return torch.sum(degrees, dim=-1) * 3 - 99

def degrees2rotation(yaw, pitch, roll, radian=False):
    yaw = yaw if radian else torch.deg2rad(yaw)
    pitch = pitch if radian else torch.deg2rad(pitch)
    roll = roll if radian else torch.deg2rad(roll)

    yaw = yaw.unsqueeze(1)
    pitch = pitch.unsqueeze(1)
    roll = roll.unsqueeze(1)

    yaw_mat = torch.cat([
        torch.cos(yaw), torch.zeros_like(yaw), torch.sin(yaw), 
        torch.zeros_like(yaw), torch.ones_like(yaw), torch.zeros_like(yaw),
        -torch.sin(yaw), torch.zeros_like(yaw), torch.cos(yaw)], dim=1
    ).view(yaw.shape[0], 3, 3)

    pitch_mat = torch.cat([
        torch.ones_like(pitch), torch.zeros_like(pitch), torch.zeros_like(pitch), 
        torch.zeros_like(pitch), torch.cos(pitch), -torch.sin(pitch),
        torch.zeros_like(pitch), torch.sin(pitch), torch.cos(pitch)], dim=1
    ).view(pitch.shape[0], 3, 3)

    roll_mat = torch.cat([
        torch.cos(roll), -torch.sin(roll), torch.zeros_like(roll),  
        torch.sin(roll), torch.cos(roll), torch.zeros_like(roll),
        torch.zeros_like(roll), torch.zeros_like(roll), torch.ones_like(roll)], dim=1
    ).view(roll.shape[0], 3, 3)

    rotation = torch.einsum('bij,bjk,bkm->bim', pitch_mat, yaw_mat, roll_mat)
    return rotation

def project_mask(mask_3d, img_size):
    mask_2d = mask_3d.max(dim=2, keepdim=False)[0]
    return F.interpolate(mask_2d, size=img_size)


class DataSampler:
    def __init__(self, num_epochs, start=10, end=5000, schedule='quad'):
        self.num_epochs = num_epochs

        if schedule == 'constant':
            self.sampling_windows = start * np.ones(num_epochs)
        elif schedule == 'linear':
            self.sampling_windows = np.linspace(start, end, num_epochs)
        elif schedule == 'quad':
            self.sampling_windows = np.linspace(start ** 0.5, end ** 0.5, num_epochs) ** 2
        else:
            raise NotImplementedError(f'Unsupported sampling scheduler {schedule}!')

        self.epoch = 0

    def step(self):
        self.epoch += 1

    def __call__(self, frames):
        sampling_window = int(self.sampling_windows[self.epoch].round())

        half_sw = sampling_window // 2
        src_fid = random.randint(0, len(frames) - 1)
        dri_fid = random.choice(range(src_fid - half_sw, src_fid + half_sw))
        return frames[src_fid], frames[dri_fid]


class ImagePyramide(torch.nn.Module):
    def __init__(self, scales, num_channels):
        super(ImagePyramide, self).__init__()
        downs = {}
        for scale in scales:
            downs[str(scale).replace('.', '-')] = AntiAliasInterpolation2d(num_channels, scale)
        self.downs = nn.ModuleDict(downs)

    def forward(self, x):
        out_dict = {}
        for scale, down_module in self.downs.items():
            out_dict['prediction_' + str(scale).replace('-', '.')] = down_module(x)
        return out_dict


class AntiAliasInterpolation2d(nn.Module):
    def __init__(self, channels, scale):
        super(AntiAliasInterpolation2d, self).__init__()
        sigma = (1 / scale - 1) / 2
        kernel_size = 2 * round(sigma * 4) + 1

        self.ka = kernel_size // 2
        self.kb = self.ka - 1 if kernel_size % 2 == 0 else self.ka

        sigma, kernel_size = [sigma, sigma], [kernel_size, kernel_size]

        meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size], indexing='ij')

        kernel = 1
        for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
            mean = (size - 1) / 2
            kernel *= torch.exp(-(mgrid - mean) ** 2 / (2 * std ** 2))

        kernel = kernel / torch.sum(kernel)
        kernel = kernel.view(1, 1, *kernel.size())
        kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

        self.register_buffer('weight', kernel)
        self.scale, self.groups = scale, channels
        self.int_inv_scale = int(1 / scale)

    def forward(self, input):
        if self.scale == 1.0:
            return input.clone()

        out = F.pad(input, (self.ka, self.kb, self.ka, self.kb))
        out = F.conv2d(out, weight=self.weight, groups=self.groups)
        out = out[:, :, ::self.int_inv_scale, ::self.int_inv_scale]
        return out


class Vgg19(torch.nn.Module):
    def __init__(self, weights=models.VGG19_Weights.DEFAULT):
        super(Vgg19, self).__init__()
        vgg = models.vgg19(weights=weights).features

        self.slice1 = torch.nn.Sequential(*[vgg[i] for i in range(2)])
        self.slice2 = torch.nn.Sequential(*[vgg[i] for i in range(2, 7)])
        self.slice3 = torch.nn.Sequential(*[vgg[i] for i in range(7, 12)])
        self.slice4 = torch.nn.Sequential(*[vgg[i] for i in range(12, 21)])
        self.slice5 = torch.nn.Sequential(*[vgg[i] for i in range(21, 30)])

    def forward(self, x):
        h_relu1 = self.slice1(apply_imagenet_normalization(x))
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        return [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]
