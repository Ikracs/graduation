import torch.nn as nn


class MorphableModel3D(nn.Module):

    def split_coeffs(self, coeffs):
        raise NotImplementedError

    def merge_coeffs(self, **kwargs):
        raise NotImplementedError

    def get_default_coeffs(self, batch_size, device):
        raise NotImplementedError

    def build_blend_shape(self, coeffs):
        raise NotImplementedError

    def build_texture(self, tex_coeff):
        raise NotImplementedError

    def forward(self, coeffs, **kwargs):
        raise NotImplementedError

    def project(self, vs, **kwargs):
        raise NotImplementedError
