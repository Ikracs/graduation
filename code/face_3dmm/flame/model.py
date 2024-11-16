import torch
import numpy as np
import torch.nn.functional as F

from face_3dmm.base_model import MorphableModel3D
from face_3dmm.flame.renderer import Renderer
from face_3dmm.flame.utils import lbs, batch_rodrigues, vertices2landmarks, rot_mat_to_euler


class FLAMEModel(MorphableModel3D):
    def __init__(self, **kwargs):
        super(FLAMEModel, self).__init__()

        self.img_size = kwargs.get('img_size', 224)
        self.uv_size = kwargs.get('uv_size', 256)

        self.scale = self.img_size / 224

        self.render_mode = kwargs.get('render_mode', 'face_eye')

        self.id_dim = kwargs.get('id_dim', 300)
        self.exp_dim = kwargs.get('exp_dim', 100)
        self.tex_dim = kwargs.get('tex_dim', 200)
        self.iet_dim = self.id_dim + self.exp_dim + self.tex_dim
        self.all_dim = self.iet_dim + 3 + 3 + 3 + 6 + 3 + 27

        self.register_buffer('mean_shape', torch.tensor(kwargs['v_template'], dtype=torch.float32).unsqueeze(0))
        self.register_buffer('id_base', torch.tensor(kwargs['shapedirs'][:, :, :self.id_dim], dtype=torch.float32))
        self.register_buffer('exp_base', torch.tensor(kwargs['shapedirs'][:, :, 300: 300 + self.exp_dim], dtype=torch.float32))
        self.register_buffer('pose_base', torch.tensor(kwargs['posedirs'], dtype=torch.float32).flatten(0, 1).permute(1, 0))

        self.register_buffer('j_regressor', torch.tensor(np.array(kwargs['J_regressor'].todense()), dtype=torch.float32))
        self.register_buffer('lbs_weights', torch.tensor(kwargs['weights'], dtype=torch.float32))

        self.register_buffer('parents', torch.tensor([-1, 0, 1, 1, 1], dtype=torch.int64))
        self.register_buffer('neck_kin_chain', torch.tensor([1, 0], dtype=torch.int64))
        self.register_buffer('tris', torch.tensor(kwargs['f'].astype(np.int64), dtype=torch.int64))

        self.lmk_type = '68'

        self.register_buffer('static_lmk_tri_inds', torch.tensor(kwargs['static_lmk_faces_idx'], dtype=torch.int64).unsqueeze(0))
        self.register_buffer('static_lmk_b_coords', torch.tensor(kwargs['static_lmk_bary_coords'], dtype=torch.float32).unsqueeze(0))
        self.register_buffer('dynamic_lmk_tri_inds', kwargs['dynamic_lmk_faces_idx'].type(torch.int64))
        self.register_buffer('dynamic_lmk_b_coords', kwargs['dynamic_lmk_bary_coords'].type(torch.float32))

        self.register_buffer('face_inds', torch.tensor(kwargs['face'], dtype=torch.int64))
        self.register_buffer('l_eyeball_inds', torch.tensor(kwargs['left_eyeball'], dtype=torch.int64))
        self.register_buffer('r_eyeball_inds', torch.tensor(kwargs['right_eyeball'], dtype=torch.int64))
        self.face_eye_inds = torch.cat((self.face_inds, self.l_eyeball_inds, self.r_eyeball_inds))

        self.register_buffer('mean_tex', torch.tensor(kwargs['mean'].reshape(1, -1), dtype=torch.float32))
        self.register_buffer('tex_base', torch.tensor(kwargs['tex_dir'].reshape(-1, 200)[:, :self.tex_dim], dtype=torch.float32).unsqueeze(0))

        self.register_buffer('uv_coords', kwargs['uv_coords'].type(torch.float32))
        self.register_buffer('uv_tris', kwargs['uv_tris'].type(torch.int64))

        if self.render_mode == 'face':
            tris, uv_tris = self._filter_tris(self.face_inds)
            self.renderer = Renderer(self.img_size, tris, uv_tris, self.uv_coords, self.uv_size)
        elif self.render_mode == 'face_eye':
            tris, uv_tris = self._filter_tris(self.face_eye_inds)
            self.renderer = Renderer(self.img_size, tris, uv_tris, self.uv_coords, self.uv_size)
        elif self.render_mode == 'full':
            self.renderer = Renderer(self.img_size, self.tris, self.uv_tris, self.uv_coords, self.uv_size)
        else:
            raise NotImplementedError(f'Unsupported render mode {self.render_mode}!')

    def _filter_tris(self, ver_inds):
        ver_mask = torch.zeros(self.mean_shape.shape[1]).scatter_(0, ver_inds, 1).to(torch.bool)
        ver2ver = {idx.item(): i for i, idx in enumerate(ver_inds)}

        tri_mask = torch.zeros_like(self.tris.flatten(), dtype=torch.bool)
        for i, idx in enumerate(self.tris.flatten()):
            tri_mask[i] = ver_mask[idx]
        tri_mask = torch.all(tri_mask.unflatten(0, (-1, 3)), dim=1)

        tris = self.tris[tri_mask].flatten()
        for i, idx in enumerate(tris):
            tris[i] = ver2ver[idx.item()]
        return tris.unflatten(0, (-1, 3)), self.uv_tris[tri_mask]

    def _find_dynamic_lms(self, pose):
        aa_pose = pose.unflatten(1, (-1, 3))[:, self.neck_kin_chain]
        r_mat = batch_rodrigues(aa_pose.flatten(0, 1)).view(pose.shape[0], -1, 3, 3)

        rel_r_mat = torch.eye(3, device=pose.device, dtype=torch.float32)
        rel_r_mat = rel_r_mat.unsqueeze(dim=0).repeat(pose.shape[0], 1, 1)
        for i in range(self.neck_kin_chain.shape[0]):
            rel_r_mat = torch.bmm(r_mat[:, i], rel_r_mat)

        yaw = torch.clamp(rot_mat_to_euler(rel_r_mat) * 180 / torch.pi, max=39)
        yaw = torch.round(yaw).type(torch.int64)

        neg_mask = (yaw < 0).type(torch.int64)
        mask = (yaw < -39).type(torch.int64)
        neg_vals = mask * 78 + (1 - mask) * (39 - yaw)
        yaw = neg_mask * neg_vals + (1 - neg_mask) * yaw

        return self.dynamic_lmk_tri_inds[yaw], self.dynamic_lmk_b_coords[yaw]

    def split_coeffs(self, coeffs, return_dict=False):
        id_coeff = self.get_id_coeff(coeffs)
        exp_coeff = self.get_exp_coeff(coeffs)
        tex_coeff = self.get_tex_coeff(coeffs)
        body_pose = self.get_body_pose(coeffs)
        neck_pose = self.get_neck_pose(coeffs)
        jaw_pose = self.get_jaw_pose(coeffs)
        eye_pose = self.get_eye_pose(coeffs)
        camera = self.get_camera(coeffs)
        gamma = self.get_gamma(coeffs)

        if return_dict:
            return {
                'id_coeff': id_coeff,
                'exp_coeff': exp_coeff,
                'tex_coeff': tex_coeff,
                'body_pose': body_pose,
                'neck_pose': neck_pose,
                'jaw_pose': jaw_pose,
                'eye_pose': eye_pose,
                'camera': camera,
                'gamma': gamma
            }

        return id_coeff, exp_coeff, tex_coeff, body_pose, neck_pose, jaw_pose, eye_pose, camera, gamma

    def merge_coeffs(self, id_coeff, exp_coeff, tex_coeff, body_pose, neck_pose, jaw_pose, eye_pose, camera, gamma):
        return torch.cat((id_coeff, exp_coeff, tex_coeff, body_pose, neck_pose, jaw_pose, eye_pose, camera, gamma), dim=1)

    def forward(self, coeffs, shape=False, render=False):
        id_coeff, exp_coeff, tex_coeff, body_pose, neck_pose, jaw_pose, eye_pose, camera, gamma = self.split_coeffs(coeffs)

        betas = torch.cat((id_coeff, exp_coeff), dim=1)
        full_pose = torch.cat([body_pose, neck_pose, jaw_pose, eye_pose], dim=1)

        vs = lbs(betas, full_pose,
                 self.mean_shape.repeat(coeffs.shape[0], 1, 1),
                 torch.cat((self.id_base, self.exp_base), dim=2),
                 self.pose_base,
                 self.j_regressor,
                 self.parents,
                 self.lbs_weights,
                 dtype=torch.float32)[0]
        vs_proj = self.project(vs, camera)

        dynamic_lmk_tri_inds, dynamic_lmk_b_coords = self._find_dynamic_lms(full_pose)
        lmk_tri_inds = torch.cat((dynamic_lmk_tri_inds, self.static_lmk_tri_inds.repeat(coeffs.shape[0], 1)), dim=1)
        lmk_b_coords = torch.cat((dynamic_lmk_b_coords, self.static_lmk_b_coords.repeat(coeffs.shape[0], 1, 1)), dim=1)
        lms_proj = vertices2landmarks(vs_proj, self.tris, lmk_tri_inds, lmk_b_coords)[:, :, :2]

        pred_dict = {'lms_proj': lms_proj}

        if shape:
            pred_dict['vs_proj'] = vs_proj

        if render:
            texture = self.build_texture(tex_coeff) / 255
            pred_dict['render_img'] = self.render(vs, vs_proj, texture, gamma)

        return pred_dict

    def build_blend_shape(self, coeffs):
        id_coeff = self.get_id_coeff(coeffs)
        exp_coeff = self.get_exp_coeff(coeffs)

        body_pose = self.get_body_pose(coeffs)
        neck_pose = self.get_neck_pose(coeffs)
        jaw_pose = self.get_jaw_pose(coeffs)
        eye_pose = self.get_eye_pose(coeffs)

        betas = torch.cat((id_coeff, exp_coeff), dim=1)
        full_pose = torch.cat([body_pose, neck_pose, jaw_pose, eye_pose], dim=1)

        vs = lbs(betas, full_pose,
                 self.mean_shape.repeat(coeffs.shape[0], 1),
                 torch.cat((self.id_base, self.exp_base), dim=1),
                 self.pose_base,
                 self.j_regressor,
                 self.parents,
                 self.lbs_weights,
                 dtype=torch.float32)[0]
        return vs

    def project(self, vs, camera):

        def orthogonal_project(vs):
            x = (vs[:, :, 0] + camera[:, None, 1]) * self.scale * camera[:, None, 0]
            y = (vs[:, :, 1] + camera[:, None, 2]) * self.scale * camera[:, None, 0]
            z = vs[:, :, 2] * self.scale * camera[:, None, 0]
            return torch.stack((x, -y, -z), dim=-1)

        return orthogonal_project(vs)

    def build_texture(self, tex_coeff):
        texture = torch.bmm(self.tex_base.repeat(tex_coeff.shape[0], 1, 1), tex_coeff.unsqueeze(2)).squeeze(2) + self.mean_tex
        return F.interpolate(texture.view(tex_coeff.shape[0], 512, 512, 3).permute(0, 3, 1, 2), size=self.uv_size)[:, [2, 1, 0]]

    def render(self, vs, vs_proj, texture, gamma):
        if self.render_mode == 'face':
            vs, vs_proj = vs[:, self.face_inds], vs_proj[:, self.face_inds]
        elif self.render_mode == 'face_eye':
            vs, vs_proj = vs[:, self.face_eye_inds], vs_proj[:, self.face_eye_inds]
        out = self.renderer(vs, vs_proj, texture, gamma.reshape(-1, 9, 3))
        return torch.cat((out['images'], out['alpha_images']), dim=1)

    def get_default_id_coeff(self, batch_size, device):
        return torch.zeros((batch_size, self.id_dim), dtype=torch.float32, device=device)

    def get_default_exp_coeff(self, batch_size, device):
        return torch.zeros((batch_size, self.exp_dim), dtype=torch.float32, device=device)

    def get_default_tex_coeff(self, batch_size, device):
        return torch.zeros((batch_size, self.tex_dim), dtype=torch.float32, device=device)

    def get_default_body_pose(self, batch_size, device):
        return torch.zeros((batch_size, 3), dtype=torch.float32, device=device)

    def get_default_neck_pose(self, batch_size, device):
        return torch.zeros((batch_size, 3), dtype=torch.float32, device=device)

    def get_default_jaw_pose(self, batch_size, device):
        return torch.zeros((batch_size, 3), dtype=torch.float32, device=device)

    def get_default_eye_pose(self, batch_size, device):
        return torch.zeros((batch_size, 6), dtype=torch.float32, device=device)

    def get_default_camera(self, batch_size, device):
        scale = torch.ones((batch_size, 1), dtype=torch.float32, device=device)
        trans = torch.zeros((batch_size, 2), dtype=torch.float32, device=device)
        return torch.cat((scale, trans), dim=1)

    def get_default_gamma(self, batch_size, device):
        gamma = torch.zeros((batch_size, 27), dtype=torch.float32, device=device)
        gamma[:, [0,  9, 18]] += 0.1
        gamma[:, [1, 10, 19]] -= 0.2
        gamma[:, [2, 11, 20]] += 0.2
        gamma[:, [4, 13, 22]] -= 0.1
        return gamma

    def get_default_coeffs(self, batch_size, device, return_dict=False):
        id_coeff = self.get_default_id_coeff(batch_size, device)
        exp_coeff = self.get_default_exp_coeff(batch_size, device)
        tex_coeff = self.get_default_tex_coeff(batch_size, device)
        body_pose = self.get_default_body_pose(batch_size, device)
        neck_pose = self.get_default_neck_pose(batch_size, device)
        jaw_pose = self.get_default_jaw_pose(batch_size, device)
        eye_pose = self.get_default_eye_pose(batch_size, device)
        camera = self.get_default_camera(batch_size, device)
        gamma = self.get_default_gamma(batch_size, device)

        if return_dict:
            return {
                'id_coeff': id_coeff,
                'exp_coeff': exp_coeff,
                'tex_coeff': tex_coeff,
                'body_pose': body_pose,
                'neck_pose': neck_pose,
                'jaw_pose': jaw_pose,
                'eye_pose': eye_pose,
                'camera': camera,
                'gamma': gamma
            }

        return self.merge_coeffs(id_coeff, exp_coeff, tex_coeff, body_pose, neck_pose, jaw_pose, eye_pose, camera, gamma)

    def get_id_coeff(self, coeffs):
        return coeffs[:, :self.id_dim]

    def get_exp_coeff(self, coeffs):
        return coeffs[:, self.id_dim: self.id_dim + self.exp_dim]

    def get_tex_coeff(self, coeffs):
        return coeffs[:, self.id_dim + self.exp_dim: self.iet_dim]

    def get_body_pose(self, coeffs):
        return coeffs[:, self.iet_dim: self.iet_dim + 3]

    def get_neck_pose(self, coeffs):
        return coeffs[:, self.iet_dim + 3: self.iet_dim + 6]

    def get_jaw_pose(self, coeffs):
        return coeffs[:, self.iet_dim + 6: self.iet_dim + 9]

    def get_eye_pose(self, coeffs):
        return coeffs[:, self.iet_dim + 9: self.iet_dim + 15]

    def get_camera(self, coeffs):
        return coeffs[:, self.iet_dim + 15: self.iet_dim + 18]

    def get_gamma(self, coeffs):
        return coeffs[:, self.iet_dim + 18:]
