import torch
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
from pytorch3d.renderer import RasterizationSettings, blending, PointLights
from pytorch3d.renderer import MeshRasterizer, HardFlatShader, MeshRenderer

from face_3dmm.base_model import MorphableModel3D


class FaceVerseModel(MorphableModel3D):
    def __init__(self, **kwargs):
        super(FaceVerseModel, self).__init__()

        self.img_size = kwargs.get('img_size', 256)
        self.focal = 1315 / 512 * self.img_size

        self.render_mode = kwargs.get('render_mode', 'face_eye')

        self.id_dim = kwargs['idBase'].shape[1]
        self.exp_dim = kwargs['exBase_52'].shape[1]
        self.tex_dim = kwargs['texBase'].shape[1]
        self.iet_dim = self.id_dim + self.exp_dim + self.tex_dim
        self.all_dim = self.iet_dim + 27 + 3 + 3 + 4

        self.lmk_type = '478'

        self.register_buffer('kp_inds', torch.tensor(kwargs['mediapipe_keypoints'], dtype=torch.int64))

        self.register_buffer('mean_shape', torch.tensor(kwargs['meanshape'].reshape(1, -1) * 0.01, dtype=torch.float32))
        self.register_buffer('mean_tex', torch.tensor(kwargs['meantex'].reshape(1, -1), dtype=torch.float32))
        self.register_buffer('uv_tex', torch.tensor(kwargs['uvtex'], dtype=torch.float32))

        self.register_buffer('ver_inds', torch.tensor(kwargs['ver_inds'], dtype=torch.int64))
        self.register_buffer('face_mask', torch.tensor(kwargs['face_mask'], dtype=torch.bool))

        self.register_buffer('face_eye_mask', self.face_mask.clone())
        self.face_eye_mask[self.ver_inds[0]: self.ver_inds[1]] = True
        self.face_eye_mask[self.ver_inds[1]: self.ver_inds[2]] = True

        self.register_buffer('id_base', torch.tensor(kwargs['idBase'] * 0.01, dtype=torch.float32).unsqueeze(0))
        self.register_buffer('exp_base', torch.tensor(kwargs['exBase_52'] * 0.01, dtype=torch.float32).unsqueeze(0))
        self.register_buffer('tex_base', torch.tensor(kwargs['texBase'], dtype=torch.float32).unsqueeze(0))

        self.register_buffer('tris', torch.tensor(kwargs['tri'], dtype=torch.int64))
        self.register_buffer('point_buf', torch.tensor(kwargs['point_buf'], dtype=torch.int64))

        self.register_buffer('l_eye_inds', torch.arange(self.ver_inds[0], self.ver_inds[1]))
        self.register_buffer('r_eye_inds', torch.arange(self.ver_inds[1], self.ver_inds[2]))

        self.register_buffer('l_eye_base', self.id_base.unflatten(1, (-1, 3))[:, self.l_eye_inds].flatten(1, 2))
        self.register_buffer('r_eye_base', self.id_base.unflatten(1, (-1, 3))[:, self.r_eye_inds].flatten(1, 2))

        self.register_buffer('l_eye_mean', self.mean_shape.unflatten(1, (-1, 3))[:, self.l_eye_inds].flatten(1, 2))
        self.register_buffer('r_eye_mean', self.mean_shape.unflatten(1, (-1, 3))[:, self.r_eye_inds].flatten(1, 2))

        self.register_buffer('face_inds', torch.where(self.face_mask)[0])
        self.register_buffer('face_eye_inds', torch.where(self.face_eye_mask)[0])

        self.register_buffer('face_tris', self._filter_tris(self.face_mask))
        self.register_buffer('face_eye_tris', self._filter_tris(self.face_eye_mask))

        self.exp_labels = kwargs['exp_name_list_52']

        self.renderer = self._get_renderer()

    def _filter_tris(self, ver_mask):
        ver_inds = torch.where(ver_mask)[0]
        ver2ver = {idx.item(): i for i, idx in enumerate(ver_inds)}

        tri_mask = torch.zeros_like(self.tris.flatten(), dtype=torch.bool)
        for i, idx in enumerate(self.tris.flatten()):
            tri_mask[i] = ver_mask[idx]
        tri_mask = torch.all(tri_mask.unflatten(0, (-1, 3)), dim=1)

        tris = self.tris[tri_mask].flatten()
        for i, idx in enumerate(tris):
            tris[i] = ver2ver[idx.item()]
        return tris.unflatten(0, (-1, 3))

    @property
    def camera_dist(self):
        return 10

    @property
    def camera_elev(self):
        return 0

    @property
    def camera_azim(self):
        return 0

    def _get_renderer(self):
        R, T = look_at_view_transform(self.camera_dist, self.camera_elev, self.camera_azim)
        fov = 2 * np.arctan(self.img_size // 2 / self.focal) * 180 / np.pi
        cameras = FoVPerspectiveCameras(R=R, T=T, znear=0.01, zfar=50, fov=fov)

        raster_settings = RasterizationSettings(image_size=self.img_size, blur_radius=0.0, faces_per_pixel=1)
        rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)

        blend_params = blending.BlendParams(background_color=[0, 0, 0])
        lights = PointLights(location=[[0.0, 0.0, 1e5]], ambient_color=[[1.0, 1.0, 1.0]],
                             specular_color=[[0.0, 0.0, 0.0]], diffuse_color=[[0.0, 0.0, 0.0]])
        shader = HardFlatShader(cameras=cameras, lights=lights, blend_params=blend_params)

        return MeshRenderer(rasterizer=rasterizer, shader=shader)

    def to(self, device):
        super(FaceVerseModel, self).to(device)
        self.renderer.to(device)
        return self

    def get_lms(self, vs):
        lms = vs[:, self.kp_inds, :]
        return lms

    def split_coeffs(self, coeffs, return_dict=False):
        id_coeff = self.get_id_coeff(coeffs)
        exp_coeff = self.get_exp_coeff(coeffs)
        tex_coeff = self.get_tex_coeff(coeffs)
        angle = self.get_angle(coeffs)
        gamma = self.get_gamma(coeffs)
        trans = self.get_trans(coeffs)
        eye_coeff = self.get_eye_coeff(coeffs)

        if return_dict:
            return {
                'id_coeff': id_coeff,
                'exp_coeff': exp_coeff,
                'tex_coeff': tex_coeff,
                'angle': angle,
                'gamma': gamma,
                'trans': trans,
                'eye_coeff': eye_coeff
            }

        return id_coeff, exp_coeff, tex_coeff, angle, gamma, trans, eye_coeff

    def merge_coeffs(self, id_coeff, exp_coeff, tex_coeff, angle, gamma, trans, eye_coeff):
        coeffs = torch.concat([id_coeff, exp_coeff, tex_coeff, angle, gamma, trans, eye_coeff], dim=1)
        return coeffs

    def forward(self, coeffs, shape=False, render=False, render_uv=False):
        rotation = self.compute_rotation_matrix(self.get_angle(coeffs))
        trans = self.get_trans(coeffs)

        vs = self.build_blend_shape(coeffs)

        if shape:
            z_zoom = self.focal / (self.camera_dist - trans[:, None, 2:])
            z_proj = torch.bmm(vs, rotation)[:, :, 2:] * z_zoom + self.img_size // 2

        if render:
            texture = self.build_texture(self.get_tex_coeff(coeffs))
            norm = torch.bmm(self.compute_norm(vs), rotation)
            color = self.add_illumination(texture, norm, self.get_gamma(coeffs))

        vs = self.rigid_transform(vs, rotation, trans)

        if shape:
            vs_proj = torch.cat((self.project(vs), z_proj), dim=2)
            pred_dict = {'vs_proj': vs_proj, 'lms_proj': self.get_lms(vs_proj)[:, :, :2]}
        else:
            pred_dict = {'lms_proj': self.project(self.get_lms(vs))}

        if render:
            pred_dict.update({'render_img': self.render(vs, color)})

        if render_uv:
            uv_tex = self.uv_tex.repeat(vs.shape[0], 1, 1)
            pred_dict.update({'uv_img': self.render(vs, uv_tex)})

        return pred_dict

    def project(self, vs):

        def perspective_project(vs):
            x = vs[:, :, 0] * self.focal / (self.camera_dist - vs[:, :, 2])
            y = vs[:, :, 1] * self.focal / (self.camera_dist - vs[:, :, 2])
            return torch.stack((x, -y), dim=-1) + self.img_size // 2

        return perspective_project(vs)

    def get_l_eye_center(self, id_coeff):
        with torch.no_grad():
            l_eye_base = self.l_eye_base.repeat(id_coeff.shape[0], 1, 1)
            l_eye = torch.bmm(l_eye_base, id_coeff.unsqueeze(2)).squeeze(2) + self.l_eye_mean
            l_eye = l_eye.unflatten(1, (-1, 3))
            l_eye[:, :, 2] += 0.005
            l_eye_center = torch.mean(l_eye, dim=1, keepdim=True)
        return l_eye_center

    def get_r_eye_center(self, id_coeff):
        with torch.no_grad():
            r_eye_base = self.r_eye_base.repeat(id_coeff.shape[0], 1, 1)
            r_eye = torch.bmm(r_eye_base, id_coeff.unsqueeze(2)).squeeze(2) + self.r_eye_mean
            r_eye = r_eye.unflatten(1, (-1, 3))
            r_eye[:, :, 2] += 0.005
            r_eye_center = torch.mean(r_eye, dim=1, keepdim=True)
        return r_eye_center

    def build_texture(self, tex_coeff):
        texture = torch.bmm(self.tex_base.repeat(tex_coeff.shape[0], 1, 1), tex_coeff.unsqueeze(2)).squeeze(2) + self.mean_tex
        return texture.view(tex_coeff.shape[0], -1, 3)

    def compute_norm(self, vs):
        v1 = vs[:, self.tris[:, 0], :]
        v2 = vs[:, self.tris[:, 1], :]
        v3 = vs[:, self.tris[:, 2], :]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = e1.cross(e2)

        v_norm = face_norm[:, self.point_buf, :].sum(2)
        v_norm = v_norm / (v_norm.norm(dim=2).unsqueeze(2) + 1e-9)
        return v_norm

    def build_blend_shape(self, coeffs):
        id_coeff = self.get_id_coeff(coeffs)
        exp_coeff = self.get_exp_coeff(coeffs)
        eye_coeff = self.get_eye_coeff(coeffs)

        l_eye_center = self.get_l_eye_center(id_coeff)
        r_eye_center = self.get_r_eye_center(id_coeff)

        l_eye_rot = self.compute_eye_rotation_matrix(eye_coeff[:, :2])
        r_eye_rot = self.compute_eye_rotation_matrix(eye_coeff[:, 2:])

        vs = self.mean_shape.unflatten(1, (-1, 3))
        vs = vs + torch.bmm(self.id_base.repeat(coeffs.shape[0], 1, 1), id_coeff.unsqueeze(2)).squeeze(2).unflatten(1, (-1, 3))
        vs = vs + torch.bmm(self.exp_base.repeat(coeffs.shape[0], 1, 1), exp_coeff.unsqueeze(2)).squeeze(2).unflatten(1, (-1, 3))

        vs[:, self.l_eye_inds] = self.eye_rotate(vs[:, self.l_eye_inds], l_eye_center, l_eye_rot)
        vs[:, self.r_eye_inds] = self.eye_rotate(vs[:, self.r_eye_inds], r_eye_center, r_eye_rot)
        return vs

    def compute_eye_rotation_matrix(self, eye):
        sinx = torch.sin(eye[:, 0])
        siny = torch.sin(eye[:, 1])
        cosx = torch.cos(eye[:, 0])
        cosy = torch.cos(eye[:, 1])

        rotation = torch.eye(3, dtype=eye.dtype, device=eye.device)
        rotation = rotation[None, None, :, :].repeat(3, eye.shape[0], 1, 1)

        rotation[0, :, 1, 1] = cosx
        rotation[0, :, 1, 2] = -sinx
        rotation[0, :, 2, 1] = sinx
        rotation[0, :, 2, 2] = cosx
        rotation[1, :, 0, 0] = cosy
        rotation[1, :, 0, 2] = siny
        rotation[1, :, 2, 0] = -siny
        rotation[1, :, 2, 2] = cosy

        rotation = torch.bmm(rotation[1], rotation[0])
        return rotation.permute(0, 2, 1)

    def compute_rotation_matrix(self, angle):
        sinx = torch.sin(angle[:, 0])
        siny = torch.sin(angle[:, 1])
        sinz = torch.sin(angle[:, 2])
        cosx = torch.cos(angle[:, 0])
        cosy = torch.cos(angle[:, 1])
        cosz = torch.cos(angle[:, 2])

        rotation = torch.eye(3, dtype=angle.dtype, device=angle.device)
        rotation = rotation[None, None, :, :].repeat(3, angle.shape[0], 1, 1)

        rotation[0, :, 1, 1] = cosx
        rotation[0, :, 1, 2] = -sinx
        rotation[0, :, 2, 1] = sinx
        rotation[0, :, 2, 2] = cosx
        rotation[1, :, 0, 0] = cosy
        rotation[1, :, 0, 2] = siny
        rotation[1, :, 2, 0] = -siny
        rotation[1, :, 2, 2] = cosy
        rotation[2, :, 0, 0] = cosz
        rotation[2, :, 0, 1] = -sinz
        rotation[2, :, 1, 0] = sinz
        rotation[2, :, 1, 1] = cosz

        rotation = torch.bmm(rotation[2], torch.bmm(rotation[1], rotation[0]))
        return rotation.permute(0, 2, 1)

    def add_illumination(self, texture, norm, gamma):
        gamma = gamma.view(-1, 3, 9).clone()
        gamma[:, :, 0] += 0.8
        gamma = gamma.permute(0, 2, 1)

        a0 = np.pi
        a1 = 2 * np.pi / np.sqrt(3.0)
        a2 = 2 * np.pi / np.sqrt(8.0)
        c0 = 1 / np.sqrt(4 * np.pi)
        c1 = np.sqrt(3.0) / np.sqrt(4 * np.pi)
        c2 = 3 * np.sqrt(5.0) / np.sqrt(12 * np.pi)
        d0 = 0.5 / np.sqrt(3.0)

        norm = norm.view(-1, 3)
        nx, ny, nz = norm[:, 0], norm[:, 1], norm[:, 2]
        arrH = []

        arrH.append(a0 * c0 * (nx * 0 + 1))
        arrH.append(-a1 * c1 * ny)
        arrH.append(a1 * c1 * nz)
        arrH.append(-a1 * c1 * nx)
        arrH.append(a2 * c2 * nx * ny)
        arrH.append(-a2 * c2 * ny * nz)
        arrH.append(a2 * c2 * d0 * (3 * nz.pow(2) - 1))
        arrH.append(-a2 * c2 * nx * nz)
        arrH.append(a2 * c2 * 0.5 * (nx.pow(2) - ny.pow(2)))

        H = torch.stack(arrH, 1)
        Y = H.view(gamma.shape[0], texture.shape[1], 9)
        lighting = torch.bmm(Y, gamma)

        color = texture * lighting
        return color

    def eye_rotate(self, eye, eye_center, rotation):
        return torch.bmm(eye - eye_center, rotation) + eye_center + 0.01

    def rigid_transform(self, vs, rotation, trans):
        return torch.bmm(vs, rotation) + trans.unsqueeze(1)

    def render(self, vs, color):
        if self.render_mode == 'face':
            vs = vs[:, self.face_inds, :]
            color = color[:, self.face_inds, :]
            tris = self.face_tris.repeat(vs.shape[0], 1, 1)
        elif self.render_mode == 'face_eye':
            vs = vs[:, self.face_eye_inds, :]
            color = color[:, self.face_eye_inds, :]
            tris = self.face_eye_tris.repeat(vs.shape[0], 1, 1)
        elif self.render_mode == 'full':
            tris = self.tris.repeat(vs.shape[0], 1, 1)
        else:
            raise NotImplementedError(f'Unsupported render mode {self.render_mode}!')
        render_img = self.renderer(Meshes(vs, tris, TexturesVertex(color)))
        return render_img.permute(0, 3, 1, 2) / 255

    def get_default_id_coeff(self, batch_size, device):
        return torch.zeros((batch_size, self.id_dim), dtype=torch.float32, device=device)

    def get_default_exp_coeff(self, batch_size, device):
        return torch.zeros((batch_size, self.exp_dim), dtype=torch.float32, device=device)

    def get_default_tex_coeff(self, batch_size, device):
        return torch.zeros((batch_size, self.tex_dim), dtype=torch.float32, device=device)

    def get_default_gamma(self, batch_size, device):
        gamma = torch.zeros((batch_size, 27), dtype=torch.float32, device=device)
        gamma[:, [0,  9, 18]] += 0.1
        gamma[:, [1, 10, 19]] -= 0.2
        gamma[:, [2, 11, 20]] += 0.2
        gamma[:, [4, 13, 22]] -= 0.1
        return gamma

    def get_default_angle(self, batch_size, device):
        return torch.zeros((batch_size, 3), dtype=torch.float32, device=device)

    def get_default_trans(self, batch_size, device):
        return torch.zeros((batch_size, 3), dtype=torch.float32, device=device)

    def get_default_eye_coeff(self, batch_size, device):
        return torch.zeros((batch_size, 4), dtype=torch.float32, device=device)

    def get_default_l_eye_coeff(self, batch_size, device):
        return torch.zeros((batch_size, 2), dtype=torch.float32, device=device)

    def get_default_r_eye_coeff(self, batch_size, device):
        return torch.zeros((batch_size, 2), dtype=torch.float32, device=device)

    def get_default_coeffs(self, batch_size, device, return_dict=False):
        id_coeff = self.get_default_id_coeff(batch_size, device)
        exp_coeff = self.get_default_exp_coeff(batch_size, device)
        tex_coeff = self.get_default_tex_coeff(batch_size, device)
        gamma = self.get_default_gamma(batch_size, device)
        angle = self.get_default_angle(batch_size, device)
        trans = self.get_default_trans(batch_size, device)
        eye_coeff = self.get_default_eye_coeff(batch_size, device)

        if return_dict:
            return {
                'id_coeff': id_coeff,
                'exp_coeff': exp_coeff,
                'tex_coeff': tex_coeff,
                'angle': angle,
                'gamma': gamma,
                'trans': trans,
                'eye_coeff': eye_coeff
            }

        return self.merge_coeffs(id_coeff, exp_coeff, tex_coeff, angle, gamma, trans, eye_coeff)

    def get_id_coeff(self, coeffs):
        return coeffs[:, : self.id_dim]

    def get_exp_coeff(self, coeffs):
        return coeffs[:, self.id_dim: self.id_dim + self.exp_dim]

    def get_tex_coeff(self, coeffs):
        return coeffs[:, self.id_dim + self.exp_dim: self.iet_dim]

    def get_gamma(self, coeffs):
        return coeffs[:, self.iet_dim + 3: self.iet_dim + 30]

    def get_angle(self, coeffs):
        return coeffs[:, self.iet_dim: self.iet_dim + 3]

    def get_trans(self, coeffs):
        return coeffs[:, self.iet_dim + 30: self.iet_dim + 33]

    def get_eye_coeff(self, coeffs):
        return coeffs[:, self.iet_dim + 33:]

    def get_l_eye_coeff(self, coeffs):
        return coeffs[:, self.iet_dim + 33: self.iet_dim + 35]

    def get_r_eye_coeff(self, coeffs):
        return coeffs[:, self.iet_dim + 35:]
