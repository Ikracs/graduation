import torch
import numpy as np

from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex
from pytorch3d.renderer import look_at_view_transform, FoVPerspectiveCameras
from pytorch3d.renderer import RasterizationSettings, blending, PointLights
from pytorch3d.renderer import MeshRasterizer, HardFlatShader, MeshRenderer

from face_3dmm.base_model import MorphableModel3D


class BaselFaceModel(MorphableModel3D):
    def __init__(self, **kwargs):
        super(BaselFaceModel, self).__init__()

        self.img_size = kwargs.get('img_size', 224)
        self.focal = 1015 / 224 * self.img_size

        self.id_dim = kwargs['idBase'].shape[1]
        self.exp_dim = kwargs['exBase'].shape[1]
        self.tex_dim = kwargs['texBase'].shape[1]
        self.iet_dim = self.id_dim + self.exp_dim + self.tex_dim
        self.all_dim = self.iet_dim + 27 + 3 + 3

        self.lmk_type = '68'

        self.register_buffer('kp_inds', torch.tensor(kwargs['keypoints'] - 1, dtype=torch.int64).squeeze(0))

        self.register_buffer('mean_shape', torch.tensor(kwargs['meanshape'], dtype=torch.float32))
        self.register_buffer('mean_tex', torch.tensor(kwargs['meantex'], dtype=torch.float32))

        self.register_buffer('id_base', torch.tensor(kwargs['idBase'], dtype=torch.float32).unsqueeze(0))
        self.register_buffer('exp_base', torch.tensor(kwargs['exBase'], dtype=torch.float32).unsqueeze(0))
        self.register_buffer('tex_base', torch.tensor(kwargs['texBase'], dtype=torch.float32).unsqueeze(0))

        self.register_buffer('tris', torch.tensor(kwargs['tri'] - 1, dtype=torch.int64))
        self.register_buffer('point_buf', torch.tensor(kwargs['point_buf'] - 1, dtype=torch.int64))

        self.register_buffer('id_base_kp', self.id_base.unflatten(1, (-1, 3))[:, self.kp_inds, ...].flatten(1, 2))
        self.register_buffer('exp_base_kp', self.exp_base.unflatten(1, (-1, 3))[:, self.kp_inds, ...].flatten(1, 2))
        self.register_buffer('mean_shape_kp', self.mean_shape.unflatten(1, (-1, 3))[:, self.kp_inds, ...].flatten(1, 2))

        self.renderer = self._get_renderer()

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
        super(BaselFaceModel, self).to(device)
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

        if return_dict:
            return {
                'id_coeff': id_coeff,
                'exp_coeff': exp_coeff,
                'tex_coeff': tex_coeff,
                'angle': angle,
                'gamma': gamma,
                'trans': trans
            }

        return id_coeff, exp_coeff, tex_coeff, angle, gamma, trans

    def merge_coeffs(self, id_coeff, exp_coeff, tex_coeff, angle, gamma, trans):
        coeffs = torch.concat([id_coeff, exp_coeff, tex_coeff, angle, gamma, trans], dim=1)
        return coeffs

    def forward(self, coeffs, shape=False, render=False):
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

        return pred_dict

    def project(self, vs):

        def perspective_project(vs):
            x = vs[:, :, 0] * self.focal / (self.camera_dist - vs[:, :, 2])
            y = vs[:, :, 1] * self.focal / (self.camera_dist - vs[:, :, 2])
            return torch.stack((x, -y), dim=-1) + self.img_size // 2

        return perspective_project(vs)

    def compute_norm(self, vs):
        v1 = vs[:, self.tris[:, 0], :]
        v2 = vs[:, self.tris[:, 1], :]
        v3 = vs[:, self.tris[:, 2], :]
        e1 = v1 - v2
        e2 = v2 - v3
        face_norm = e1.cross(e2)
        empty = torch.zeros_like(face_norm[:, -1:])
        face_norm = torch.cat((face_norm, empty), dim=1)

        v_norm = face_norm[:, self.point_buf, :].sum(2)
        v_norm = v_norm / v_norm.norm(dim=2).unsqueeze(2)
        return v_norm

    def build_texture(self, tex_coeff):
        texture = torch.bmm(self.tex_base.repeat(tex_coeff.shape[0], 1, 1), tex_coeff.unsqueeze(2)).squeeze(2) + self.mean_tex
        return texture.view(tex_coeff.shape[0], -1, 3)

    def build_blend_shape(self, coeffs):
        vs = self.mean_shape.unflatten(1, (-1, 3))
        vs = vs + torch.bmm(self.id_base.repeat(coeffs.shape[0], 1, 1), self.get_id_coeff(coeffs).unsqueeze(2)).squeeze(2).unflatten(1, (-1, 3))
        vs = vs + torch.bmm(self.exp_base.repeat(coeffs.shape[0], 1, 1), self.get_exp_coeff(coeffs).unsqueeze(2)).squeeze(2).unflatten(1, (-1, 3))
        return vs

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
        Y = H.view(texture.shape[0], texture.shape[1], 9)
        lighting = torch.bmm(Y, gamma)

        color = texture * lighting
        return color

    def rigid_transform(self, vs, rotation, trans):
        return torch.bmm(vs, rotation) + trans.unsqueeze(1)

    def render(self, vs, color):
        render_img = self.renderer(Meshes(vs, self.tris.repeat(vs.shape[0], 1, 1), TexturesVertex(color)))
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

    def get_default_coeffs(self, batch_size, device, return_dict=False):
        id_coeff = self.get_default_id_coeff(batch_size, device)
        exp_coeff = self.get_default_exp_coeff(batch_size, device)
        tex_coeff = self.get_default_tex_coeff(batch_size, device)
        gamma = self.get_default_gamma(batch_size, device)
        angle = self.get_default_angle(batch_size, device)
        trans = self.get_default_trans(batch_size, device)

        if return_dict:
            return {
                'id_coeff': id_coeff,
                'exp_coeff': exp_coeff,
                'tex_coeff': tex_coeff,
                'angle': angle,
                'gamma': gamma,
                'trans': trans
            }

        return self.merge_coeffs(id_coeff, exp_coeff, tex_coeff, angle, gamma, trans)

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
