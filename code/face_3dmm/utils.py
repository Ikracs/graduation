import numpy as np
from scipy import sparse


class DeformationTransfer:
    '''
    Implementation of "Deformation Transfer for Triangle Meshes", used for expression transfer here.
    
    Usage:
        transfer = DeformationTransfer(**faceverse_dict, alpha=1.0)

        transfer.set_source_mesh(source_id_coeff)
        transfer.set_target_mesh(target_id_coeff)
        source_with_target_exp = transfer(target_exp_coeff)
    '''
    def __init__(self, **kwargs):
        self.alpha = kwargs.get('alpha', 0.0)

        self.id_base = kwargs['idBase'] * 0.01
        self.exp_base = kwargs['exBase_52'] * 0.01
        self.mean_shape = kwargs['meanshape'].reshape(-1) * 0.01

        self.id_dims = self.id_base.shape[1]
        self.exp_dims = self.exp_base.shape[1]

        self.num_faces = kwargs['tri'].shape[0]
        self.num_vertex = kwargs['meanshape'].shape[0]

        new_ver_inds = np.arange(self.num_vertex, self.num_vertex + self.num_faces)
        self.faces = np.concatenate((kwargs['tri'], new_ver_inds[:, None]), axis=1)

        self.reg_l = sparse.csc_matrix(
            (np.ones(self.num_vertex), [np.arange(self.num_vertex)] * 2),
            dtype=self.id_base.dtype, shape=[self.num_vertex + self.num_faces] * 2
        )

    def _span_components(self, vs):
        v1, v2, v3 = vs[self.faces.T[:3]]
        a = v2 - v1
        b = v3 - v1
        c = np.cross(a, b)
        c /= np.linalg.norm(c, axis=1, keepdims=True)
        return a, b, c

    def get_id_coeff(self, coeffs):
        return coeffs[: self.id_dims]

    def get_exp_coeff(self, coeffs):
        return coeffs[self.id_dims: self.id_dims + self.exp_dims]

    def set_source_mesh(self, id_coeff):
        self.source = (np.dot(self.id_base, id_coeff) + self.mean_shape).reshape(-1, 3)

        a, b, c = self._span_components(self.source)
        self.source_expanded = np.concatenate((self.source, self.source[self.faces[:, 0]] + c))
        self.inv_source_span = np.linalg.inv(np.stack((a, b, c), axis=-1))

    def set_target_mesh(self, id_coeff):
        self.target = (np.dot(self.id_base, id_coeff) + self.mean_shape).reshape(-1, 3)

        a, b, c = self._span_components(self.target)
        self.target_expanded = np.concatenate((self.target, self.target[self.faces[:, 0]] + c))
        self.inv_target_span = np.linalg.inv(np.stack((a, b, c), axis=-1))

        size = self.target_expanded.shape[0]

        def _expand(f, inv):
            i0, i1, i2, i3 = f
            row = np.array([0, 1, 2] * 4)
            col = np.array([i0, i1, i2, i3]).repeat(3)
            data = np.concatenate([-inv.sum(axis=0), *inv])
            return sparse.coo_matrix((data, (row, col)), shape=(3, size))

        self._A = sparse.vstack([_expand(f, inv) for f, inv in zip(self.faces, self.inv_target_span)]).tocsc()
        self.reg_r = np.concatenate((self.target, np.zeros((self.num_faces, 3), dtype=self.target.dtype)), axis=0)

    def __call__(self, exp_coeff):
        vs = np.dot(self.exp_base, exp_coeff).reshape(-1, 3) + self.source

        a, b, c = self._span_components(vs)
        C = np.concatenate((np.stack((a, b, c), axis=-1) @ self.inv_source_span).transpose(0, 2, 1))
        A = sparse.linalg.splu((self._A.T @ self._A).tocsc() + self.alpha * self.reg_l)
        x = A.solve(self._A.T @ C + self.alpha * self.reg_r)[: self.num_vertex]
        return x
