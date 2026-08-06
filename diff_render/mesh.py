import torch
from .utils import safe_normalize


class MeshProxy:
    _GEO_ATTRS = {'v0', 'v1', 'v2', 'n0', 'n1', 'n2', 'uv0', 'uv1', 'uv2', 'tan', 'bitan'}
    _MAT_ATTRS = {'albedo', 'emission', 'is_metal', 'roughness', 'is_dielectric', 'ior', 'tex_idx', 'normal_idx'}

    def __init__(self, scene, mesh_id, part_idx):
        object.__setattr__(self, '_scene', scene)
        object.__setattr__(self, 'id', mesh_id)
        object.__setattr__(self, 'part_idx', part_idx)

    def __getattr__(self, name):
        geo, mat, _ = self._scene._parts[self.part_idx]
        if name in self._GEO_ATTRS:
            return getattr(geo, name)
        if name in self._MAT_ATTRS:
            return getattr(mat, name)
        raise AttributeError(name)

    def __setattr__(self, name, val):
        geo, mat, _ = self._scene._parts[self.part_idx]
        if name in self._GEO_ATTRS:
            setattr(geo, name, torch.as_tensor(val, dtype=torch.float32, device=geo.v0.device))
            if name in {'v0', 'v1', 'v2'}:
                self._scene._mesh.build_bvh()
            return
        if name in self._MAT_ATTRS:
            dtype = torch.bool if name in {'is_metal', 'is_dielectric'} else torch.long if name in {'tex_idx', 'normal_idx'} else torch.float32
            setattr(mat, name, torch.as_tensor(val, dtype=dtype, device=mat.albedo.device))
            return
        object.__setattr__(self, name, val)

    def translate(self, offset=[0.0, 0.0, 0.0]):
        geo, _, _ = self._scene._parts[self.part_idx]
        off = torch.as_tensor(offset, dtype=torch.float32, device=geo.v0.device)

        geo.v0 = geo.v0 + off
        geo.v1 = geo.v1 + off
        geo.v2 = geo.v2 + off

        self._scene._mesh.build_bvh()

    def apply_transform(self, matrix):
        geo, _, _ = self._scene._parts[self.part_idx]
        M = torch.as_tensor(matrix, dtype=torch.float32, device=geo.v0.device)
        R, t = M[:3, :3], M[:3, 3]

        geo.v0 = (geo.v0 @ R.T) + t
        geo.v1 = (geo.v1 @ R.T) + t
        geo.v2 = (geo.v2 @ R.T) + t

        R_inv_t = torch.linalg.inv(R).T

        geo.n0 = safe_normalize(geo.n0 @ R_inv_t.T)
        geo.n1 = safe_normalize(geo.n1 @ R_inv_t.T)
        geo.n2 = safe_normalize(geo.n2 @ R_inv_t.T)
        
        self._scene._mesh.build_bvh()
