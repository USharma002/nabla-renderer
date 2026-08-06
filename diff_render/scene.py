import torch
from .ray import Ray
from .intersection import SurfaceIntersection
from .primitives import Mesh, GeometryData, MaterialData, _empty_mesh_data
from .mesh import MeshProxy


class Scene:
    def __init__(self):
        self._parts = []  # list of (geo, mat, mesh_id)
        self.mesh_map = {}
        self.textures = []
        self.normal_maps = []
        self._mesh = None

    def add_texture(self, tex: torch.Tensor) -> int:
        self.textures.append(tex)
        self._mesh = None
        return len(self.textures) - 1

    def add_normal_map(self, nmap: torch.Tensor) -> int:
        self.normal_maps.append(nmap)
        self._mesh = None
        return len(self.normal_maps) - 1

    def add_mesh(self, geo: GeometryData, mat: MaterialData, mesh_id=None):
        mesh_id = mesh_id or f"mesh_{len(self._parts)}"
        self._parts.append((geo, mat, mesh_id))
        self._mesh = None

    def build(self):
        self.mesh_map.clear()

        if not self._parts:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            geo, mat = _empty_mesh_data(device)
            self._mesh = Mesh([(geo, mat, "empty")])
            self._mesh.build_bvh()
            return

        for i, (geo, mat, mesh_id) in enumerate(self._parts):
            self.mesh_map[mesh_id] = i

        self._mesh = Mesh(self._parts, textures=self.textures, normal_maps=self.normal_maps)
        self._mesh.build_bvh()

    def get_mesh(self, mesh_id: str) -> MeshProxy:
        return MeshProxy(self, mesh_id, self.mesh_map[mesh_id])

    def intersect(self, ray: Ray) -> SurfaceIntersection:
        return self._mesh.intersect(ray)