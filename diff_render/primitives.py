import torch
import torch.nn.functional as F
from .ray import Ray
from .intersection import SurfaceIntersection
from .bsdf import DiffuseBSDF, MetallicBSDF, MirrorBSDF, DielectricBSDF
from .utils import safe_normalize
from dataclasses import dataclass
from .texture import sample_textures, sample_normal_maps

@dataclass
class GeometryData:
    v0: torch.Tensor
    v1: torch.Tensor
    v2: torch.Tensor
    n0: torch.Tensor
    n1: torch.Tensor
    n2: torch.Tensor
    uv0: torch.Tensor
    uv1: torch.Tensor
    uv2: torch.Tensor
    tan: torch.Tensor
    bitan: torch.Tensor

@dataclass
class MaterialData:
    albedo: torch.Tensor
    emission: torch.Tensor
    is_metal: torch.Tensor
    roughness: torch.Tensor
    is_dielectric: torch.Tensor
    ior: torch.Tensor
    tex_idx: torch.Tensor
    normal_idx: torch.Tensor

def _empty_mesh_data(device):
    empty3 = torch.zeros(0, 3, device=device)
    empty2 = torch.zeros(0, 2, device=device)
    empty1 = torch.zeros(0, device=device)
    empty_long = torch.zeros(0, dtype=torch.long, device=device)
    geo = GeometryData(empty3, empty3, empty3, empty3, empty3, empty3, empty2, empty2, empty2, empty3, empty3)
    mat = MaterialData(empty3, empty3, empty1.bool(), empty1, empty1.bool(), empty1, empty_long, empty_long)
    return geo, mat


class CompositeBSDF:
    """Routes material sampling through appropriate BSDF based on material type."""
    
    def __init__(self, is_metal, albedo, roughness, is_dielectric=None, ior=None):
        self.is_metal = is_metal
        self.albedo = albedo
        self.roughness = roughness
        self.is_dielectric = is_dielectric if is_dielectric is not None else torch.zeros_like(is_metal)
        self.ior = ior if ior is not None else torch.ones_like(roughness)

    def sample(self, wo: torch.Tensor, normal: torch.Tensor):
        safe_n = safe_normalize(normal)
        rand_dir = safe_normalize(torch.randn_like(safe_n))
        rand_uv = torch.rand_like(safe_n[..., :2])

        wi_diffuse, val_diffuse, pdf_diffuse = DiffuseBSDF.sample(self.albedo, wo, safe_n, rand_dir)
        wi_metallic, val_metallic, pdf_metallic = MetallicBSDF.sample(self.albedo, self.roughness, wo, safe_n, rand_uv)
        wi_dielectric, val_dielectric, pdf_dielectric = DielectricBSDF.sample(self.albedo, self.ior, wo, safe_n, rand_uv)

        is_m = self.is_metal.unsqueeze(-1)
        is_d = self.is_dielectric.unsqueeze(-1)
        
        wi = torch.where(is_m, wi_metallic, torch.where(is_d, wi_dielectric, wi_diffuse))
        val_out = torch.where(is_m, val_metallic, torch.where(is_d, val_dielectric, val_diffuse))
        pdf_out = torch.where(is_m, pdf_metallic, torch.where(is_d, pdf_dielectric, pdf_diffuse))

        value = torch.nan_to_num(val_out, nan=0.0, posinf=0.0, neginf=0.0)
        pdf = torch.nan_to_num(pdf_out, nan=0.0, posinf=0.0, neginf=0.0)
        return wi, value, pdf

    def eval(self, wo: torch.Tensor, normal: torch.Tensor, wi: torch.Tensor):
        safe_n = safe_normalize(normal)
        
        val_diffuse = DiffuseBSDF.eval(self.albedo, wo, safe_n, wi)
        val_metallic = MetallicBSDF.eval(self.albedo, self.roughness, wo, safe_n, wi)
        val_dielectric = DielectricBSDF.eval(self.albedo, self.ior, wo, safe_n, wi)
        
        is_m = self.is_metal.unsqueeze(-1)
        is_d = self.is_dielectric.unsqueeze(-1)
        
        val_out = torch.where(is_m, val_metallic, torch.where(is_d, val_dielectric, val_diffuse))
        return torch.nan_to_num(val_out, nan=0.0, posinf=0.0, neginf=0.0)

    def pdf(self, wo: torch.Tensor, normal: torch.Tensor, wi: torch.Tensor):
        safe_n = safe_normalize(normal)
        
        pdf_diffuse = DiffuseBSDF.pdf(wo, safe_n, wi)
        pdf_metallic = MetallicBSDF.pdf(self.roughness, wo, safe_n, wi)
        pdf_dielectric = DielectricBSDF.pdf(self.ior, wo, safe_n, wi)
        
        is_m = self.is_metal.unsqueeze(-1)
        is_d = self.is_dielectric.unsqueeze(-1)
        
        pdf_out = torch.where(is_m, pdf_metallic, torch.where(is_d, pdf_dielectric, pdf_diffuse))
        return torch.nan_to_num(pdf_out, nan=0.0, posinf=0.0, neginf=0.0)


class Mesh:
    """Triangle mesh with BVH acceleration."""
    
    def __init__(self, parts, textures=None, normal_maps=None, leaf_size=4, device='cuda'):
        self.parts = parts
        self.textures = textures if textures is not None else []
        self.normal_maps = normal_maps if normal_maps is not None else []
        self.leaf_size = leaf_size
        self._bvh = None

    def build_bvh(self):
        from .bvh import BVHTree
        v0 = torch.cat([p[0].v0 for p in self.parts])
        if v0.shape[0] > 0:
            v1 = torch.cat([p[0].v1 for p in self.parts])
            v2 = torch.cat([p[0].v2 for p in self.parts])
            self._bvh = BVHTree(
                v0.detach(), v1.detach(), v2.detach(),
                max_leaf_size=self.leaf_size
            )

    def intersect(self, ray: Ray) -> SurfaceIntersection:
        if self._bvh is None:
            self.build_bvh()

        lead_shape = ray.origins.shape[:-1]
        ray_o = ray.origins.reshape(-1, 3)
        ray_d = ray.dirs.reshape(-1, 3)

        v0_full = torch.cat([p[0].v0 for p in self.parts])
        v1_full = torch.cat([p[0].v1 for p in self.parts])
        v2_full = torch.cat([p[0].v2 for p in self.parts])

        t, u, v, tri_idx, hit = self._bvh.intersect(
            ray_o.detach(), ray_d.detach(),
            v0_full.detach(), v1_full.detach(), v2_full.detach()
        )

        idx = torch.clamp(tri_idx, min=0)
        v0, v1, v2 = v0_full[idx], v1_full[idx], v2_full[idx]
        
        # Differentiable re-evaluation of intersection (Möller-Trumbore)
        e1, e2 = v1 - v0, v2 - v0
        h = torch.linalg.cross(ray_d, e2, dim=-1)
        a = (e1 * h).sum(dim=-1, keepdim=True)
        f_inv = 1.0 / torch.where(a >= 0, a + 1e-8, a - 1e-8)
        
        s = ray_o - v0
        u_re = f_inv * (s * h).sum(dim=-1, keepdim=True)
        q = torch.linalg.cross(s, e1, dim=-1)
        v_re = f_inv * (ray_d * q).sum(dim=-1, keepdim=True)
        t_re = f_inv * (e2 * q).sum(dim=-1, keepdim=True)

        u = torch.where(hit, u_re.squeeze(-1), u)
        v = torch.where(hit, v_re.squeeze(-1), v)
        t = torch.where(hit, t_re.squeeze(-1), t)

        w = 1.0 - u - v
        hit_pos = w.unsqueeze(-1) * v0 + u.unsqueeze(-1) * v1 + v.unsqueeze(-1) * v2

        n0_full = torch.cat([p[0].n0 for p in self.parts])
        n1_full = torch.cat([p[0].n1 for p in self.parts])
        n2_full = torch.cat([p[0].n2 for p in self.parts])
        n0, n1, n2 = n0_full[idx], n1_full[idx], n2_full[idx]
        normal = safe_normalize(w.unsqueeze(-1) * n0 + u.unsqueeze(-1) * n1 + v.unsqueeze(-1) * n2)
        normal = torch.where(hit.unsqueeze(-1), normal, torch.tensor([0.0, 1.0, 0.0], device=ray_o.device))

        albedo_full = torch.cat([p[1].albedo for p in self.parts])
        color = albedo_full[idx]
        
        tex_idx_full = torch.cat([p[1].tex_idx for p in self.parts])
        norm_idx_full = torch.cat([p[1].normal_idx for p in self.parts])
        
        # Texture and Normal Mapping
        has_tex = tex_idx_full is not None and len(self.textures) > 0
        has_nmap = norm_idx_full is not None and len(self.normal_maps) > 0
        
        uv0_sample = self.parts[0][0].uv0
        if uv0_sample is not None and (has_tex or has_nmap):
            uv0_full = torch.cat([p[0].uv0 for p in self.parts])
            uv1_full = torch.cat([p[0].uv1 for p in self.parts])
            uv2_full = torch.cat([p[0].uv2 for p in self.parts])
            uv0, uv1, uv2 = uv0_full[idx], uv1_full[idx], uv2_full[idx]
            uv = w.unsqueeze(-1) * uv0 + u.unsqueeze(-1) * uv1 + v.unsqueeze(-1) * uv2

            if has_tex:
                color = sample_textures(uv, tex_idx_full[idx], self.textures, color)

            if has_nmap:
                tan_full = torch.cat([p[0].tan for p in self.parts])
                bitan_full = torch.cat([p[0].bitan for p in self.parts])
                T, B = tan_full[idx], bitan_full[idx]
                normal = sample_normal_maps(uv, norm_idx_full[idx], self.normal_maps, normal, T, B)

        emit_full = torch.cat([p[1].emission for p in self.parts])
        emit = emit_full[idx] * hit.unsqueeze(-1).float()
        
        is_metal_full = torch.cat([p[1].is_metal for p in self.parts])
        roughness_full = torch.cat([p[1].roughness for p in self.parts])
        is_diel_full = torch.cat([p[1].is_dielectric for p in self.parts])
        ior_full = torch.cat([p[1].ior for p in self.parts])
        
        is_m = is_metal_full[idx]
        roughness = roughness_full[idx]
        is_d = is_diel_full[idx]
        ior = ior_full[idx]
        t = torch.where(hit, t, torch.full_like(t, 1e6))

        hit_pos = torch.nan_to_num(hit_pos, nan=0.0, posinf=0.0, neginf=0.0)
        normal = torch.nan_to_num(normal, nan=0.0, posinf=0.0, neginf=0.0)
        emit = torch.nan_to_num(emit, nan=0.0, posinf=0.0, neginf=0.0)

        # Restore original tensor shapes
        def unflatten(x, extra=()):
            return x.reshape(*lead_shape, *extra)

        bsdf = CompositeBSDF(
            unflatten(is_m), unflatten(color, (3,)), unflatten(roughness), 
            unflatten(is_d), unflatten(ior)
        )
        
        return SurfaceIntersection(
            t=unflatten(t), normals=unflatten(normal, (3,)), hit_pos=unflatten(hit_pos, (3,)), 
            color=unflatten(color, (3,)), emission=unflatten(emit, (3,)), bsdf=bsdf
        )