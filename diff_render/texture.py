import torch
import torch.nn.functional as F
from .utils import safe_normalize

def sample_textures(hit_uv: torch.Tensor, tex_idx: torch.Tensor, textures: list[torch.Tensor], base_color: torch.Tensor) -> torch.Tensor:
    """
    Samples textures and applies them to the base color for matching indices.
    """
    for i, tex in enumerate(textures):
        mask = (tex_idx == i)
        if not mask.any():
            continue
        # Map [0, 1] to [-1, 1] for grid_sample
        uv_masked = hit_uv[mask] * 2.0 - 1.0
        # grid_sample expects [B, C, H, W] for grid and [B, H, W, 2] for coordinates.
        grid = uv_masked.view(1, 1, -1, 2)
        tex_input = tex.permute(2, 0, 1).unsqueeze(0) # [1, C, H, W]
        sampled = F.grid_sample(tex_input, grid, align_corners=False, mode='bilinear', padding_mode='reflection')
        sampled = sampled.squeeze(0).squeeze(1).transpose(0, 1) # [N, C]
        base_color[mask] = sampled
    return base_color

def sample_normal_maps(hit_uv: torch.Tensor, norm_idx: torch.Tensor, normal_maps: list[torch.Tensor], base_normal: torch.Tensor, hit_T: torch.Tensor, hit_B: torch.Tensor) -> torch.Tensor:
    """
    Samples normal maps and perturbs the base normal using TBN matrices.
    """
    for i, nmap in enumerate(normal_maps):
        mask = (norm_idx == i)
        if not mask.any():
            continue
        
        uv_masked = hit_uv[mask] * 2.0 - 1.0
        grid = uv_masked.view(1, 1, -1, 2)
        nmap_input = nmap.permute(2, 0, 1).unsqueeze(0)
        sampled = F.grid_sample(nmap_input, grid, align_corners=False, mode='bilinear', padding_mode='reflection')
        sampled = sampled.squeeze(0).squeeze(1).transpose(0, 1)
        
        # Normal maps usually store (x,y,z) in [0,1]. Map back to [-1, 1].
        sampled_n = sampled * 2.0 - 1.0
        
        c_T = hit_T[mask]
        c_B = hit_B[mask]
        c_N = base_normal[mask]

        # Gram-Schmidt orthogonalization
        c_T = safe_normalize(c_T - c_N * (c_T * c_N).sum(-1, keepdim=True))
        # Recompute B to ensure orthogonality and correct handedness
        c_B = safe_normalize(torch.linalg.cross(c_N, c_T, dim=-1))

        # TBN transform: perturbed_normal = T * sx + B * sy + N * sz
        mapped_normal = c_T * sampled_n[..., 0:1] + c_B * sampled_n[..., 1:2] + c_N * sampled_n[..., 2:3]
        
        base_normal[mask] = safe_normalize(mapped_normal)
        
    return base_normal
