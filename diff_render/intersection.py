import torch


class SurfaceIntersection:
    """Surface intersection data."""
    
    def __init__(self, t: torch.Tensor, normals: torch.Tensor, hit_pos: torch.Tensor, 
                 color: torch.Tensor, emission: torch.Tensor = None, bsdf=None,
                 motion_pos: torch.Tensor = None):
        self.t = t
        self.n = normals
        self.p = hit_pos
        self.motion_p = motion_pos if motion_pos is not None else hit_pos
        self.color = color
        self.emission = emission if emission is not None else torch.zeros_like(color)
        self.bsdf = bsdf

    def is_valid(self):
        """Check if intersection is within valid range."""
        return ((self.t < 1e5) & (self.t > 0)).unsqueeze(-1)