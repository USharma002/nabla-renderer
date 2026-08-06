import torch


class Ray:
    def __init__(self, origins: torch.Tensor, dirs: torch.Tensor):
        self.origins = origins
        self.dirs = dirs

    def at(self, t: torch.Tensor) -> torch.Tensor:
        """Point along ray at distance t."""
        return self.origins + t.unsqueeze(-1) * self.dirs
