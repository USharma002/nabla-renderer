import torch


def safe_normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-6) -> torch.Tensor:
    """Safe normalization that won't backprop through zero-magnitude vectors."""
    norm = torch.sqrt(torch.sum(x * x, dim=dim, keepdim=True) + eps)
    return x / norm
