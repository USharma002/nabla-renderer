import torch
from .scene import Scene
from .camera import Camera
from .ray import Ray


def _flip_normal(n, d):
    """Flip shading normal to face against the ray direction."""
    return torch.where((n * d).sum(-1, keepdim=True) > 0, -n, n)


def L_i(scene: Scene, x, wi, max_depth):
    """Estimate incoming radiance without building an autograd graph."""
    with torch.no_grad():
        L = torch.zeros_like(x)
        throughput = torch.ones_like(x)
        ray = Ray(x, wi)
        for _ in range(max_depth):
            si = scene.intersect(ray)
            valid = si.is_valid()
            n = _flip_normal(si.n, ray.dirs)

            L += torch.where(valid, throughput * si.emission, 0.0)
            wi, bsdf_value, bsdf_pdf = si.bsdf.sample(-ray.dirs, n)
            throughput = torch.where(valid, throughput * bsdf_value / bsdf_pdf, 0.0)
            ray = Ray(si.p + n * 1e-3, wi)
        return L


class RBPathTracer:
    """Radiative Backpropagation (Nimier-David et al. 2020)."""

    def __init__(self, max_depth=5, num_samples=128):
        self.max_depth = max_depth
        self.num_samples = num_samples

    def sample_path(self, scene: Scene, camera: Camera, seed: int = 42):
        """Primal render, independent of the adjoint pass."""
        torch.manual_seed(seed)
        accum = torch.zeros_like(camera.origins)

        for _ in range(self.num_samples):
            rays = camera.sample()
            accum += L_i(scene, rays.origins, rays.dirs, self.max_depth)

        return accum / self.num_samples

    def radiative_backprop_sample(self, scene: Scene, x, wo, weight):
        """radiative_backprop_sample(π, x, ω_o, weight), unrolled over max_depth bounces."""
        weight = weight.detach()
        ray = Ray(x, wo)

        for depth in range(self.max_depth):
            si = scene.intersect(ray)
            valid = si.is_valid()
            n = _flip_normal(si.n, ray.dirs)

            Le = torch.where(valid, si.emission, torch.zeros_like(si.emission))
            if Le.requires_grad:
                (Le * weight).sum().backward()

            if depth + 1 == self.max_depth:
                break

            wo = -ray.dirs.detach()
            wi, bsdf_value, bsdf_pdf = si.bsdf.sample(wo, n.detach())
            f_s = si.bsdf.eval(wo, n, wi.detach())
            y = si.p.detach() + n.detach() * 1e-3
            Li = L_i(scene, y, wi.detach(), self.max_depth - depth - 1)

            adjoint = torch.where(valid, weight * Li / bsdf_pdf.detach(), 0.0)
            if f_s.requires_grad:
                (f_s * adjoint.detach()).sum().backward()

            with torch.no_grad():
                weight = torch.where(valid, weight * bsdf_value / bsdf_pdf, 0.0)
                ray = Ray(si.p + n * 1e-3, wi)

    def radiative_backprop(self, scene: Scene, camera: Camera, dL):
        """radiative_backprop(π, δ_y): seed each sensor ray with weight = δ_y / num_samples."""
        for _ in range(self.num_samples):
            rays = camera.sample()
            weight = dL / self.num_samples
            self.radiative_backprop_sample(scene, rays.origins, rays.dirs, weight)