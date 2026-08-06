import torch
from .scene import Scene
from .camera import Camera
from .ray import Ray


def relative_grad(x, eps=1e-10):
    x_d = x.detach()
    safe = x_d.abs() > eps
    denom = torch.where(safe, x_d, torch.ones_like(x_d))
    return torch.where(safe, x / denom, torch.zeros_like(x))


def _flip_normal(n, d):
    """Flip shading normal to face against the ray direction."""
    return torch.where((n * d).sum(-1, keepdim=True) > 0, -n, n)


class PRBPathTracer:
    def __init__(self, max_depth=5, num_samples=128):
        self.max_depth = max_depth
        self.num_samples = num_samples

        self.seed = 42
        self._primal_samples = []

    def sample_path(self, scene: Scene, camera: Camera, seed: int = 42):
        self.seed = seed
        torch.manual_seed(seed)

        self._primal_samples.clear()
        accum = torch.zeros_like(camera.origins)

        with torch.no_grad():
            for _ in range(self.num_samples):
                ray = camera.sample()
                L = torch.zeros_like(ray.origins)
                throughput = torch.ones_like(ray.origins)

                for _ in range(self.max_depth):
                    si = scene.intersect(ray)
                    valid = si.is_valid()
                    n = _flip_normal(si.n, ray.dirs)

                    L += torch.where(valid, throughput * si.emission, 0.0)
                    wi, bsdf_value, bsdf_pdf = si.bsdf.sample(-ray.dirs, n)
                    throughput = torch.where(valid & (bsdf_pdf > 1e-8), throughput * bsdf_value / torch.clamp(bsdf_pdf, min=1e-8), 0.0)
                    ray = Ray(si.p + n * 1e-3, wi)

                self._primal_samples.append(L)
                accum += L

        return accum / self.num_samples

    def sample_adjoint(self, scene: Scene, camera: Camera, _primal_img, dL):
        torch.manual_seed(self.seed)

        scale = dL / self.num_samples

        for sample_idx in range(self.num_samples):
            ray = camera.sample()
            ray = Ray(ray.origins.detach(), ray.dirs.detach())
            L = self._primal_samples[sample_idx]
            throughput = torch.ones_like(ray.origins)

            for _ in range(self.max_depth):
                si = scene.intersect(ray)
                valid = si.is_valid()
                n = _flip_normal(si.n, ray.dirs)

                # L -= β · L_e  ->  L is now suffix radiance R_k
                Le = throughput.detach() * si.emission
                Le = torch.where(valid, Le, torch.zeros_like(Le))
                L = L - Le.detach()

                # same random stream -> identical (wi, w)
                wi, bsdf_value, bsdf_pdf = si.bsdf.sample(-ray.dirs, n)

                # differentiable f_s re-evaluation
                f_s = si.bsdf.eval((-ray.dirs).detach(), n, wi.detach())

                # dπ += J_{Le}^T(dL)  +  J_{f_s}^T(dL * R_k / f_s)
                Lo = Le + torch.where(valid, L * relative_grad(f_s), 0.0)
                (scale.detach() * Lo).sum().backward()

                # advance (fully detached)
                with torch.no_grad():
                    throughput = torch.where(valid & (bsdf_pdf > 1e-8), throughput * bsdf_value / torch.clamp(bsdf_pdf, min=1e-8),
                                             torch.zeros_like(throughput))
                    ray = Ray((si.p + n * 1e-3).detach(), wi.detach())