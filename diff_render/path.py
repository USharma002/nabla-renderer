import torch
from .scene import Scene
from .camera import Camera
from .ray import Ray


class PathTracer:
    """Monte Carlo path tracer."""
    
    def __init__(self, max_depth=5, num_samples=128):
        self.max_depth = max_depth
        self.num_samples = num_samples

    def sample(self, scene: Scene, camera: Camera):
        """Render via path tracing."""
        accum_L = torch.zeros_like(camera.origins)

        for _ in range(self.num_samples):
            ray = camera.sample()
            β = torch.ones_like(ray.origins) # Path throughput
            L = torch.zeros_like(ray.origins) # Accumulated radiance

            for depth in range(self.max_depth):
                si = scene.intersect(ray)

                # Direct emission from light sources
                Le = β * si.emission
                L = L + torch.where(si.is_valid(), Le, 0.0)

                # flip normal for rays hitting the back face
                shading_n = torch.where((si.n * ray.dirs).sum(-1, keepdim=True) > 0.0, -si.n, si.n)

                # Sample BSDF to get new ray direction and update throughput
                bsdf_wi, bsdf_value, bsdf_pdf = si.bsdf.sample(-ray.dirs, shading_n)

                # Update ray for next bounce
                ray = Ray(si.p + shading_n * 1e-3, bsdf_wi)
                β = torch.where(si.is_valid(), β * bsdf_value / bsdf_pdf, 0.0)

            accum_L += L

        return accum_L / self.num_samples