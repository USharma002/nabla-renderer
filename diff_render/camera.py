import torch
import torch.nn.functional as F
from .ray import Ray


class Camera:
    """Pinhole perspective camera."""

    def __init__(self, pos=[0.0, 1.1, 2.2], target=[0.0, 0.35, -0.4], up=[0.0, 1.0, 0.0], fov=60.0, res=128, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.fov = torch.as_tensor(float(fov), device=self.device)
        self.width, self.height = (res if isinstance(res, (tuple, list)) else (res, res))
        self.up = torch.as_tensor(up, dtype=torch.float32, device=self.device)
        self.look_at(pos, target)

    def look_at(self, pos, target, up=None):
        """Set camera pose and compute basis vectors."""
        self.pos = torch.as_tensor(pos, dtype=torch.float32, device=self.device)
        self.target = torch.as_tensor(target, dtype=torch.float32, device=self.device)
        if up is not None:
            self.up = torch.as_tensor(up, dtype=torch.float32, device=self.device)

        self.forward = F.normalize(self.target - self.pos, dim=0)
        self.right = F.normalize(torch.linalg.cross(self.forward, self.up), dim=0)
        self.cam_up = torch.linalg.cross(self.right, self.forward)

        self.half_h = torch.tan(torch.deg2rad(self.fov) / 2.0)
        self.aspect = self.width / self.height
        self.half_w = self.half_h * self.aspect
        self.pixel_area = 4.0 * self.aspect / (self.width * self.height)

        self.origins = self.pos.view(1, 1, 3).expand(self.height, self.width, 3).contiguous()
        self.res = (self.width, self.height)

    def set_resolution(self, width, height=None):
        if height is None and isinstance(width, (tuple, list)):
            width, height = width
        self.width, self.height = int(width), int(height or width)
        self.look_at(self.pos, self.target)

    def sample(self, count=None):
        """Generate jittered primary rays over the full sensor."""
        h, w = self.height, self.width
        pw = 2.0 * self.half_w / w
        ph = 2.0 * self.half_h / h

        iy = torch.arange(h, device=self.device)
        ix = torch.arange(w, device=self.device)
        gy, gx = torch.meshgrid(
            (1.0 - 2.0 * (iy + 0.5) / h) * self.half_h,
            (2.0 * (ix + 0.5) / w - 1.0) * self.half_w,
            indexing='ij')

        shape = (h, w) if count is None else (count, h, w)
        gx = gx + (torch.rand(shape, device=self.device) - 0.5) * pw
        gy = gy + (torch.rand(shape, device=self.device) - 0.5) * ph
        dirs = gx.unsqueeze(-1) * self.right + gy.unsqueeze(-1) * self.cam_up + self.forward
        dirs = F.normalize(dirs, dim=-1)
        o = self.origins if count is None else self.origins.unsqueeze(0).expand(count, -1, -1, -1)
        return Ray(o, dirs)

    def world_to_ndc(self, p):
        """World points → NDC (u,v) + depth z."""
        rel = p - self.pos
        xc = (rel * self.right).sum(-1)
        yc = (rel * self.cam_up).sum(-1)
        zc = (rel * self.forward).sum(-1)
        denom = (zc + 1e-8) * self.half_h
        return torch.stack((xc / denom, yc / denom), dim=-1), zc

    def _ndc_to_col_row(self, ndc):
        """NDC (u,v) → continuous pixel (col, row)."""
        col = (ndc[..., 0] / self.aspect + 1) * 0.5 * self.width
        row = (1 - ndc[..., 1]) * 0.5 * self.height
        return col, row

    def ndc_to_ray(self, ndc):
        """NDC (u,v) → world rays."""
        dirs = ndc[..., :1] * self.half_h * self.right + ndc[..., 1:] * self.half_h * self.cam_up + self.forward
        return Ray(self.pos.expand_as(dirs), F.normalize(dirs, dim=-1))

    def ndc_to_pixel(self, ndc):
        """NDC (u,v) → integer pixel (row, col) + in-frame mask."""
        col, row = self._ndc_to_col_row(ndc)
        col, row = col.floor().long(), row.floor().long()
        inside = (row >= 0) & (row < self.height) & (col >= 0) & (col < self.width)
        return row.clamp(0, self.height - 1), col.clamp(0, self.width - 1), inside

    def world_to_pixel(self, p):
        """World points → continuous pixel (col, row) + depth z."""
        ndc, zc = self.world_to_ndc(p)
        col, row = self._ndc_to_col_row(ndc)
        return torch.stack((col, row), dim=-1), zc

    def pixel_to_ray(self, px):
        """Pixel (col, row) → world rays."""
        u = (2.0 * px[..., 0] / self.width - 1.0) * self.aspect
        v = 1.0 - 2.0 * px[..., 1] / self.height
        return self.ndc_to_ray(torch.stack((u, v), dim=-1))

    def project(self, p):
        return self.world_to_ndc(p)

    def ray(self, ndc):
        return self.ndc_to_ray(ndc)

    def pixel(self, ndc):
        return self.ndc_to_pixel(ndc)
