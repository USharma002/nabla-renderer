import torch
import torch.nn.functional as F
from .ray import Ray


def _radiance(scene, ray, max_depth):
    L = torch.zeros_like(ray.origins)
    throughput = torch.ones_like(ray.origins)

    with torch.no_grad():
        for _ in range(max_depth):
            si = scene.intersect(ray)
            valid = si.is_valid()
            n = torch.where((si.n * ray.dirs).sum(-1, keepdim=True) > 0, -si.n, si.n)

            L += torch.where(valid, throughput * si.emission, 0.0)
            wi, f_s, pdf = si.bsdf.sample(-ray.dirs, n)
            throughput = torch.where(valid, throughput * f_s / pdf, 0.0)
            ray = Ray(si.p + n * 1e-3, wi)

    return L


class EdgeSampler:
    def __init__(self, num_samples=1024, max_depth=5, epsilon=1e-4):
        self.num_samples = num_samples
        self.max_depth = max_depth
        self.epsilon = epsilon

    def _edges(self, scene, camera):
        geo = scene._mesh.geo
        vertices = torch.stack((geo.v0, geo.v1, geo.v2), dim=1)
        screen, z = camera.project(vertices)
        e1 = screen[:, 1] - screen[:, 0]
        e2 = screen[:, 2] - screen[:, 0]
        area = (e1[:, 0] * e2[:, 1] - e1[:, 1] * e2[:, 0]).detach()

        corners = ((0, 1), (1, 2), (2, 0))
        adjacency = {}
        positions = vertices.detach().cpu()
        for face in range(vertices.shape[0]):
            for i, j in corners:
                a = tuple(positions[face, i].tolist())
                b = tuple(positions[face, j].tolist())
                key = tuple(sorted((a, b)))
                adjacency.setdefault(key, []).append((face, i, j))

        edges = []
        for adjacent in adjacency.values():
            signs = area[[face for face, _, _ in adjacent]]
            if len(adjacent) > 1 and not (signs.min() < 0 and signs.max() > 0):
                continue

            face, i, j = adjacent[0]
            a = vertices[face, i]
            endpoints_a = []
            endpoints_b = []
            for adjacent_face, adjacent_i, adjacent_j in adjacent:
                x = vertices[adjacent_face, adjacent_i]
                y = vertices[adjacent_face, adjacent_j]
                same_order = torch.linalg.vector_norm(x.detach() - a.detach()) <= torch.linalg.vector_norm(y.detach() - a.detach())
                endpoints_a.append(torch.where(same_order, x, y))
                endpoints_b.append(torch.where(same_order, y, x))
            edges.append((
                torch.stack(endpoints_a).mean(0),
                torch.stack(endpoints_b).mean(0),
            ))

        if not edges:
            return None

        world0 = torch.stack([edge[0] for edge in edges])
        world1 = torch.stack([edge[1] for edge in edges])
        p0, z0 = camera.project(world0)
        p1, z1 = camera.project(world1)
        in_front = (z0 > 0) & (z1 > 0)
        return p0[in_front], p1[in_front]

    def sample(self, scene, camera):
        
        edges = self._edges(scene, camera)
        image = torch.zeros(camera.height, camera.width, 3, device=camera.device)
        if edges is None or edges[0].shape[0] == 0:
            return image

        p0, p1 = edges
        tangent = p1 - p0
        lengths = torch.linalg.vector_norm(tangent, dim=-1).detach()
        total_length = lengths.sum()
        if total_length == 0:
            return image

        edge = torch.multinomial(lengths, self.num_samples, replacement=True)
        t = torch.rand(self.num_samples, 1, device=p0.device)
        p = torch.lerp(p0[edge], p1[edge], t)
        tangent = F.normalize(tangent[edge].detach(), dim=-1)
        normal = torch.stack((-tangent[:, 1], tangent[:, 0]), dim=-1)

        L_plus = _radiance(scene, camera.ray(p + self.epsilon * normal), self.max_depth)
        L_minus = _radiance(scene, camera.ray(p - self.epsilon * normal), self.max_depth)
        jump = (L_minus - L_plus).detach()

        row, col, inside = camera.pixel(p)
        weight = total_length / (self.num_samples * camera.pixel_area)
        motion = ((p - p.detach()) * normal).sum(-1, keepdim=True)
        value = weight * jump * motion * inside.unsqueeze(-1)
        return image.index_put((row, col), value, accumulate=True)