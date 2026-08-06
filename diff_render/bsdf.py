import torch
from .utils import safe_normalize


class BSDF:
    def __init__(self, albedo=[0.8, 0.8, 0.8], device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.albedo = torch.as_tensor(albedo, dtype=torch.float32, device=self.device)


class DiffuseBSDF(BSDF):
    @staticmethod
    def sample(albedo, wo, normal, rand_noise):
        wi = torch.where(torch.sum(rand_noise * normal, dim=-1, keepdim=True) < 0, -rand_noise, rand_noise)
        value = DiffuseBSDF.eval(albedo, wo, normal, wi)
        pdf = DiffuseBSDF.pdf(wo, normal, wi)
        return wi, value, pdf

    @staticmethod
    def eval(albedo, wo, normal, wi):
        cos_theta = torch.clamp(torch.sum(normal * wi, dim=-1, keepdim=True), min=0.0)
        return albedo / torch.pi * cos_theta

    @staticmethod
    def pdf(wo, normal, wi):
        cos_theta = torch.clamp(torch.sum(normal * wi, dim=-1, keepdim=True), min=0.0)
        return torch.where(cos_theta > 0, torch.full_like(cos_theta, 1.0 / (2.0 * torch.pi)), torch.zeros_like(cos_theta))


class MetallicBSDF(BSDF):
    def __init__(self, tint=[0.85, 0.85, 0.9], roughness=0.08, device='cuda'):
        super().__init__(albedo=tint, device=device)
        self.roughness = roughness

    @staticmethod
    def sample(albedo, roughness, wo, normal, rand_noise):
        # 1. Compute Blinn-Phong exponent alpha
        alpha = 2.0 / (roughness**2 + 1e-4) - 2.0
        alpha = torch.clamp(alpha, min=1.0)
        
        # Align alpha with rand_noise by adding trailing dimensions
        while alpha.ndim < rand_noise.ndim:
            alpha = alpha.unsqueeze(-1)

        # 2. Extract 2D uniform random variables
        u1 = torch.clamp(rand_noise[..., 0:1], 1e-5, 1.0 - 1e-5)
        u2 = torch.clamp(rand_noise[..., 1:2], 0.0, 1.0)

        # 3. Differentiable Inverse-CDF sampling
        exponent = 1.0 / (alpha + 1.0)
        cos_theta = u1 ** exponent
        sin_theta = torch.sqrt(torch.clamp(1.0 - cos_theta**2, min=0.0))
        phi = 2.0 * torch.pi * u2

        local_x = sin_theta * torch.cos(phi)
        local_y = sin_theta * torch.sin(phi)
        local_z = cos_theta

        # 4. Construct ONB frame
        reflect_dir = safe_normalize(-wo + 2.0 * torch.sum(wo * normal, dim=-1, keepdim=True) * normal)
        
        up = torch.where(torch.abs(reflect_dir[..., 2:3]) < 0.9, 
                         torch.tensor([0.0, 0.0, 1.0], device=reflect_dir.device), 
                         torch.tensor([1.0, 0.0, 0.0], device=reflect_dir.device))
        tangent = safe_normalize(torch.cross(up, reflect_dir, dim=-1))
        bitangent = torch.cross(reflect_dir, tangent, dim=-1)

        wi = local_x * tangent + local_y * bitangent + local_z * reflect_dir
        wi = safe_normalize(wi)
        wi = torch.where(torch.sum(wi * normal, dim=-1, keepdim=True) < 0, reflect_dir, wi)

        value = MetallicBSDF.eval(albedo, roughness, wo, normal, wi)
        pdf = MetallicBSDF.pdf(roughness, wo, normal, wi)
        return wi, value, pdf

    @staticmethod
    def pdf(roughness, wo, normal, wi):
        reflect_dir = safe_normalize(-wo + 2.0 * torch.sum(wo * normal, dim=-1, keepdim=True) * normal)
        cos_alpha = torch.clamp(torch.sum(reflect_dir * wi, dim=-1, keepdim=True), min=0.0)
        
        alpha = 2.0 / (roughness**2 + 1e-4) - 2.0
        alpha = torch.clamp(alpha, min=1.0)
        
        # Align alpha with cos_alpha by adding trailing dimensions
        while alpha.ndim < cos_alpha.ndim:
            alpha = alpha.unsqueeze(-1)

        return ((alpha + 1.0) / (2.0 * torch.pi)) * (cos_alpha ** alpha)

    @staticmethod
    def eval(albedo, roughness, wo, normal, wi):
        return albedo * MetallicBSDF.pdf(roughness, wo, normal, wi)
    

class MirrorBSDF(BSDF):
    @staticmethod
    def sample(albedo, wo, normal):
        reflect_dir = -wo + 2.0 * torch.sum(wo * normal, dim=-1, keepdim=True) * normal
        wi = safe_normalize(reflect_dir)
        return wi, albedo, torch.ones_like(wi[..., :1])

    @staticmethod
    def eval(albedo, wo, normal, wi):
        return torch.zeros_like(albedo)

    @staticmethod
    def pdf(wo, normal, wi):
        return torch.zeros_like(wo[..., 0:1])


class DielectricBSDF(BSDF):
    def __init__(self, ior=1.5, device='cuda'):
        super().__init__(albedo=[1.0, 1.0, 1.0], device=device)
        self.ior = ior

    @staticmethod
    def sample(albedo, ior, wo, normal, rand_noise):
        cos_theta_i = torch.sum(wo * normal, dim=-1, keepdim=True)
        entering = cos_theta_i > 0
        
        eta = torch.where(entering, 1.0 / ior.unsqueeze(-1), ior.unsqueeze(-1) / 1.0)
        n = torch.where(entering, normal, -normal)
        cos_theta_i = torch.abs(cos_theta_i)
        
        r0 = ((1.0 - ior.unsqueeze(-1)) / (1.0 + ior.unsqueeze(-1)))**2
        fresnel = r0 + (1.0 - r0) * (1.0 - cos_theta_i)**5
        
        sin2_theta_t = eta**2 * (1.0 - cos_theta_i**2)
        total_internal_reflection = sin2_theta_t >= 1.0
        
        cos_theta_t = torch.sqrt(torch.clamp(1.0 - sin2_theta_t, min=0.0))
        refract_dir = eta * (-wo) + (eta * cos_theta_i - cos_theta_t) * n
        refract_dir = safe_normalize(refract_dir)
        
        reflect_dir = -wo + 2.0 * cos_theta_i * n
        reflect_dir = safe_normalize(reflect_dir)
        
        do_reflect = (rand_noise[..., 0:1] < fresnel) | total_internal_reflection
        wi = torch.where(do_reflect, reflect_dir, refract_dir)
        pdf = torch.where(do_reflect, fresnel, 1.0 - fresnel)
        pdf = torch.where(total_internal_reflection, torch.ones_like(pdf), pdf)
        return wi, albedo * pdf, pdf

    @staticmethod
    def eval(albedo, ior, wo, normal, wi):
        return torch.zeros_like(albedo)

    @staticmethod
    def pdf(ior, wo, normal, wi):
        return torch.zeros_like(wo[..., 0:1])