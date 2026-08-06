import os
import xml.etree.ElementTree as ET
import numpy as np
import torch
import trimesh
from .scene import Scene
from .camera import Camera
from .bsdf import DiffuseBSDF, MetallicBSDF, MirrorBSDF, DielectricBSDF
from .primitives import GeometryData, MaterialData
from .utils import safe_normalize


class XMLSceneParser:
    """Mitsuba XML scene parser with mesh ID and transform support."""
    
    def __init__(self, device='cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.bsdf_dict = {}

    def parse_vector(self, val_str, default=None):
        if not val_str:
            return default
        parts = [float(p) for p in val_str.replace(',', ' ').strip().split() if p]
        if len(parts) == 1:
            return [parts[0], parts[0], parts[0]]
        return parts if parts else default

    def parse_color(self, elem, default=[0.8, 0.8, 0.8]):
        if elem is None:
            return default
        if 'value' in elem.attrib:
            return self.parse_vector(elem.attrib['value'], default)
        for child in elem:
            if child.tag in ['color', 'rgb', 'spectrum']:
                return self.parse_vector(child.attrib.get('value'), default)
        return default

    def parse_transform(self, elem):
        if elem is None:
            return np.eye(4, dtype=np.float32)

        M = np.eye(4, dtype=np.float32)
        for child in elem:
            tag = child.tag.lower()
            if tag == 'matrix':
                vals = [float(x) for x in child.attrib.get('value', '').replace(',', ' ').split() if x]
                if len(vals) == 16:
                    child_m = np.array(vals, dtype=np.float32).reshape(4, 4)
                    M = child_m @ M
            elif tag == 'scale':
                s = self.parse_vector(child.attrib.get('value'))
                if not s:
                    s = [float(child.attrib.get('x', 1)), float(child.attrib.get('y', 1)), float(child.attrib.get('z', 1))]
                scale_m = np.diag([s[0], s[1], s[2], 1.0]).astype(np.float32)
                M = scale_m @ M
            elif tag == 'translate':
                t = self.parse_vector(child.attrib.get('value'))
                if not t:
                    t = [float(child.attrib.get('x', 0)), float(child.attrib.get('y', 0)), float(child.attrib.get('z', 0))]
                trans_m = np.eye(4, dtype=np.float32)
                trans_m[:3, 3] = t
                M = trans_m @ M
            elif tag == 'rotate':
                axis = [float(child.attrib.get('x', 0)), float(child.attrib.get('y', 0)), float(child.attrib.get('z', 0))]
                angle_deg = float(child.attrib.get('angle', 0))
                rot_m = trimesh.transformations.rotation_matrix(np.radians(angle_deg), axis).astype(np.float32)
                M = rot_m @ M
        return M

    def parse_bsdf(self, elem):
        if elem is None:
            return DiffuseBSDF(albedo=[0.8, 0.8, 0.8], device=self.device)

        if elem.tag == 'ref' or elem.attrib.get('type') == 'ref':
            ref_id = elem.attrib.get('id') or elem.attrib.get('value')
            if ref_id in self.bsdf_dict:
                return self.bsdf_dict[ref_id]

        bsdf_type = elem.attrib.get('type', 'diffuse').lower()
        
        texture_path = None
        normal_path = None
        for child in elem:
            name = child.attrib.get('name', '')
            path = None
            
            if child.tag == 'texture':
                sub = next((s for s in child if s.tag == 'string' and s.attrib.get('name') == 'filename'), None)
                if sub is not None:
                    path = sub.attrib.get('value')
            elif child.tag == 'string' and name in ('texture', 'filename', 'normalmap', 'normal'):
                path = child.attrib.get('value')
                
            if path:
                if name in ('normalmap', 'normal'):
                    normal_path = path
                else:
                    texture_path = path

        if bsdf_type in ['diffuse', 'lambertian']:
            bsdf_obj = DiffuseBSDF(albedo=self.parse_color(elem, [0.8, 0.8, 0.8]), device=self.device)
        elif bsdf_type in ['mirror', 'conductor', 'smoothconductor']:
            bsdf_obj = MirrorBSDF(albedo=self.parse_color(elem, [1.0, 1.0, 1.0]), device=self.device)
        elif bsdf_type in ['dielectric', 'glass', 'roughdielectric']:
            ior = 1.5
            for child in elem:
                if child.attrib.get('name') in ['int_ior', 'ext_ior', 'ior']:
                    ior = float(child.attrib.get('value', 1.5))
            bsdf_obj = DielectricBSDF(ior=ior, device=self.device)
        elif bsdf_type in ['metallic', 'metal', 'roughconductor']:
            tint = self.parse_color(elem, [0.85, 0.85, 0.9])
            roughness = 0.08
            for child in elem:
                if child.attrib.get('name') == 'roughness':
                    roughness = float(child.attrib.get('value', 0.08))
            bsdf_obj = MetallicBSDF(tint=tint, roughness=roughness, device=self.device)
        else:
            bsdf_obj = DiffuseBSDF(albedo=[0.8, 0.8, 0.8], device=self.device)

        if texture_path:
            bsdf_obj.texture_path = texture_path
        if normal_path:
            bsdf_obj.normal_path = normal_path
        return bsdf_obj

    def _mesh_to_tensors(self, loaded_mesh, bsdf, radiance=None, tex_idx=-1, normal_idx=-1):
        vertices = torch.tensor(loaded_mesh.vertices, dtype=torch.float32, device=self.device)
        faces = loaded_mesh.faces
        N = len(faces)

        v0, v1, v2 = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]

        if hasattr(loaded_mesh, 'vertex_normals') and loaded_mesh.vertex_normals is not None and len(loaded_mesh.vertex_normals) > 0:
            normals = torch.tensor(loaded_mesh.vertex_normals, dtype=torch.float32, device=self.device)
            n0, n1, n2 = normals[faces[:, 0]], normals[faces[:, 1]], normals[faces[:, 2]]
            n0, n1, n2 = safe_normalize(n0), safe_normalize(n1), safe_normalize(n2)
        else:
            e1, e2 = v1 - v0, v2 - v0
            fn = safe_normalize(torch.linalg.cross(e1, e2, dim=-1))
            n0, n1, n2 = fn, fn, fn

        if hasattr(loaded_mesh, 'visual') and hasattr(loaded_mesh.visual, 'uv') and loaded_mesh.visual.uv is not None and len(loaded_mesh.visual.uv) > 0:
            uvs = torch.tensor(loaded_mesh.visual.uv, dtype=torch.float32, device=self.device)
            uv0, uv1, uv2 = uvs[faces[:, 0]], uvs[faces[:, 1]], uvs[faces[:, 2]]
        else:
            uv0 = torch.zeros((N, 2), dtype=torch.float32, device=self.device)
            uv1 = torch.zeros((N, 2), dtype=torch.float32, device=self.device)
            uv2 = torch.zeros((N, 2), dtype=torch.float32, device=self.device)

        # Compute Tangent Space
        delta_pos1 = v1 - v0
        delta_pos2 = v2 - v0
        delta_uv1 = uv1 - uv0
        delta_uv2 = uv2 - uv0

        r = 1.0 / (delta_uv1[:, 0] * delta_uv2[:, 1] - delta_uv1[:, 1] * delta_uv2[:, 0] + 1e-8)
        tan = (delta_pos1 * delta_uv2[:, 1].unsqueeze(-1) - delta_pos2 * delta_uv1[:, 1].unsqueeze(-1)) * r.unsqueeze(-1)
        bitan = (delta_pos2 * delta_uv1[:, 0].unsqueeze(-1) - delta_pos1 * delta_uv2[:, 0].unsqueeze(-1)) * r.unsqueeze(-1)
        tan = safe_normalize(tan)
        bitan = safe_normalize(bitan)

        tex_idx_tensor = torch.full((N,), tex_idx, dtype=torch.long, device=self.device)
        normal_idx_tensor = torch.full((N,), normal_idx, dtype=torch.long, device=self.device)

        albedo_vec = bsdf.albedo if hasattr(bsdf, 'albedo') else torch.tensor([0.8, 0.8, 0.8], device=self.device)
        albedo = albedo_vec.unsqueeze(0).expand(N, 3).contiguous()

        is_metal_val = isinstance(bsdf, (MetallicBSDF, MirrorBSDF))
        is_metal = torch.full((N,), is_metal_val, dtype=torch.bool, device=self.device)
        roughness_val = getattr(bsdf, 'roughness', 0.0 if is_metal_val else 1.0)
        roughness = torch.full((N,), roughness_val, dtype=torch.float32, device=self.device)

        is_diel_val = isinstance(bsdf, DielectricBSDF)
        is_dielectric = torch.full((N,), is_diel_val, dtype=torch.bool, device=self.device)
        ior_val = getattr(bsdf, 'ior', 1.0)
        ior = torch.full((N,), ior_val, dtype=torch.float32, device=self.device)

        if radiance is not None:
            emission = torch.as_tensor(radiance, dtype=torch.float32, device=self.device).unsqueeze(0).expand(N, 3).contiguous()
        else:
            emission = torch.zeros(N, 3, dtype=torch.float32, device=self.device)

        geo = GeometryData(v0, v1, v2, n0, n1, n2, uv0, uv1, uv2, tan, bitan)
        mat = MaterialData(albedo, emission, is_metal, roughness, is_dielectric, ior, tex_idx_tensor, normal_idx_tensor)
        return geo, mat

    def load(self, xml_path, override_res=None):
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"Scene XML file not found at: {xml_path}")

        base_dir = os.path.dirname(os.path.abspath(xml_path))
        tree = ET.parse(xml_path)
        root = tree.getroot()

        scene = Scene()
        camera = None
        scene_info = {'integrator': 'path', 'sample_count': 128}

        integrator_elem = root.find('integrator')
        if integrator_elem is not None:
            scene_info['integrator'] = integrator_elem.attrib.get('type', 'path')

        sampler_elem = root.find('sampler')
        if sampler_elem is not None:
            for child in sampler_elem:
                if child.attrib.get('name') in ['sampleCount', 'sample_count', 'count']:
                    scene_info['sample_count'] = int(child.attrib.get('value', 128))

        for bsdf_elem in root.findall('bsdf'):
            bsdf_id = bsdf_elem.attrib.get('id')
            if bsdf_id:
                self.bsdf_dict[bsdf_id] = self.parse_bsdf(bsdf_elem)

        camera_elem = root.find('camera') or root.find('sensor')
        if camera_elem is not None:
            fov, width, height = 60.0, 512, 512
            origin, target, up = [0.0, 1.0, 5.0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]

            def scan_camera_children(parent):
                nonlocal fov, width, height, origin, target, up
                for child in parent:
                    name = child.attrib.get('name', '').lower()
                    if child.tag == 'float' and name == 'fov':
                        fov = float(child.attrib.get('value', fov))
                    elif child.tag == 'integer' and name in ['width', 'res_x']:
                        width = int(child.attrib.get('value', width))
                    elif child.tag == 'integer' and name in ['height', 'res_y']:
                        height = int(child.attrib.get('value', height))
                    elif child.tag == 'film':
                        scan_camera_children(child)
                    elif child.tag in ['transform', 'lookat']:
                        sub_lookat = child if child.tag == 'lookat' else child.find('lookat')
                        if sub_lookat is None and child.tag == 'transform' and child.attrib.get('origin'):
                            sub_lookat = child
                        if sub_lookat is not None:
                            origin = self.parse_vector(sub_lookat.attrib.get('origin'), origin)
                            target = self.parse_vector(sub_lookat.attrib.get('target'), target)
                            up = self.parse_vector(sub_lookat.attrib.get('up'), up)

            scan_camera_children(camera_elem)

            if override_res is not None:
                if isinstance(override_res, (tuple, list)):
                    width, height = override_res[0], override_res[1]
                else:
                    width, height = override_res, override_res

            camera = Camera(pos=origin, target=target, up=up, fov=fov, res=(width, height), device=self.device)

        if camera is None:
            camera = Camera(device=self.device)

        mesh_elems = root.findall('mesh') + root.findall('shape')
        for mesh_elem in mesh_elems:
            mesh_id = mesh_elem.attrib.get('id') or mesh_elem.attrib.get('name')
            filename = mesh_elem.attrib.get('filename')
            if not filename:
                for child in mesh_elem:
                    if child.tag == 'string' and child.attrib.get('name') in ['filename', 'file']:
                        filename = child.attrib.get('value')

            if not filename:
                continue

            mesh_path = os.path.join(base_dir, filename)
            if not os.path.exists(mesh_path):
                print(f"Warning: Mesh file {mesh_path} not found.")
                continue

            loaded_mesh = trimesh.load(mesh_path, force='mesh', process=False)

            transform_elem = mesh_elem.find('transform')
            if transform_elem is not None:
                M = self.parse_transform(transform_elem)
                loaded_mesh.apply_transform(M)

            trimesh.repair.fix_winding(loaded_mesh)

            bsdf = self.parse_bsdf(mesh_elem.find('bsdf') or mesh_elem.find('ref'))

            tex_idx = -1
            if getattr(bsdf, 'texture_path', None):
                tex_path = os.path.join(base_dir, bsdf.texture_path)
                if os.path.exists(tex_path):
                    from PIL import Image
                    img = Image.open(tex_path).convert('RGB')
                    tex_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                    tex_tensor = tex_tensor.to(self.device)
                    tex_tensor.requires_grad_(True)
                    tex_idx = scene.add_texture(tex_tensor)
                else:
                    print(f"Warning: Texture file {tex_path} not found.")

            normal_idx = -1
            if getattr(bsdf, 'normal_path', None):
                nmap_path = os.path.join(base_dir, bsdf.normal_path)
                if os.path.exists(nmap_path):
                    from PIL import Image
                    img = Image.open(nmap_path).convert('RGB')
                    nmap_tensor = torch.from_numpy(np.array(img)).float() / 255.0
                    nmap_tensor = nmap_tensor.to(self.device)
                    nmap_tensor.requires_grad_(True)
                    normal_idx = scene.add_normal_map(nmap_tensor)
                else:
                    print(f"Warning: Normal map file {nmap_path} not found.")

            emitter_elem = mesh_elem.find('emitter')
            radiance = self.parse_color(emitter_elem, [10.0, 10.0, 10.0]) if emitter_elem is not None else None

            geo, mat = self._mesh_to_tensors(loaded_mesh, bsdf, radiance, tex_idx, normal_idx)
            scene.add_mesh(geo, mat, mesh_id=mesh_id)

        scene.build()
        return scene, camera, scene_info


def load_scene_from_xml(xml_path, device='cuda', override_res=None):
    return XMLSceneParser(device=device).load(xml_path, override_res=override_res)
