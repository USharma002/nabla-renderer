# Nabla Renderer

A differentiable, physically-based renderer built in PyTorch. Nabla (∇) computes gradients of rendered images with respect to scene parameters (geometry, materials, textures), so shapes, BSDFs, and textures can be optimized directly from image-space losses.<div align="center">

| Base Render | Gradient Map |
| :---: | :---: |
| <img src="assets/teapot/teapot_base.png" width="100%"> | <img src="assets/teapot/teapot_fd_grad_x_shift_h_0.01.png" width="100%"> |

</div>

## Project Goals

- **Correctness first**: every gradient estimator is validated against finite-difference references before being used for optimization.
- **First principles**: naive autodiff, boundary-aware estimators, and adjoint/replay-based methods, implemented rather than only read about.
- **Practical inverse rendering**: recover geometry, textures, and normal maps from target images via gradient descent.
- **Modular, inspectable code**: small, readable modules over a monolithic renderer.

## Features

- **Differentiable ray-triangle intersection**: the BVH traversal is a non-differentiable lookup, but barycentric coordinates, hit point, depth, and interpolated normal are re-derived against the hit triangle inside the autograd graph, so gradients flow into vertex positions, normals, and camera parameters.
- **Custom BVH**: SAH construction with two interchangeable backends: a CUDA kernel (via `cupy`) for fast primal traversal, and a pure PyTorch fallback for portability/debugging.
- **Mitsuba-style XML scene loading**: meshes (OBJ/PLY via `trimesh`), nested transforms, named/referenced BSDFs, textures, normal maps, area lights, camera/sampler settings.
- **BSDFs**: diffuse (Lambertian), rough metallic (Phong-style specular), perfect mirror, and dielectric (Fresnel-weighted reflection/refraction, Schlick's approximation).
- **Differentiable texture & normal mapping**: bilinear `grid_sample` lookups, tangent-space normal mapping with per-shading-point Gram-Schmidt re-orthogonalization.
- **Live scene editing (`MeshProxy`)**: read/write any per-triangle attribute directly (vertices, normals, albedo, roughness, IOR, UVs), with automatic BVH rebuilds on geometry edits.
- **Chunked Monte Carlo path tracer**: BSDF-sampled unidirectional path tracing, batched in chunks to control memory at high sample counts.
- **Gradient estimators**: Finite Differences (ground truth reference), PyTorch Autodiff (AD), **Path Replay Backpropagation (PRB)**, and **Radiative Backpropagation (RB)** for memory-efficient gradient computation.
- **Boundary-aware estimators (coming soon)**: Primary edge sampling, Loubet et al. reparameterization, and silhouette projection.

### Planned Features

- **Differentiable SDF rendering** (Vicini et al. 2022): implicit surfaces via sphere tracing, enabling topology changes during optimization.
- **Low-variance biased SDF rendering** (Wang et al. 2024): trading a small amount of bias for a large reduction in variance and implementation complexity.
- **Projective silhouette sampling** (Zhang et al. 2023): projecting primal path segments onto nearby silhouette edges to cut boundary-term variance.
- **Many-Worlds inverse rendering** (Zhang et al. 2025): superposed surface hypotheses for optimization robustness from empty/occluded starting scenes.

## Gradient Validation

### Cornell Box Bunny: Translation Gradients

| Base Render | Perturbed Render | Gradient Heatmap |
| :---: | :---: | :---: |
| <img src="assets/cbox_bunny/cbox_bunny_base.png" width="100%"> | <img src="assets/cbox_bunny/cbox_bunny_pert_x_shift_h_0.01.png" width="100%"> | <img src="assets/cbox_bunny/cbox_bunny_fd_grad_x_shift_h_0.01.png" width="100%"> |

### Cornell Box Bunny: Albedo Gradients

| Base Render | Perturbed Render | Gradient Heatmap |
| :---: | :---: | :---: |
| <img src="assets/cbox_bunny/cbox_bunny_base.png" width="100%"> | <img src="assets/cbox_bunny/cbox_bunny_pert_albedo_red_h_0.1.png" width="100%"> | <img src="assets/cbox_bunny/cbox_bunny_fd_grad_albedo_red_h_0.1.png" width="100%"> |

### Box Cubes: Translation Gradients

| Base Render | Perturbed Render | Gradient Heatmap |
| :---: | :---: | :---: |
| <img src="assets/box_cubes/box_cubes_base.png" width="100%"> | <img src="assets/box_cubes/box_cubes_pert_x_shift_h_0.01.png" width="100%"> | <img src="assets/box_cubes/box_cubes_fd_grad_x_shift_h_0.01.png" width="100%"> |

### Box Cubes: Albedo Gradients

| Base Render | Perturbed Render | Gradient Heatmap |
| :---: | :---: | :---: |
| <img src="assets/box_cubes/box_cubes_base.png" width="100%"> | <img src="assets/box_cubes/box_cubes_pert_h_0.1.png" width="100%"> | <img src="assets/box_cubes/box_cubes_fd_grad_h_0.1.png" width="100%"> |

### Shadow Glossy: Translation Gradients

| Base Render | Perturbed Render | Gradient Heatmap |
| :---: | :---: | :---: |
| <img src="assets/shadow_glossy/shadow_glossy_base.png" width="100%"> | <img src="assets/shadow_glossy/shadow_glossy_pert_mesh_1_x_shift_h_0.01.png" width="100%"> | <img src="assets/shadow_glossy/shadow_glossy_fd_grad_mesh_1_x_shift_h_0.01.png" width="100%"> |

### Teapot: Translation Gradients

| Base Render | Perturbed Render | Gradient Heatmap |
| :---: | :---: | :---: |
| <img src="assets/teapot/teapot_base.png" width="100%"> | <img src="assets/teapot/teapot_pert_x_shift_h_0.01.png" width="100%"> | <img src="assets/teapot/teapot_fd_grad_x_shift_h_0.01.png" width="100%"> |

### Teapot: Roughness Gradients

| Base Render | Perturbed Render | Gradient Heatmap |
| :---: | :---: | :---: |
| <img src="assets/teapot/teapot_base.png" width="100%"> | <img src="assets/teapot/teapot_pert_roughness_h_0.1.png" width="100%"> | <img src="assets/teapot/teapot_fd_grad_roughness_h_0.1.png" width="100%"> |

## Inverse Rendering

### Texture Reconstruction
Optimizing a planet's surface texture from a target render:

| Initial Render | Target Render | Optimization Progress |
| :---: | :---: | :---: |
| <img src="assets/planets/initial_render.png" width="100%"> | <img src="assets/planets/target_render.png" width="100%"> | <img src="assets/planets/render_timelapse.gif" width="100%"> |

| Initial Texture | Target Texture | Texture Timelapse |
| :---: | :---: | :---: |
| <img src="assets/planets/initial_texture.png" width="100%"> | <img src="assets/planets/target_texture.png" width="100%"> | <img src="assets/planets/texture_timelapse.gif" width="100%"> |

### Normal Map Reconstruction
Recovering surface normals from a target render:

| Initial Render | Target Render | Optimization Progress |
| :---: | :---: | :---: |
| <img src="assets/steps/initial_render.png" width="100%"> | <img src="assets/steps/target_render.png" width="100%"> | <img src="assets/steps/render_timelapse.gif" width="100%"> |

| Initial Normal Map | Target Normal Map | Normal Map Timelapse |
| :---: | :---: | :---: |
| <img src="assets/steps/initial_normal.png" width="100%"> | <img src="assets/steps/target_normal.png" width="100%"> | <img src="assets/steps/normal_timelapse.gif" width="100%"> |


## Benchmarks & Performance

We evaluate standard **Automatic Differentiation (AD)** against advanced estimators (**Path Replay Backpropagation (PRB)** and **Radiative Backpropagation (RB)**) on the **Box Cubes** scene (`scenes/box_cubes/box_cubes.xml`) across sample counts (SPP) at 512×512 resolution:

<div align="center">
  <img src="assets/benchmarks/method_comparison_histogram.png" width="98%">
  <p><i>Figure: Performance and VRAM consumption benchmark evaluated on the Box Cubes scene.</i></p>
</div>

- **Memory Efficiency**: Path Replay Backpropagation (PRB) and Radiative Backpropagation (RB) achieve up to a **9.2× VRAM reduction** over standard PyTorch AD at 512×512 resolution (509 MB vs 4,668 MB).
- **Reproducibility**: Run `python tests/benchmark.py` to regenerate `benchmark_results.csv` and the publication figure.


## Architecture

1. **Scene loading**: `scene_parser.py` parses an XML scene into `Scene`, `Camera`, and per-mesh `GeometryData`/`MaterialData`.
2. **Ray generation**: `camera.py` generates (optionally jittered) primary rays from the sensor.
3. **Intersection**: `scene.py` → `mesh.py`/`primitives.py` batch all meshes into one BVH (`bvh.py`) and re-derive intersection attributes differentiably.
4. **Shading**: `bsdf.py` evaluates/samples the appropriate BSDF per hit, dispatched through `CompositeBSDF`.
5. **Integration**: `path.py` accumulates radiance along sampled paths and tone-maps the result.
6. **Differentiation**: `loss.backward()` propagates gradients through every step above to scene parameters.

```
diff_render/
├── scene.py            # Scene graph, mesh batching/finalization
├── scene_parser.py      # Mitsuba-style XML scene loader
├── camera.py            # Perspective camera, ray generation
├── mesh.py              # MeshProxy: editable, differentiable mesh views
├── primitives.py        # Geometry/material data, BVH-integrated intersection
├── bvh.py               # SAH BVH construction, CUDA + PyTorch traversal
├── bsdf.py              # Diffuse, metallic, mirror, dielectric BSDFs
├── path.py              # Autodiff (AD) Monte Carlo path tracer
├── prb.py               # Path Replay Backpropagation (PRB)
├── rb.py                # Radiative Backpropagation (RB)
├── edge.py              # Edge sampling
└── texture.py / ray.py / intersection.py / utils.py
tests/                   # Automated benchmarks & publication plot generators
scenes/                  # XML scene descriptions + assets
```

## Getting Started

```python
import torch
from diff_render import load_scene_from_xml, PathTracer

# Load scene and perspective camera
scene, camera, info = load_scene_from_xml("scenes/teapot/teapot.xml", device="cuda", override_res=512)

# Enable gradients for scene parameters (e.g. albedo, roughness, geometry)
mesh = scene.get_mesh("teapot")
teapot_albedo = mesh.albedo.detach().clone().requires_grad_(True)
mesh.albedo = teapot_albedo

# Initialize path tracer and define target image
integrator = PathTracer(max_depth=5, num_samples=128)
target_img = torch.zeros((512, 512, 3), device="cuda")

# Render image & propagate gradients back to scene parameters
rendered_img = integrator.sample(scene, camera)
loss = ((rendered_img - target_img) ** 2).mean()
loss.backward()

print("Albedo gradient:", teapot_albedo.grad.shape)
```

### Requirements

Install dependencies via `pip install -r requirements.txt`:

- Python 3.11+, PyTorch, NumPy, `trimesh`, Pillow, Pandas, Matplotlib, ImageIO
- `cupy` *(optional; enables CUDA BVH traversal kernel, falls back to pure PyTorch otherwise)*

## References

- [Mitsuba Renderer Documentation](https://www.mitsuba-renderer.org/)

## License

This project is created for educational purposes.