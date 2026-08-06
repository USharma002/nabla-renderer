import os
import torch
import matplotlib.pyplot as plt

from diff_render import load_scene_from_xml, PRBPathTracer

# Determine device and scene file path
device = 'cuda' if torch.cuda.is_available() else 'cpu'
scene_path = os.path.join(os.path.dirname(__file__), "scenes", "box_cubes", "box_cubes.xml")
if not os.path.exists(scene_path):
    scene_path = os.path.join(os.path.dirname(__file__), "scenes", "box_cubes", "scene.xml")

# Load scene & camera
scene, cam, _ = load_scene_from_xml(scene_path, device=device)

# Enable gradients for specific mesh properties
red_cube = scene.get_mesh("red_cube")
red_cube_albedo = red_cube.albedo.detach().clone().requires_grad_(True)
red_cube.albedo = red_cube_albedo

wall = scene.get_mesh("wall")
wall_roughness = wall.roughness.detach().clone().requires_grad_(True)
wall.roughness = wall_roughness

# Initialize PRB Integrator
integrator = PRBPathTracer(max_depth=3, num_samples=16)

# Primal Pass: Render image
print("Rendering primal image...")
img = integrator.sample_path(scene, cam, seed=42)

# Adjoint Pass: Compute loss gradient & perform Path Replay Backpropagation
target = torch.zeros_like(img)
dL = 2.0 * (img - target)

print("Running PRB adjoint pass...")
integrator.sample_adjoint(scene, cam, img, dL)

# Check accumulated gradients
if red_cube_albedo.grad is not None:
    print(f"Red Cube Albedo gradient shape: {red_cube_albedo.grad.shape}")
    print(f"Red Cube Albedo gradient mean magnitude: {red_cube_albedo.grad.abs().mean().item():.6f}")

if wall_roughness.grad is not None:
    print(f"Wall Roughness gradient shape: {wall_roughness.grad.shape}")
    print(f"Wall Roughness gradient mean magnitude: {wall_roughness.grad.abs().mean().item():.6f}")

plt.figure(figsize=(6, 6))
plt.imshow(img.detach().cpu().numpy().clip(0, 1))
plt.title("PRB Rendered Image")
plt.axis("off")
plt.savefig("test_output.png")
print("Saved render to test_output.png")