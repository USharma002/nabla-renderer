import os
import time
import gc
import torch
import pandas as pd
import matplotlib.pyplot as plt

from diff_render import load_scene_from_xml, PathTracer, PRBPathTracer, RBPathTracer


def reset_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def get_vram_usage():
    if torch.cuda.is_available():
        return torch.cuda.max_memory_allocated() / (1024 ** 2)  # MB
    return 0.0


def run_benchmark():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    scene_name = "box_cubes"

    scene_path = os.path.join(project_root, f'scenes/{scene_name}/{scene_name}.xml')
    if not os.path.exists(scene_path):
        scene_path = os.path.join(project_root, f'scenes/{scene_name}/scene.xml')

    resolutions = [64, 128, 256, 512]
    spps = [16, 64, 128]

    results = []

    print(f"Starting Benchmarks on device: {device}...")

    # --- Run Benchmarks ---
    for res in resolutions:
        for spp in spps:
            print(f"\nRes: {res}x{res} | SPP: {spp}")
            bench_methods = {
                'AD': PathTracer(max_depth=3, num_samples=spp),
                'PRB': PRBPathTracer(max_depth=3, num_samples=spp),
                'RB': RBPathTracer(max_depth=3, num_samples=spp)
            }
            
            for name, integrator in bench_methods.items():
                reset_memory()
                
                scene, cam, _ = load_scene_from_xml(scene_path, device=device, override_res=res)
                
                # Target something to require grad
                mesh = scene.get_mesh("red_cube")
                mesh_albedo = mesh.albedo.detach().clone().requires_grad_(True)
                mesh.albedo = mesh_albedo
                
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                start_time = time.time()
                
                # Full render + gradient pass
                if name == 'PRB':
                    img = integrator.sample_path(scene, cam, seed=42)
                    dL = torch.ones_like(img)
                    integrator.sample_adjoint(scene, cam, img, dL)
                elif name == 'RB':
                    img = integrator.sample_path(scene, cam, seed=42)
                    dL = torch.ones_like(img)
                    integrator.radiative_backprop(scene, cam, dL)
                else:  # AD
                    torch.manual_seed(42)
                    img = integrator.sample(scene, cam)
                    dL = torch.ones_like(img)
                    (img * dL).sum().backward()
                    
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    
                elapsed_time = time.time() - start_time
                vram = get_vram_usage()
                
                results.append({
                    'Method': name,
                    'Resolution': res,
                    'SPP': spp,
                    'Time (s)': elapsed_time,
                    'VRAM (MB)': vram
                })
                
                print(f"  -> {name:<4}: Time = {elapsed_time:.2f}s, VRAM = {vram:.1f}MB")
                
                # Cleanup for next run to ensure accurate VRAM measurements
                del img, dL, scene, cam, mesh, mesh_albedo, integrator
                reset_memory()

    # --- Save Results to CSV and Display ---
    df = pd.DataFrame(results)

    # Output directories
    out_dir = os.path.join(project_root, "renders", scene_name, "benchmarks")
    tests_out_dir = current_dir
    os.makedirs(out_dir, exist_ok=True)

    csv_path = os.path.join(out_dir, "benchmark_results.csv")
    tests_csv_path = os.path.join(tests_out_dir, "benchmark_results.csv")
    df.to_csv(csv_path, index=False)
    df.to_csv(tests_csv_path, index=False)
    print(f"\nSaved CSV results to:\n - {csv_path}\n - {tests_csv_path}")

    # --- Plotting ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot for the highest resolution
    max_res = resolutions[-1]
    df_res = df[df['Resolution'] == max_res]

    # 1. VRAM plot
    for name in df['Method'].unique():
        subset = df_res[df_res['Method'] == name]
        axes[0].plot(subset['SPP'], subset['VRAM (MB)'], marker='o', label=name)
    axes[0].set_title(f"VRAM Usage vs SPP (Res: {max_res}x{max_res})")
    axes[0].set_xlabel("Samples Per Pixel (SPP)")
    axes[0].set_ylabel("Peak VRAM (MB)")
    axes[0].legend()
    axes[0].grid(True, linestyle='--', alpha=0.7)

    # 2. Time plot
    for name in df['Method'].unique():
        subset = df_res[df_res['Method'] == name]
        axes[1].plot(subset['SPP'], subset['Time (s)'], marker='o', label=name)
    axes[1].set_title(f"Execution Time vs SPP (Res: {max_res}x{max_res})")
    axes[1].set_xlabel("Samples Per Pixel (SPP)")
    axes[1].set_ylabel("Time (s)")
    axes[1].legend()
    axes[1].grid(True, linestyle='--', alpha=0.7)

    plot_path = os.path.join(out_dir, "vram_time_plot.png")
    plt.tight_layout()
    plt.savefig(plot_path, bbox_inches='tight')
    print(f"Saved benchmark plot to: {plot_path}")

    return df


if __name__ == "__main__":
    run_benchmark()
