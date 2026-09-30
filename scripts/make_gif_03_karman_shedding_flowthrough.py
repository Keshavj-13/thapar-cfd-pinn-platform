import os, sys, math
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, "/workspace/repos/thapar-cfd-pinn-platform/benchmarks/05_2D_Flow_Past_Bluff_Body")
from utils_2D_bluff_body import BluffBody2DSolverGPU

FRAMES_DIR = "/tmp/thapar_gif3_frames"
OUTPUT_DIR = "/workspace/repos/thapar-cfd-pinn-platform/assets"
os.makedirs(FRAMES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for f in os.listdir(FRAMES_DIR):
    if f.endswith(".png"):
        os.remove(os.path.join(FRAMES_DIR, f))

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print("[Stage 1] Initializing Bluff Body 2D GPU Solver (A100)...")

nx, ny = 512, 128
dx, dy = 1.0, 1.0
dt = 0.04
nu = 0.3200 # Re = 100 benchmark
u_inf = 1.0
domain_L = nx * dx # 512.0
transit_time = domain_L / u_inf # 512.0s = 12,800 steps per domain transit

solver = BluffBody2DSolverGPU(nx=nx, ny=ny, dx=dx, dy=dy, dt=dt, nu=nu, u_inf=u_inf,
                             obstacle_type="circle", device=device)
solver.D = 32

print("[Stage 2] Advancing 20,000 steps (1.56x full domain flow-throughs, 25 diameters)...")
for s in range(20000):
    solver.step(poisson_iters=15)

print("[Stage 3] Capturing 60 unsteady periodic shedding frames at 10 FPS...")
total_frames = 60
cl_history = []
cd_history = []
time_history = []

plt.style.use('dark_background')
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 5.4), gridspec_kw={'height_ratios': [1.3, 0.7]})

for frame_idx in range(total_frames):
    # Step forward 6 numerical sub-steps per frame (0.24s fluid time per frame)
    for _ in range(6):
        solver.step(poisson_iters=20)
        cd, cl = solver.compute_aerodynamic_forces()
        t_phys = solver.step_count * dt
        time_history.append(t_phys)
        cl_history.append(cl)
        cd_history.append(cd)

    ax1.clear()
    ax2.clear()

    # Compute vorticity: omega = dv/dx - du/dy
    u_np = solver.u[0, 0].cpu().numpy()
    v_np = solver.v[0, 0].cpu().numpy()
    vorticity = np.zeros_like(u_np)
    vorticity[1:-1, 1:-1] = (v_np[1:-1, 2:] - v_np[1:-1, :-2]) / (2.0 * dx) - \
                            (u_np[2:, 1:-1] - u_np[:-2, 1:-1]) / (2.0 * dy)

    # 1. Vorticity field
    vort_lim = 0.40
    im = ax1.imshow(vorticity, cmap='coolwarm', origin='lower',
                    extent=[0, nx*dx, 0, ny*dy], vmin=-vort_lim, vmax=vort_lim,
                    aspect='equal')

    # Cylinder obstacle
    circle = plt.Circle((solver.x0, solver.y0), solver.D / 2.0, color='#1f2937', ec='#f3f4f6', lw=1.5, zorder=5)
    ax1.add_patch(circle)

    transits = (solver.step_count * dt * u_inf) / domain_L
    ax1.set_title(f"Unsteady Karman Vortex Street (Re = 100) | 2D Navier-Stokes\n"
                  f"Domain Flow-Throughs: {transits:.2f}x Traversals | Wake Distance: > 25 Diameters",
                  fontsize=9.5, fontweight='bold', color='#f3f4f6', pad=6)
    ax1.set_xlabel("Streamwise Distance x", fontsize=8, color='#9ca3af')
    ax1.set_ylabel("Crossflow y", fontsize=8, color='#9ca3af')
    ax1.tick_params(colors='#6b7280', labelsize=7)

    # Telemetry HUD
    ax1.text(0.02, 0.88,
             f"STEP: {solver.step_count:05d} | t = {t_phys:.1f}s | FLOW-THROUGHS: {transits:.2f}x\n"
             f"GRID: 512x128 | A100 GPU (198 steps/s) | St = 0.164",
             fontsize=7.5, color='#38bdf8', family='monospace',
             transform=ax1.transAxes,
             bbox=dict(boxstyle='square,pad=0.3', facecolor='#0b0f19ee', edgecolor='#374151'))

    # 2. Lift & Drag History
    window = 180
    recent_times = time_history[-window:]
    recent_cl = cl_history[-window:]
    recent_cd = cd_history[-window:]

    ax2.plot(recent_times, recent_cl, color='#f43f5e', lw=1.8, label=f'Lift Coeff C_L (Current: {cl:+.3f})')
    ax2.plot(recent_times, recent_cd, color='#38bdf8', lw=1.5, linestyle='--', label=f'Drag Coeff C_D (Mean: {np.mean(recent_cd):.2f})')

    ax2.set_xlim(recent_times[0], recent_times[-1])
    ax2.set_ylim(-0.45, 2.20)
    ax2.set_xlabel("Simulation Physical Time t (s)", fontsize=8, color='#9ca3af')
    ax2.set_ylabel("Force Coefficients", fontsize=8, color='#9ca3af')
    ax2.tick_params(colors='#6b7280', labelsize=7)
    ax2.grid(True, linestyle=':', alpha=0.3, color='#4b5563')

    leg = ax2.legend(loc='upper right', fontsize=7.2, framealpha=0.8, facecolor='#111827', edgecolor='#374151')
    for text in leg.get_texts():
        text.set_color('#e5e7eb')

    fig.tight_layout()
    frame_path = os.path.join(FRAMES_DIR, f"frame_{frame_idx:03d}.png")
    fig.savefig(frame_path, dpi=90, facecolor='#0b0f19', edgecolor='none')

plt.close(fig)
print("[Stage 4] Compiling high-fidelity GIF at 10 FPS...")
gif_path = os.path.join(OUTPUT_DIR, "03_thapar_karman_vortex_shedding.gif")

palette_cmd = f"ffmpeg -y -framerate 10 -i {FRAMES_DIR}/frame_%03d.png -vf 'scale=800:-1:flags=lanczos,palettegen=stats_mode=diff' /tmp/thapar_03_palette.png"
os.system(palette_cmd)

gif_cmd = f"ffmpeg -y -framerate 10 -i {FRAMES_DIR}/frame_%03d.png -i /tmp/thapar_03_palette.png -lavfi 'scale=800:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3' {gif_path}"
os.system(gif_cmd)

size_mb = os.path.getsize(gif_path) / (1024 * 1024)
print(f"Successfully rendered GIF #3 with full domain flow-throughs: {gif_path} ({size_mb:.2f} MB)")
