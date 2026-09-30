import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def generate_speedup_gif():
    out_dir = "/tmp/thapar_frames_05"
    os.makedirs(out_dir, exist_ok=True)
    gif_path = "/workspace/repos/thapar-cfd-pinn-platform/assets/05_thapar_openfoam_gpu_speedup.gif"

    report_path = "/workspace/repos/thapar-cfd-pinn-platform/openfoam_vs_thapar_performance_report.json"
    with open(report_path) as f:
        data = json.load(f)

    runs = data["runs"]
    cells = np.array([r["cells"] for r in runs])
    labels = [f"{r['N']}^3 ({r['cells']:,})" for r in runs]
    of_ms = np.array([r["openfoam"]["ms_per_step"] for r in runs])
    of_wall = np.array([r["openfoam"]["wall_time_s"] for r in runs])
    gpu_ms = np.array([r["thapar_gpu"]["ms_per_step"] for r in runs])
    gpu_wall = np.array([r["thapar_gpu"]["total_ms"] / 1000.0 for r in runs])
    speedups = np.array([r["speedup_x"] for r in runs])
    vrams = np.array([r["thapar_gpu"]["vram_mb"] for r in runs])

    n_frames = 50
    frames = []

    print("Generating OpenFOAM vs Thapar GPU benchmark frames...")

    for f_idx in range(n_frames):
        fig = plt.figure(figsize=(10.0, 5.0), dpi=100)
        fig.patch.set_facecolor('#0b0f19')

        gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1.0], height_ratios=[1.0, 1.0], wspace=0.28, hspace=0.38)
        ax_bars = fig.add_subplot(gs[:, 0])
        ax_log = fig.add_subplot(gs[0, 1])
        ax_speedup = fig.add_subplot(gs[1, 1])

        for ax in [ax_bars, ax_log, ax_speedup]:
            ax.set_facecolor('#111827')
            for spine in ax.spines.values():
                spine.set_color('#374151')
            ax.tick_params(colors='#9ca3af', labelsize=8)

        # Fraction of animation progress
        progress = (f_idx + 1) / n_frames
        active_idx = int(np.clip(progress * (len(runs) + 1) - 0.5, 0, len(runs) - 1))
        
        # 1. Left Panel: Wall-clock Execution Comparison Bars
        y_pos = np.arange(len(runs))
        ax_bars.set_yticks(y_pos)
        ax_bars.set_yticklabels(labels, color='#e5e7eb', fontsize=8.5)
        ax_bars.invert_yaxis()

        # Partial bar scaling up to active index
        bar_scale = np.zeros(len(runs))
        for i in range(len(runs)):
            if i < active_idx:
                bar_scale[i] = 1.0
            elif i == active_idx:
                local_p = (progress * (len(runs) + 1) - 0.5) - active_idx
                bar_scale[i] = np.clip(local_p, 0.0, 1.0)
            else:
                bar_scale[i] = 0.0

        # Draw OpenFOAM bar (log wall time)
        of_vals = of_wall * bar_scale
        gpu_vals = gpu_wall * bar_scale

        b1 = ax_bars.barh(y_pos - 0.18, np.log10(np.maximum(0.01, of_vals)), height=0.32, color='#e11d48', alpha=0.85, label='OpenFOAM v1912 (1x AMD EPYC CPU)')
        b2 = ax_bars.barh(y_pos + 0.18, np.log10(np.maximum(0.01, gpu_vals)), height=0.32, color='#0ea5e9', alpha=0.9, label='Thapar CFD Native (1x NVIDIA A100)')

        # Custom ticks for log10 time
        ax_bars.set_xlim(-2.0, 2.7)
        ax_bars.set_xticks([-2, -1, 0, 1, 2])
        ax_bars.set_xticklabels(['0.01s', '0.1s', '1.0s', '10s', '100s'], color='#9ca3af', fontsize=8)
        ax_bars.set_xlabel("Wall-Clock Time for 50 Timesteps (Log Scale)", color='#9ca3af', fontsize=8.5)
        ax_bars.set_title("Benchmark: 3D Lid-Driven Cavity ($Re=1000$)", color='#f3f4f6', fontsize=9.5, fontweight='bold', pad=8)
        ax_bars.grid(True, linestyle='--', alpha=0.2, color='#4b5563', axis='x')

        # Annotations on bars
        for i in range(len(runs)):
            if bar_scale[i] > 0.1:
                cur_sp = speedups[i] * bar_scale[i]
                ax_bars.text(1.4, i, f"{speedups[i]:.1f}x", color='#38bdf8', fontweight='bold', fontsize=8.5, va='center')
                ax_bars.text(-1.9, i - 0.18, f"{of_wall[i]:.2f}s", color='#fca5a5', fontsize=7, va='center')
                ax_bars.text(-1.9, i + 0.18, f"{gpu_wall[i]*1000:.1f}ms", color='#bae6fd', fontsize=7, va='center')

        leg_bars = ax_bars.legend(loc='lower left', fontsize=7.5, facecolor='#1f2937', edgecolor='#374151')
        for t in leg_bars.get_texts():
            t.set_color('#e5e7eb')

        # Telemetry badge
        cur_cells = cells[active_idx]
        cur_speedup = speedups[active_idx]
        cur_vram = vrams[active_idx]
        ax_bars.text(0.03, 0.03,
                     f"CURRENT MESH: {cur_cells:,} CELLS\n"
                     f"A100 VRAM ALLOC: {cur_vram:.1f} MB\n"
                     f"PEAK ACCELERATION: {cur_speedup:.1f}x",
                     color='#10b981' if progress > 0.8 else '#38bdf8',
                     fontsize=8, family='monospace', transform=ax_bars.transAxes,
                     bbox=dict(boxstyle='square,pad=0.3', facecolor='#000000dd', edgecolor='#374151'))

        # 2. Right Top: Log-Log Runtime (ms/step)
        sub_n = min(len(cells), int(np.ceil(progress * len(cells))))
        c_sub = cells[:sub_n]
        of_sub = of_ms[:sub_n]
        gpu_sub = gpu_ms[:sub_n]

        ax_log.loglog(cells, of_ms, 'o--', color='#e11d48', alpha=0.3, markersize=4)
        ax_log.loglog(cells, gpu_ms, 's--', color='#0ea5e9', alpha=0.3, markersize=4)
        if sub_n > 0:
            ax_log.loglog(c_sub, of_sub, 'o-', color='#e11d48', linewidth=2, markersize=6, label='OpenFOAM CPU')
            ax_log.loglog(c_sub, gpu_sub, 's-', color='#0ea5e9', linewidth=2, markersize=6, label='Thapar A100 GPU')

        ax_log.set_xlim(3e3, 1.2e6)
        ax_log.set_ylim(1.0, 1e4)
        ax_log.set_xlabel("Mesh Cells", color='#9ca3af', fontsize=8)
        ax_log.set_ylabel("ms / timestep", color='#9ca3af', fontsize=8)
        ax_log.set_title("Step Latency Scaling ($16^3 \\to 96^3$)", color='#f3f4f6', fontsize=9, fontweight='bold')
        ax_log.grid(True, which="both", linestyle='--', alpha=0.25, color='#4b5563')
        leg_log = ax_log.legend(loc='upper left', fontsize=7.5, facecolor='#1f2937', edgecolor='#374151')
        for t in leg_log.get_texts():
            t.set_color('#e5e7eb')

        # 3. Right Bottom: Speedup Factor Curve
        sp_sub = speedups[:sub_n]
        ax_speedup.semilogx(cells, speedups, 'k--', color='#4b5563', linewidth=1)
        if sub_n > 0:
            ax_speedup.semilogx(c_sub, sp_sub, 'D-', color='#a855f7', linewidth=2.2, markersize=6, label=f'Speedup Multiplier ({speedups[sub_n-1]:.1f}x)')

        ax_speedup.set_xlim(3e3, 1.2e6)
        ax_speedup.set_ylim(0, 750)
        ax_speedup.set_xlabel("Mesh Cells", color='#9ca3af', fontsize=8)
        ax_speedup.set_ylabel("Speedup Ratio (x)", color='#9ca3af', fontsize=8)
        ax_speedup.set_title(f"A100 GPU vs Single EPYC Core Speedup", color='#f3f4f6', fontsize=9, fontweight='bold')
        ax_speedup.grid(True, which="both", linestyle='--', alpha=0.25, color='#4b5563')
        leg_sp = ax_speedup.legend(loc='upper left', fontsize=7.5, facecolor='#1f2937', edgecolor='#374151')
        for t in leg_sp.get_texts():
            t.set_color('#e5e7eb')

        fig.subplots_adjust(left=0.07, right=0.96, top=0.91, bottom=0.12)
        frame_file = f"{out_dir}/frame_{f_idx:03d}.png"
        fig.savefig(frame_file, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        frames.append(frame_file)

    # Dwell on final state (18 frames @ 10 FPS = 1.8s dwell)
    for _ in range(18):
        frames.append(frames[-1])

    print(f"Generated {len(frames)} frames. Encoding GIF...")
    palette_cmd = f"ffmpeg -y -framerate 10 -i {out_dir}/frame_%03d.png -vf 'scale=800:-1:flags=lanczos,palettegen=stats_mode=diff' /tmp/thapar_05_palette.png"
    os.system(palette_cmd)

    gif_cmd = f"ffmpeg -y -framerate 10 -i {out_dir}/frame_%03d.png -i /tmp/thapar_05_palette.png -lavfi 'scale=800:-1:flags=lanczos [x]; [x][1:v] paletteuse=dither=bayer:bayer_scale=3' {gif_path}"
    os.system(gif_cmd)

    size_mb = os.path.getsize(gif_path) / (1024 * 1024)
    print(f"Successfully created GIF #5: {gif_path} ({size_mb:.2f} MB)")

if __name__ == '__main__':
    generate_speedup_gif()
