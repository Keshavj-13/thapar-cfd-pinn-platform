#!/usr/bin/env python3
"""
Thapar CFD Platform - OpenFOAM vs Native GPU Engine Performance Comparison Plot
Reads openfoam_vs_thapar_performance_report.json (produced by
tools/run_openfoam_vs_thapar_benchmark.py, which runs the SAME 3D
lid-driven cavity Re=1000 case on stock OpenFOAM v1912 icoFoam,
single AMD EPYC 7742 core, and the native Thapar CUDA engine on a
single NVIDIA A100-SXM4) and renders the head-to-head wall-clock
speed comparison referenced in the engineering plan's capability
matrix (Section G) that had not previously been measured.
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = "/workspace/thapar CFD platform"
REPORT_PATH = os.path.join(REPO, "openfoam_vs_thapar_performance_report.json")
OUT_DIRS = [os.path.join(REPO, "validation_plots")]
for d in OUT_DIRS:
    os.makedirs(d, exist_ok=True)

plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['legend.fontsize'] = 9

COLOR_OF = '#e11d48'
COLOR_GPU = '#0ea5e9'
COLOR_SPEEDUP = '#059669'


def save_fig(fig, filename):
    for d in OUT_DIRS:
        path = os.path.join(d, filename)
        fig.savefig(path, dpi=300, bbox_inches='tight')
    print(f"[Plot Generator] Saved: {filename}")


def main():
    with open(REPORT_PATH) as f:
        report = json.load(f)

    runs = [r for r in report["runs"] if "error" not in r.get("openfoam", {}) and "error" not in r.get("thapar_gpu", {})]
    cells = np.array([r["cells"] for r in runs])
    of_ms = np.array([r["openfoam"]["ms_per_step"] for r in runs])
    gpu_ms = np.array([r["thapar_gpu"]["ms_per_step"] for r in runs])
    speedup = np.array([r["speedup_x"] for r in runs])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.2))

    # --- Panel 1: raw wall-clock cost per solver timestep ---
    ax1.loglog(cells, of_ms, 'o-', color=COLOR_OF, linewidth=2, markersize=7,
               label='OpenFOAM v1912 icoFoam\n(1x AMD EPYC 7742 core)')
    ax1.loglog(cells, gpu_ms, 's-', color=COLOR_GPU, linewidth=2, markersize=7,
               label='Thapar CUDA Engine\n(1x NVIDIA A100-SXM4)')
    ax1.set_xlabel('Grid Resolution (cells, $N^3$)')
    ax1.set_ylabel('Wall-Clock Time per Solver Timestep (ms)')
    ax1.set_title('Identical 3D Lid-Driven Cavity Case (Re = 1000)\nPer-Timestep Wall-Clock Cost')
    ax1.legend(loc='upper left')
    ax1.grid(True, which='both', ls='--', alpha=0.5)
    for x, y, r in zip(cells, of_ms, runs):
        ax1.annotate(f'{r["N"]}³', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8, color=COLOR_OF)

    # --- Panel 2: speedup factor vs problem size ---
    ax2.plot(cells, speedup, 'D-', color=COLOR_SPEEDUP, linewidth=2.5, markersize=8)
    ax2.set_xscale('log')
    ax2.set_xlabel('Grid Resolution (cells, $N^3$)')
    ax2.set_ylabel('Speedup Factor ($\\times$ vs single-core OpenFOAM)')
    ax2.set_title('GPU Speedup Grows With Problem Size\n(A100-resident stencils amortize kernel-launch overhead)')
    ax2.grid(True, which='both', ls='--', alpha=0.5)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d×'))
    for x, y, r in zip(cells, speedup, runs):
        ax2.annotate(f'{y:.1f}×\n({r["N"]}³ cells)', (x, y), textcoords="offset points",
                     xytext=(0, 10), ha='center', fontsize=8, fontweight='bold', color=COLOR_SPEEDUP)

    fig.suptitle(
        'Thapar CFD Platform vs OpenFOAM: Measured Wall-Clock Performance\n'
        f'Single A100-SXM4 GPU vs Single EPYC 7742 CPU Core — Same Case, Same Timestep, Same Mesh',
        fontsize=13, fontweight='bold', y=1.04)
    fig.text(0.5, -0.03,
              'Methodology: identical unit-cube 3D lid-driven cavity (5x no-slip walls + 1x moving lid, Re=1000, CFL=0.3-matched dt), 50 timesteps per resolution.\n'
              'OpenFOAM: PISO (2 correctors), PCG/DIC pressure solve, single core, no MPI decomposition. Thapar: eager dispatch (CUDA-graph capture path disabled/unverified, see note).',
              ha='center', fontsize=8, color='#475569', style='italic')

    fig.tight_layout()
    save_fig(fig, "plot9_openfoam_vs_thapar_gpu_speedup.png")
    plt.close(fig)

    # Console summary table
    print("\n=== OpenFOAM (1-core CPU) vs Thapar CUDA Engine (1x A100) ===")
    print(f"{'N':>5} {'cells':>10} {'OF ms/step':>14} {'GPU ms/step':>14} {'speedup':>10}")
    for r in runs:
        print(f"{r['N']:>5} {r['cells']:>10} {r['openfoam']['ms_per_step']:>14.3f} "
              f"{r['thapar_gpu']['ms_per_step']:>14.4f} {r['speedup_x']:>9.1f}x")


if __name__ == "__main__":
    main()
