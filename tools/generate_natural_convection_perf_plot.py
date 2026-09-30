#!/usr/bin/env python3
"""
Thapar CFD Platform - OpenFOAM vs Native GPU Engine: Natural Convection
(Multiphysics: Momentum + Boussinesq-Coupled Heat Transport) Performance Plot.
Companion to generate_performance_comparison_plot.py (plain incompressible
lid-driven cavity) -- reads
openfoam_vs_thapar_natural_convection_report.json (produced by
tools/run_openfoam_natural_convection_benchmark.py, which runs the SAME
differentially-heated cavity, Ra=10^5, Pr=0.71, de Vahl Davis 1983 case
already accuracy-validated in tests/test_natural_convection.cu) on stock
OpenFOAM v1912 buoyantBoussinesqPimpleFoam (1 CPU core) vs the native
Thapar CUDA engine's fused UCOF momentum+heat predictor (1x A100).
"""
import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = "/workspace/thapar CFD platform"
REPORT_PATH = os.path.join(REPO, "openfoam_vs_thapar_natural_convection_report.json")
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
COLOR_SPEEDUP = '#7c3aed'


def save_fig(fig, filename):
    for d in OUT_DIRS:
        fig.savefig(os.path.join(d, filename), dpi=300, bbox_inches='tight')
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

    ax1.loglog(cells, of_ms, 'o-', color=COLOR_OF, linewidth=2, markersize=7,
               label='OpenFOAM v1912\nbuoyantBoussinesqPimpleFoam\n(1x AMD EPYC 7742 core)')
    ax1.loglog(cells, gpu_ms, 's-', color=COLOR_GPU, linewidth=2, markersize=7,
               label='Thapar CUDA Engine\nfused UCOF momentum+heat\n(1x NVIDIA A100-SXM4)')
    ax1.set_xlabel('Grid Resolution (cells, $N^2 \\times 4$)')
    ax1.set_ylabel('Wall-Clock Time per Solver Timestep (ms)')
    ax1.set_title('Differentially Heated Cavity, Ra=$10^5$, Pr=0.71\n(de Vahl Davis 1983) — Per-Timestep Cost')
    ax1.legend(loc='upper left', fontsize=8)
    ax1.grid(True, which='both', ls='--', alpha=0.5)
    for x, y, r in zip(cells, of_ms, runs):
        ax1.annotate(f'{r["N"]}²×4', (x, y), textcoords="offset points", xytext=(0, 8), ha='center', fontsize=8, color=COLOR_OF)

    ax2.plot(cells, speedup, 'D-', color=COLOR_SPEEDUP, linewidth=2.5, markersize=8)
    ax2.set_xscale('log')
    ax2.set_xlabel('Grid Resolution (cells, $N^2 \\times 4$)')
    ax2.set_ylabel('Speedup Factor ($\\times$ vs single-core OpenFOAM)')
    ax2.set_title('Multiphysics (Momentum+Heat) Speedup\nSmaller cell counts than the lid-cavity sweep — thin pseudo-2D slab (nz=4)')
    ax2.grid(True, which='both', ls='--', alpha=0.5)
    ax2.yaxis.set_major_formatter(mticker.FormatStrFormatter('%d×'))
    for x, y, r in zip(cells, speedup, runs):
        ax2.annotate(f'{y:.1f}×', (x, y), textcoords="offset points", xytext=(0, 10), ha='center', fontsize=8, fontweight='bold', color=COLOR_SPEEDUP)

    fig.suptitle(
        'Thapar CFD Platform vs OpenFOAM: Multiphysics (Momentum + Boussinesq Heat) Performance\n'
        'Single A100-SXM4 GPU vs Single EPYC 7742 CPU Core — Same Case, Same Timestep, Same Mesh',
        fontsize=13, fontweight='bold', y=1.05)
    fig.text(0.5, -0.05,
              'Methodology: identical differentially-heated square cavity (hot/cold isothermal walls, adiabatic top/bottom, free-slip front/back), Ra=10^5, Pr=0.71, 50 timesteps per resolution.\n'
              'OpenFOAM: PIMPLE (2 correctors), PCG/DIC p_rgh solve, single core, no MPI. Thapar: enable_heat=1 Boussinesq buoyancy, eager dispatch (CUDA-graph path disabled/unverified).\n'
              'Grid is intentionally a thin nz=4 pseudo-2D slab (matching tests/test_natural_convection.cu\'s already-validated Nu_avg=4.519 config), so absolute cell counts and hence GPU\n'
              'occupancy stay far lower than the cubic lid-cavity sweep in plot9 — this is why speedup here (~5-16x) is more modest than the 3.6x-701x range there, not a contradiction.',
              ha='center', fontsize=7.5, color='#475569', style='italic')

    fig.tight_layout()
    save_fig(fig, "plot10_openfoam_vs_thapar_natural_convection_speedup.png")
    plt.close(fig)

    print("\n=== OpenFOAM (1-core CPU) vs Thapar CUDA Engine (1x A100) -- Natural Convection ===")
    print(f"{'N':>5} {'cells':>8} {'OF ms/step':>12} {'GPU ms/step':>13} {'speedup':>9}")
    for r in runs:
        print(f"{r['N']:>5} {r['cells']:>8} {r['openfoam']['ms_per_step']:>12.3f} "
              f"{r['thapar_gpu']['ms_per_step']:>13.4f} {r['speedup_x']:>8.1f}x")


if __name__ == "__main__":
    main()
