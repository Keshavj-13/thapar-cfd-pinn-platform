#!/usr/bin/env python3
"""
Thapar CFD Platform - GPU vs OpenFOAM Wall-Clock Performance Benchmark

Runs the SAME physical case -- a 3D lid-driven cavity, Re = 1000, unit cube,
CFL-matched timestep -- on:
  (a) stock OpenFOAM v1912 `icoFoam` (single CPU core, AMD EPYC 7742), and
  (b) the native Thapar CUDA engine (`bench_lid_cavity`, single NVIDIA A100-SXM4)

at a sweep of grid resolutions, and records wall-clock time per solver
timestep for each. This produces the head-to-head speed comparison
referenced in the engineering plan's capability matrix (Section G,
"Hardware Execution": OpenFOAM CPU MPI vs Thapar Native A100 GPU-Resident)
which had not previously been measured -- only numerical accuracy against
OpenFOAM had been validated (openfoam_validation_report.json,
openfoam_nuclear_2d_3d_report.json).

Usage:
    python3 tools/run_openfoam_vs_thapar_benchmark.py

Requires:
    - OpenFOAM v1912 `icoFoam`/`blockMesh` on PATH (apt install openfoam)
    - The CUDA benchmark binary built at
      src/cuda/build/bench_lid_cavity (see src/cuda/tests/bench_lid_cavity.c)
"""
import os
import re
import json
import shutil
import subprocess
import time
import sys

REPO = "/workspace/thapar CFD platform"
SCRATCH = os.environ.get("BENCH_SCRATCH", "/tmp/thapar_openfoam_bench")
CUDA_BENCH_BIN = os.path.join(REPO, "src/cuda/build/bench_lid_cavity")
CUDA_LIBDIR = os.path.join(REPO, "src/cuda/build")

RE = 1000.0
UB = 1.0
L = 1.0
CFL = 0.3
NSTEPS = 50

# Grid resolutions (N x N x N cells) swept for the benchmark.
RESOLUTIONS = [16, 24, 32, 48, 64, 96]

FOAM_ENV = dict(os.environ)
FOAM_ENV["FOAM_ETC"] = "/usr/share/openfoam/etc"
FOAM_ENV["WM_PROJECT_DIR"] = "/usr/share/openfoam"


def foam_header(klass, obj, location):
    return f"""FoamFile
{{
    version     2.0;
    format      ascii;
    class       {klass};
    location    "{location}";
    object      {obj};
}}
"""


def write_case(case_dir, n, dt, nsteps, nu):
    os.makedirs(os.path.join(case_dir, "0"), exist_ok=True)
    os.makedirs(os.path.join(case_dir, "constant"), exist_ok=True)
    os.makedirs(os.path.join(case_dir, "system"), exist_ok=True)

    block_mesh = foam_header("dictionary", "blockMeshDict", "system") + f"""
scale 1.0;

vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 1)
    (1 0 1)
    (1 1 1)
    (0 1 1)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({n} {n} {n}) simpleGrading (1 1 1)
);

edges
(
);

boundary
(
    xMin
    {{
        type wall;
        faces ((0 4 7 3));
    }}
    xMax
    {{
        type wall;
        faces ((1 2 6 5));
    }}
    yMin
    {{
        type wall;
        faces ((0 1 5 4));
    }}
    yMax
    {{
        type wall;
        faces ((2 3 7 6));
    }}
    zMin
    {{
        type wall;
        faces ((0 3 2 1));
    }}
    movingLid
    {{
        type wall;
        faces ((4 5 6 7));
    }}
);

mergePatchPairs
(
);
"""
    with open(os.path.join(case_dir, "system", "blockMeshDict"), "w") as f:
        f.write(block_mesh)

    control_dict = foam_header("dictionary", "controlDict", "system") + f"""
application     icoFoam;
startFrom       startTime;
startTime       0;
stopAt          endTime;
endTime         {nsteps * dt:.10f};
deltaT          {dt:.10f};
writeControl    timeStep;
writeInterval   {nsteps + 1};
purgeWrite      0;
writeFormat     binary;
writePrecision  6;
writeCompression off;
timeFormat      general;
timePrecision   6;
runTimeModifiable false;
adjustTimeStep  no;
"""
    with open(os.path.join(case_dir, "system", "controlDict"), "w") as f:
        f.write(control_dict)

    fv_schemes = foam_header("dictionary", "fvSchemes", "system") + """
ddtSchemes
{
    default         Euler;
}
gradSchemes
{
    default         Gauss linear;
    grad(p)         Gauss linear;
}
divSchemes
{
    default         none;
    div(phi,U)      Gauss linear;
}
laplacianSchemes
{
    default         Gauss linear orthogonal;
}
interpolationSchemes
{
    default         linear;
}
snGradSchemes
{
    default         orthogonal;
}
"""
    with open(os.path.join(case_dir, "system", "fvSchemes"), "w") as f:
        f.write(fv_schemes)

    fv_solution = foam_header("dictionary", "fvSolution", "system") + """
solvers
{
    p
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-06;
        relTol          0.05;
    }
    pFinal
    {
        $p;
        relTol          0;
    }
    U
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-05;
        relTol          0;
    }
}
PISO
{
    nCorrectors     2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}
"""
    with open(os.path.join(case_dir, "system", "fvSolution"), "w") as f:
        f.write(fv_solution)

    transport = foam_header("dictionary", "transportProperties", "constant") + f"""
transportModel  Newtonian;
nu              nu [0 2 -1 0 0 0 0] {nu:.8f};
"""
    with open(os.path.join(case_dir, "constant", "transportProperties"), "w") as f:
        f.write(transport)

    u_field = foam_header("volVectorField", "U", "0") + """
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{
    movingLid
    {
        type            fixedValue;
        value           uniform (1 0 0);
    }
    xMin  { type noSlip; }
    xMax  { type noSlip; }
    yMin  { type noSlip; }
    yMax  { type noSlip; }
    zMin  { type noSlip; }
}
"""
    with open(os.path.join(case_dir, "0", "U"), "w") as f:
        f.write(u_field)

    p_field = foam_header("volScalarField", "p", "0") + """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    movingLid { type zeroGradient; }
    xMin      { type zeroGradient; }
    xMax      { type zeroGradient; }
    yMin      { type zeroGradient; }
    yMax      { type zeroGradient; }
    zMin      { type zeroGradient; }
}
"""
    with open(os.path.join(case_dir, "0", "p"), "w") as f:
        f.write(p_field)


def run(cmd, cwd, log_path):
    env = dict(FOAM_ENV)
    # OpenFOAM's fileName parser rejects paths containing spaces; the parent
    # shell's PWD ("/workspace/thapar CFD platform") would otherwise leak in
    # via the inherited environment even though the child's real cwd (set
    # below, under SCRATCH) has none.
    env["PWD"] = cwd
    with open(log_path, "w") as logf:
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return p.returncode


def bench_openfoam(n):
    dx = L / n
    dt = CFL * dx / UB
    nu = UB * L / RE

    case_dir = os.path.join(SCRATCH, f"cavity_N{n}")
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(case_dir)
    write_case(case_dir, n, dt, NSTEPS, nu)

    rc = run(["blockMesh"], case_dir, os.path.join(case_dir, "blockMesh.log"))
    if rc != 0:
        return {"error": f"blockMesh failed (rc={rc})", "N": n}

    t0 = time.perf_counter()
    rc = run(["icoFoam"], case_dir, os.path.join(case_dir, "icoFoam.log"))
    t1 = time.perf_counter()
    if rc != 0:
        return {"error": f"icoFoam failed (rc={rc})", "N": n}

    wall_s = t1 - t0
    log_text = open(os.path.join(case_dir, "icoFoam.log")).read()
    exec_times = re.findall(r"ExecutionTime\s*=\s*([0-9.eE+-]+)\s*s", log_text)
    clock_times = re.findall(r"ClockTime\s*=\s*([0-9.eE+-]+)\s*s", log_text)
    foam_exec_time_s = float(exec_times[-1]) if exec_times else None
    foam_clock_time_s = float(clock_times[-1]) if clock_times else None

    cells = n * n * n
    return {
        "N": n,
        "cells": cells,
        "dt": dt,
        "nsteps": NSTEPS,
        "wall_time_s": wall_s,
        "ms_per_step": (wall_s * 1000.0) / NSTEPS,
        "foam_reported_execution_time_s": foam_exec_time_s,
        "foam_reported_clock_time_s": foam_clock_time_s,
    }


def bench_thapar(n):
    dx = L / n
    dt = CFL * dx / UB

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = CUDA_LIBDIR + ":" + env.get("LD_LIBRARY_PATH", "")
    p = subprocess.run(
        [CUDA_BENCH_BIN, str(n), str(NSTEPS), f"{dt:.10f}", str(RE)],
        capture_output=True, text=True, env=env,
    )
    if p.returncode != 0:
        return {"error": f"bench_lid_cavity failed (rc={p.returncode}): {p.stderr}", "N": n}

    m = re.search(
        r"BENCH_RESULT N=(\d+) cells=(\d+) nsteps=(\d+) total_ms=([0-9.]+) "
        r"ms_per_step=([0-9.]+) steps_per_sec=([0-9.]+) vram_mb=([0-9.]+) max_div=([0-9.eE+-]+)",
        p.stdout,
    )
    if not m:
        return {"error": f"could not parse output: {p.stdout}\n{p.stderr}", "N": n}

    return {
        "N": int(m.group(1)),
        "cells": int(m.group(2)),
        "dt": dt,
        "nsteps": int(m.group(3)),
        "total_ms": float(m.group(4)),
        "ms_per_step": float(m.group(5)),
        "steps_per_sec": float(m.group(6)),
        "vram_mb": float(m.group(7)),
        "max_div": float(m.group(8)),
    }


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    results = {"config": {
        "Re": RE, "Ub": UB, "L": L, "CFL": CFL, "nsteps": NSTEPS,
        "cpu": "AMD EPYC 7742 (single core, single icoFoam process)",
        "gpu": "NVIDIA A100-SXM4-80GB (single device)",
        "openfoam_version": "v1912 (OpenFOAM Foundation)",
        "case": "3D lid-driven cavity, unit cube, 5x no-slip wall + 1x moving lid",
    }, "runs": []}

    for n in RESOLUTIONS:
        print(f"\n=== N={n} ({n**3} cells) ===", flush=True)
        print("  -> OpenFOAM icoFoam (1 core)...", flush=True)
        of = bench_openfoam(n)
        print(f"     {of}", flush=True)
        print("  -> Thapar CUDA engine (A100)...", flush=True)
        th = bench_thapar(n)
        print(f"     {th}", flush=True)

        entry = {"N": n, "cells": n ** 3, "openfoam": of, "thapar_gpu": th}
        if "error" not in of and "error" not in th:
            entry["speedup_x"] = of["ms_per_step"] / th["ms_per_step"]
        results["runs"].append(entry)

        # persist incrementally so partial progress isn't lost
        out_path = os.path.join(REPO, "openfoam_vs_thapar_performance_report.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print("\nDone. Report written to openfoam_vs_thapar_performance_report.json")


if __name__ == "__main__":
    main()
