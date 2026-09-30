#!/usr/bin/env python3
"""
Thapar CFD Platform - GPU vs OpenFOAM Natural Convection (Multiphysics) Benchmark

Companion to run_openfoam_vs_thapar_benchmark.py (which covers the plain
incompressible lid-driven cavity). This one runs the SAME differentially
heated square cavity, Ra = 10^5, Pr = 0.71 (de Vahl Davis 1983 -- the exact
case already accuracy-validated in tests/test_natural_convection.cu against
Nu_avg = 4.519 +/- 2.0%) on:
  (a) OpenFOAM v1912 `buoyantBoussinesqPimpleFoam` (1 CPU core), and
  (b) the native Thapar CUDA engine with enable_heat=1 / Boussinesq buoyancy
      (`bench_natural_convection`, 1x A100)

This is the multiphysics counterpart to the lid-cavity sweep: it exercises
the UCOF fused momentum+heat predictor kernel rather than momentum alone.
"""
import os
import re
import json
import shutil
import subprocess
import time

REPO = "/workspace/thapar CFD platform"
SCRATCH = os.environ.get("BENCH_SCRATCH", "/tmp/thapar_openfoam_nc_bench")
CUDA_BENCH_BIN = os.path.join(REPO, "src/cuda/build/bench_natural_convection")
CUDA_LIBDIR = os.path.join(REPO, "src/cuda/build")

NSTEPS = 50
RESOLUTIONS = [16, 24, 32, 48, 64]

# Dimensionless Ra=10^5/Pr=0.71 parameters, identical to tests/test_natural_convection.cu
NU = 0.01
ALPHA = NU / 0.71
BETA = (1e5 * NU * ALPHA) / (10.0 * 1.0)
T_HOT = 1.0
T_COLD = 0.0
T_REF = 0.5
GY = -10.0

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


def write_case(case_dir, n, dt, nsteps):
    nz = 4
    for sub in ("0", "constant", "system"):
        os.makedirs(os.path.join(case_dir, sub), exist_ok=True)

    block_mesh = foam_header("dictionary", "blockMeshDict", "system") + f"""
scale 1.0;

vertices
(
    (0 0 0)
    (1 0 0)
    (1 1 0)
    (0 1 0)
    (0 0 0.125)
    (1 0 0.125)
    (1 1 0.125)
    (0 1 0.125)
);

blocks
(
    hex (0 1 2 3 4 5 6 7) ({n} {n} {nz}) simpleGrading (1 1 1)
);

boundary
(
    hotWall   {{ type wall; faces ((0 4 7 3)); }}
    coldWall  {{ type wall; faces ((1 2 6 5)); }}
    bottomWall {{ type wall; faces ((0 1 5 4)); }}
    topWall   {{ type wall; faces ((2 3 7 6)); }}
    frontAndBack {{ type empty; faces ((0 3 2 1) (4 5 6 7)); }}
);
"""
    with open(os.path.join(case_dir, "system", "blockMeshDict"), "w") as f:
        f.write(block_mesh)

    control_dict = foam_header("dictionary", "controlDict", "system") + f"""
application     buoyantBoussinesqPimpleFoam;
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
ddtSchemes { default Euler; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default          none;
    div(phi,U)       Gauss linear;
    div(phi,T)       Gauss linear;
    div(phi,K)       Gauss linear;
    div(phi,Ekp)     Gauss linear;
    div((nuEff*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear orthogonal; }
interpolationSchemes { default linear; }
snGradSchemes { default orthogonal; }
"""
    with open(os.path.join(case_dir, "system", "fvSchemes"), "w") as f:
        f.write(fv_schemes)

    fv_solution = foam_header("dictionary", "fvSolution", "system") + """
solvers
{
    "p_rgh.*"
    {
        solver          PCG;
        preconditioner  DIC;
        tolerance       1e-06;
        relTol          0.05;
    }
    "(U|T).*"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-06;
        relTol          0;
    }
}
PIMPLE
{
    momentumPredictor yes;
    nOuterCorrectors  1;
    nCorrectors       2;
    nNonOrthogonalCorrectors 0;
    pRefCell        0;
    pRefValue       0;
}
"""
    with open(os.path.join(case_dir, "system", "fvSolution"), "w") as f:
        f.write(fv_solution)

    transport = foam_header("dictionary", "transportProperties", "constant") + f"""
transportModel  Newtonian;
nu              nu [0 2 -1 0 0 0 0] {NU};
beta            beta [0 0 0 -1 0 0 0] {BETA};
TRef            TRef [0 0 0 1 0 0 0] {T_REF};
Pr              Pr [0 0 0 0 0 0 0] 0.71;
Prt             Prt [0 0 0 0 0 0 0] 0.71;
"""
    with open(os.path.join(case_dir, "constant", "transportProperties"), "w") as f:
        f.write(transport)

    g_file = foam_header("uniformDimensionedVectorField", "g", "constant") + f"""
dimensions      [0 1 -2 0 0 0 0];
value           (0 {GY} 0);
"""
    with open(os.path.join(case_dir, "constant", "g"), "w") as f:
        f.write(g_file)

    turb_props = foam_header("dictionary", "turbulenceProperties", "constant") + """
simulationType  laminar;
"""
    with open(os.path.join(case_dir, "constant", "turbulenceProperties"), "w") as f:
        f.write(turb_props)

    u_field = foam_header("volVectorField", "U", "0") + """
dimensions      [0 1 -1 0 0 0 0];
internalField   uniform (0 0 0);
boundaryField
{
    hotWall    { type noSlip; }
    coldWall   { type noSlip; }
    bottomWall { type noSlip; }
    topWall    { type noSlip; }
    frontAndBack { type empty; }
}
"""
    with open(os.path.join(case_dir, "0", "U"), "w") as f:
        f.write(u_field)

    t_field = foam_header("volScalarField", "T", "0") + f"""
dimensions      [0 0 0 1 0 0 0];
internalField   uniform {T_REF};
boundaryField
{{
    hotWall    {{ type fixedValue; value uniform {T_HOT}; }}
    coldWall   {{ type fixedValue; value uniform {T_COLD}; }}
    bottomWall {{ type zeroGradient; }}
    topWall    {{ type zeroGradient; }}
    frontAndBack {{ type empty; }}
}}
"""
    with open(os.path.join(case_dir, "0", "T"), "w") as f:
        f.write(t_field)

    p_rgh_field = foam_header("volScalarField", "p_rgh", "0") + """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    hotWall    { type fixedFluxPressure; value uniform 0; }
    coldWall   { type fixedFluxPressure; value uniform 0; }
    bottomWall { type fixedFluxPressure; value uniform 0; }
    topWall    { type fixedFluxPressure; value uniform 0; }
    frontAndBack { type empty; }
}
"""
    with open(os.path.join(case_dir, "0", "p_rgh"), "w") as f:
        f.write(p_rgh_field)

    p_field = foam_header("volScalarField", "p", "0") + """
dimensions      [0 2 -2 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    hotWall    { type calculated; value uniform 0; }
    coldWall   { type calculated; value uniform 0; }
    bottomWall { type calculated; value uniform 0; }
    topWall    { type calculated; value uniform 0; }
    frontAndBack { type empty; }
}
"""
    with open(os.path.join(case_dir, "0", "p"), "w") as f:
        f.write(p_field)

    alphat_field = foam_header("volScalarField", "alphat", "0") + """
dimensions      [0 2 -1 0 0 0 0];
internalField   uniform 0;
boundaryField
{
    hotWall    { type calculated; value uniform 0; }
    coldWall   { type calculated; value uniform 0; }
    bottomWall { type calculated; value uniform 0; }
    topWall    { type calculated; value uniform 0; }
    frontAndBack { type empty; }
}
"""
    with open(os.path.join(case_dir, "0", "alphat"), "w") as f:
        f.write(alphat_field)


def run(cmd, cwd, log_path):
    env = dict(FOAM_ENV)
    env["PWD"] = cwd
    with open(log_path, "w") as logf:
        p = subprocess.run(cmd, cwd=cwd, env=env, stdout=logf, stderr=subprocess.STDOUT)
    return p.returncode


def bench_openfoam(n):
    dx = 1.0 / n
    dt = 0.3 * dx / 1.0  # same CFL convention as the lid-cavity sweep, velocity scale ~O(1)

    case_dir = os.path.join(SCRATCH, f"nc_N{n}")
    if os.path.exists(case_dir):
        shutil.rmtree(case_dir)
    os.makedirs(case_dir)
    write_case(case_dir, n, dt, NSTEPS)

    rc = run(["blockMesh"], case_dir, os.path.join(case_dir, "blockMesh.log"))
    if rc != 0:
        return {"error": f"blockMesh failed (rc={rc})", "N": n}

    t0 = time.perf_counter()
    rc = run(["buoyantBoussinesqPimpleFoam"], case_dir, os.path.join(case_dir, "solve.log"))
    t1 = time.perf_counter()
    if rc != 0:
        return {"error": f"buoyantBoussinesqPimpleFoam failed (rc={rc})", "N": n}

    wall_s = t1 - t0
    cells = n * n * 4
    return {
        "N": n, "cells": cells, "dt": dt, "nsteps": NSTEPS,
        "wall_time_s": wall_s, "ms_per_step": (wall_s * 1000.0) / NSTEPS,
    }


def bench_thapar(n):
    dx = 1.0 / n
    dt = 0.3 * dx / 1.0
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = CUDA_LIBDIR + ":" + env.get("LD_LIBRARY_PATH", "")
    p = subprocess.run([CUDA_BENCH_BIN, str(n), str(NSTEPS), f"{dt:.10f}"],
                        capture_output=True, text=True, env=env)
    if p.returncode != 0:
        return {"error": f"bench_natural_convection failed (rc={p.returncode}): {p.stderr}", "N": n}
    m = re.search(r"BENCH_RESULT N=(\d+) cells=(\d+) nsteps=(\d+) total_ms=([0-9.]+) "
                  r"ms_per_step=([0-9.]+) steps_per_sec=([0-9.]+)", p.stdout)
    if not m:
        return {"error": f"could not parse output: {p.stdout}\n{p.stderr}", "N": n}
    return {
        "N": int(m.group(1)), "cells": int(m.group(2)), "dt": dt, "nsteps": int(m.group(3)),
        "total_ms": float(m.group(4)), "ms_per_step": float(m.group(5)), "steps_per_sec": float(m.group(6)),
    }


def main():
    os.makedirs(SCRATCH, exist_ok=True)
    results = {"config": {
        "case": "Differentially heated square cavity, Ra=10^5, Pr=0.71 (de Vahl Davis 1983)",
        "nu": NU, "alpha_thermal": ALPHA, "beta_thermal": BETA, "T_hot": T_HOT, "T_cold": T_COLD, "gy": GY,
        "nsteps": NSTEPS,
        "cpu": "AMD EPYC 7742 (single core, single buoyantBoussinesqPimpleFoam process)",
        "gpu": "NVIDIA A100-SXM4-80GB (single device, enable_heat=1 Boussinesq UCOF path)",
        "openfoam_version": "v1912 (OpenFOAM Foundation)",
    }, "runs": []}

    for n in RESOLUTIONS:
        print(f"\n=== N={n} ({n*n*4} cells) ===", flush=True)
        print("  -> OpenFOAM buoyantBoussinesqPimpleFoam (1 core)...", flush=True)
        of = bench_openfoam(n)
        print(f"     {of}", flush=True)
        print("  -> Thapar CUDA engine (A100, enable_heat=1)...", flush=True)
        th = bench_thapar(n)
        print(f"     {th}", flush=True)

        entry = {"N": n, "cells": n * n * 4, "openfoam": of, "thapar_gpu": th}
        if "error" not in of and "error" not in th:
            entry["speedup_x"] = of["ms_per_step"] / th["ms_per_step"]
        results["runs"].append(entry)

        out_path = os.path.join(REPO, "openfoam_vs_thapar_natural_convection_report.json")
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2)

    print("\nDone. Report written to openfoam_vs_thapar_natural_convection_report.json")


if __name__ == "__main__":
    main()
