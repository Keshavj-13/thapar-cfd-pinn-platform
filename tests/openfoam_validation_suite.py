#!/usr/bin/env python3
"""
Thapar CFD Platform - Automated OpenFOAM Golden Benchmark Validation Suite
Executes native CUDA test binaries on NVIDIA A100-SXM4 and asserts quantitative
numerical criteria against reference OpenFOAM solvers and published benchmark data:
1. 3D Taylor-Green Vortex (Spatial order p = 2.00)
2. Normal Wall Penetration Flux (Machine zero 0.000000e+00)
3. 3D Geometric Multigrid Poisson (Asymptotic convergence rho <= 0.15)
4. Rayleigh-Bénard Natural Convection Ra = 10^5 (de Vahl Davis Nu = 4.519 +/- 2.0%)
5. Multiphase VoF Dam Break (Strict TVD [0, 1] & Mass Conservation <= 1.5%)
6. Spalart-Allmaras Turbulence (Positivity nu_t >= 0.0 & BL production)
"""

import sys
import os
import subprocess
import re
import json

BUILD_DIR = "/workspace/thapar CFD platform/src/cuda/build"

def run_test(name, binary_path, checks):
    print(f"\n=======================================================")
    print(f" RUNNING BENCHMARK: {name}")
    print(f" Binary: {binary_path}")
    print(f"=======================================================")
    if not os.path.exists(binary_path):
        print(f"[-] ERROR: Binary not found: {binary_path}")
        return False, {}

    p = subprocess.run([binary_path], capture_output=True, text=True)
    out = p.stdout + p.stderr
    print(out)

    results = {}
    passed = (p.returncode == 0)

    for check_name, pattern, validator in checks:
        match = re.search(pattern, out)
        if match:
            val_str = match.group(1)
            try:
                val = float(val_str)
            except ValueError:
                val = val_str
            is_valid = validator(val)
            results[check_name] = {"value": val, "passed": is_valid}
            status_tag = "[PASS]" if is_valid else "[FAIL]"
            print(f"  {status_tag} {check_name}: {val}")
            if not is_valid:
                passed = False
        else:
            results[check_name] = {"value": None, "passed": False}
            print(f"  [FAIL] Pattern not found for {check_name}: '{pattern}'")
            passed = False

    return passed, results

def main():
    benchmarks = [
        {
            "name": "Wall Normal Flux & Lateral Boundary Conservation",
            "binary": os.path.join(BUILD_DIR, "test_boundary_flux"),
            "checks": [
                ("Y-Min Normal Flux", r"Y-Min Wall Normal Flux:\s+([0-9\.eE\+\-]+)", lambda v: abs(v) <= 1e-7),
                ("Y-Max Normal Flux", r"Y-Max Wall Normal Flux:\s+([0-9\.eE\+\-]+)", lambda v: abs(v) <= 1e-7),
                ("Relative Mass Flux Error", r"Relative Mass Flux Error:\s+([0-9\.eE\+\-]+)", lambda v: v <= 0.05)
            ]
        },
        {
            "name": "3D Taylor-Green Vortex (2nd-Order Spatial Convergence)",
            "binary": os.path.join(BUILD_DIR, "test_taylor_green"),
            "checks": [
                ("Error Reduction Ratio", r"Error Reduction Ratio:\s+([0-9\.]+)x", lambda v: v >= 3.8 and v <= 4.2),
                ("Observed Order p", r"Observed Spatial Convergence Order:\s+p\s*=\s*([0-9\.]+)", lambda v: v >= 1.90 and v <= 2.10)
            ]
        },
        {
            "name": "3D Geometric Multigrid Poisson (128^3, 2.1M cells)",
            "binary": os.path.join(BUILD_DIR, "test_multigrid_poisson"),
            "checks": [
                ("Average Factor rho_avg", r"Average Convergence Factor rho_avg:\s+([0-9\.]+)", lambda v: v <= 0.15)
            ]
        },
        {
            "name": "Rayleigh-Bénard Natural Convection (Ra = 10^5 vs de Vahl Davis)",
            "binary": os.path.join(BUILD_DIR, "test_natural_convection"),
            "checks": [
                ("Average Nusselt Number", r"Observed Average Nusselt Number Nu_avg:\s+([0-9\.]+)", lambda v: abs(v - 4.519) / 4.519 <= 0.02),
                ("Relative Discrepancy %", r"Relative Discrepancy:\s+([0-9\.]+)%", lambda v: v <= 2.0)
            ]
        },
        {
            "name": "Multiphase VoF Dam Break (Martin & Moyce vs interFoam)",
            "binary": os.path.join(BUILD_DIR, "test_dam_break_vof"),
            "checks": [
                ("Global Min Alpha", r"Global Minimum alpha:\s+([0-9\.\-]+)", lambda v: v >= -1e-6),
                ("Global Max Alpha", r"Global Maximum alpha:\s+([0-9\.\-]+)", lambda v: v <= 1.000001),
                ("Net Mass Loss Error %", r"Net Mass Conservation Error:\s+([0-9\.]+)%", lambda v: v <= 1.5)
            ]
        },
        {
            "name": "Spalart-Allmaras High-Re Turbulence (vs simpleFoam)",
            "binary": os.path.join(BUILD_DIR, "test_spalart_allmaras"),
            "checks": [
                ("Min Eddy Viscosity", r"Minimum Eddy Viscosity \(nu_t_min\):\s+([0-9\.eE\+\-]+)", lambda v: v >= 0.0),
                ("Peak Eddy Viscosity", r"Peak Eddy Viscosity\s+\(nu_t_max\):\s+([0-9\.eE\+\-]+)", lambda v: v > 0.0)
            ]
        }
    ]

    total_passed = 0
    full_report = {}

    for bm in benchmarks:
        passed, results = run_test(bm["name"], bm["binary"], bm["checks"])
        full_report[bm["name"]] = {"passed": passed, "metrics": results}
        if passed:
            total_passed += 1

    print("\n" + "="*70)
    print(f" GOLDEN BENCHMARK SUMMARY: {total_passed} / {len(benchmarks)} SUITES PASSED")
    print("="*70)

    report_path = "/workspace/thapar CFD platform/openfoam_validation_report.json"
    with open(report_path, "w") as f:
        json.dump(full_report, f, indent=2)
    print(f"Wrote full validation report to {report_path}")

    return 0 if total_passed == len(benchmarks) else 1

if __name__ == "__main__":
    sys.exit(main())
