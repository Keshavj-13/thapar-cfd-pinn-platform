/*
 * Thapar CFD Platform - GPU vs OpenFOAM Wall-Clock Performance Benchmark
 *
 * Runs the native 3D lid-driven cavity (Re=1000) on the A100-resident
 * CUDA engine at a caller-specified grid resolution and step count,
 * mirroring the exact case geometry, viscosity, and CFL-limited
 * timestep used in the matched OpenFOAM icoFoam benchmark case
 * (see tools/run_openfoam_benchmark.sh).
 *
 * Usage: bench_lid_cavity <N> <nsteps> <dt>
 * Prints a single machine-parsable line:
 *   BENCH_RESULT N=<N> cells=<C> nsteps=<S> total_ms=<T> ms_per_step=<M> steps_per_sec=<R>
 */
#include "thapar_cfd_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int compute_nlevel(int n) {
    int levels = 0;
    int v = n;
    while (v >= 4) { v /= 2; levels++; }
    if (levels < 2) levels = 2;
    if (levels > 6) levels = 6;
    return levels;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "Usage: %s <N> <nsteps> <dt>\n", argv[0]);
        return 1;
    }
    int n = atoi(argv[1]);
    int nsteps = atoi(argv[2]);
    double dt = atof(argv[3]);
    double re = (argc > 4) ? atof(argv[4]) : 1000.0;
    double L = 1.0;
    double ub = 1.0;
    double nu = ub * L / re;

    thapar_config_t config;
    memset(&config, 0, sizeof(config));
    config.nx = n; config.ny = n; config.nz = n;
    config.dx = (float)(L / n); config.dy = (float)(L / n); config.dz = (float)(L / n);
    config.dt = (float)dt;
    config.nu = (float)nu;
    config.rho = 1.0f;
    config.ub = (float)ub;
    config.nlevel = compute_nlevel(n);
    config.mg_iterations = 5;
    config.enable_cuda_graph = 0;
    config.enable_heat = 0;
    config.enable_turbulence = 0;
    config.enable_vof = 0;

    config.bc_xmin.type = THAPAR_BC_NO_SLIP_WALL;
    config.bc_xmax.type = THAPAR_BC_NO_SLIP_WALL;
    config.bc_ymin.type = THAPAR_BC_NO_SLIP_WALL;
    config.bc_ymax.type = THAPAR_BC_NO_SLIP_WALL;
    config.bc_zmin.type = THAPAR_BC_NO_SLIP_WALL;
    config.bc_zmax.type = THAPAR_BC_MOVING_LID;
    config.bc_zmax.u_val = (float)ub;

    thapar_solver_t* solver = thapar_create_solver(&config);
    if (!solver) {
        fprintf(stderr, "ERROR: solver creation failed for N=%d\n", n);
        return 1;
    }

    /* Warmup: one step outside the timing window to absorb CUDA graph
       capture / JIT / first-touch allocation cost, matching how the
       UCOF engine amortizes one-time specialization before steady-state
       execution (see Section D of the engineering plan). */
    thapar_status_t st = thapar_step(solver);
    if (st != THAPAR_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: warmup step failed: %s\n", thapar_get_last_error(solver));
        thapar_destroy_solver(solver);
        return 1;
    }

    double t0 = now_ms();
    st = thapar_step_multiple(solver, nsteps);
    double t1 = now_ms();

    if (st != THAPAR_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: step_multiple failed: %s\n", thapar_get_last_error(solver));
        thapar_destroy_solver(solver);
        return 1;
    }

    double total_ms = t1 - t0;
    double ms_per_step = total_ms / nsteps;
    double steps_per_sec = 1000.0 / ms_per_step;
    long long cells = (long long)n * n * n;

    float max_div=0, p_res=0, mass_flux=0; size_t vram=0; double last_ms=0;
    thapar_get_diagnostics(solver, &max_div, &p_res, &mass_flux, &vram, &last_ms);

    printf("BENCH_RESULT N=%d cells=%lld nsteps=%d total_ms=%.4f ms_per_step=%.6f steps_per_sec=%.3f vram_mb=%.2f max_div=%.4e\n",
           n, cells, nsteps, total_ms, ms_per_step, steps_per_sec, (double)vram / (1024.0*1024.0), max_div);

    thapar_destroy_solver(solver);
    return 0;
}
