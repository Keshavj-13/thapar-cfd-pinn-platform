/*
 * Thapar CFD Platform - GPU vs OpenFOAM Natural Convection Performance Benchmark
 *
 * Mirrors tests/test_natural_convection.cu's exact Ra=10^5 de Vahl Davis
 * differentially-heated-cavity configuration (already accuracy-validated
 * there against Nu_avg = 4.519 +/- 2.0%), but times raw wall-clock cost per
 * timestep instead of running to steady state -- this is the multiphysics
 * (momentum + Boussinesq-coupled heat transport) counterpart to the
 * incompressible-only lid-driven-cavity benchmark in bench_lid_cavity.c,
 * matched against OpenFOAM's buoyantBoussinesqPimpleFoam
 * (tools/run_openfoam_natural_convection_benchmark.py).
 *
 * Usage: bench_natural_convection <N> <nsteps> <dt>
 */
#include "thapar_cfd_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1000.0 + (double)ts.tv_nsec / 1e6;
}

static int compute_nlevel(int n) {
    int levels = 0, v = n;
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

    thapar_config_t cfg;
    memset(&cfg, 0, sizeof(cfg));

    cfg.nx = n; cfg.ny = n; cfg.nz = 4;
    float L = 1.0f;
    cfg.dx = L / n; cfg.dy = L / n; cfg.dz = L / cfg.nz;

    /* Same Ra=10^5, Pr=0.71 dimensionless parameters as test_natural_convection.cu */
    cfg.nu = 0.01f;
    cfg.rho = 1.0f;
    cfg.thermal_diffusivity = cfg.nu / 0.71f;
    cfg.dt = (float)dt;
    cfg.gy = -10.0f;
    cfg.beta_thermal = (1e5f * cfg.nu * cfg.thermal_diffusivity) / (10.0f * 1.0f);
    cfg.t_ref = 0.5f;
    cfg.enable_heat = 1;
    cfg.nlevel = compute_nlevel(n);
    cfg.mg_iterations = 5;
    cfg.enable_cuda_graph = 0;

    cfg.bc_xmin.type = THAPAR_BC_ISOTHERMAL_WALL; cfg.bc_xmin.t_val = 1.0f;
    cfg.bc_xmax.type = THAPAR_BC_ISOTHERMAL_WALL; cfg.bc_xmax.t_val = 0.0f;
    cfg.bc_ymin.type = THAPAR_BC_NO_SLIP_WALL;
    cfg.bc_ymax.type = THAPAR_BC_NO_SLIP_WALL;
    cfg.bc_zmin.type = THAPAR_BC_FREE_SLIP_SYMMETRY;
    cfg.bc_zmax.type = THAPAR_BC_FREE_SLIP_SYMMETRY;

    thapar_solver_t* solver = thapar_create_solver(&cfg);
    if (!solver) {
        fprintf(stderr, "ERROR: solver creation failed for N=%d\n", n);
        return 1;
    }

    /* Linear temperature profile IC, exactly as test_natural_convection.cu */
    size_t total = (size_t)n * n * cfg.nz;
    float* init_T = (float*)malloc(total * sizeof(float));
    for (int z = 0; z < cfg.nz; ++z)
        for (int y = 0; y < n; ++y)
            for (int x = 0; x < n; ++x) {
                float x_coord = (x + 0.5f) * cfg.dx;
                init_T[z * (n * n) + y * n + x] = 1.0f - (x_coord / L);
            }
    thapar_set_field_3d(solver, THAPAR_FIELD_TEMPERATURE, init_T, total);
    free(init_T);

    thapar_status_t st = thapar_step(solver); /* untimed warmup */
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
    long long cells = (long long)n * n * cfg.nz;

    printf("BENCH_RESULT N=%d cells=%lld nsteps=%d total_ms=%.4f ms_per_step=%.6f steps_per_sec=%.3f\n",
           n, cells, nsteps, total_ms, ms_per_step, 1000.0 / ms_per_step);

    thapar_destroy_solver(solver);
    return 0;
}
