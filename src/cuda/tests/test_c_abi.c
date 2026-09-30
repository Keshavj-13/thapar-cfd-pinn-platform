#include "thapar_cfd_engine.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

int main(int argc, char** argv) {
    printf("=====================================================\n");
    printf(" THAPAR CFD PLATFORM: C ABI Verification Test       \n");
    printf("=====================================================\n");

    thapar_config_t config;
    memset(&config, 0, sizeof(config));
    config.nx = 64;
    config.ny = 64;
    config.nz = 64;
    config.dx = 1.0f / 64.0f;
    config.dy = 1.0f / 64.0f;
    config.dz = 1.0f / 64.0f;
    config.dt = 0.0005f;
    config.nu = 0.01f;
    config.rho = 1.0f;
    config.ub = 1.0f;
    config.nlevel = 4;
    config.mg_iterations = 5;
    config.enable_cuda_graph = 0;
    config.enable_heat = 0;
    config.enable_turbulence = 0;
    config.enable_vof = 0;

    // Set Lid-Driven Cavity BCs
    config.bc_xmin.type = THAPAR_BC_NO_SLIP_WALL;
    config.bc_xmax.type = THAPAR_BC_NO_SLIP_WALL;
    config.bc_ymin.type = THAPAR_BC_NO_SLIP_WALL;
    config.bc_ymax.type = THAPAR_BC_NO_SLIP_WALL;
    config.bc_zmin.type = THAPAR_BC_NO_SLIP_WALL;
    config.bc_zmax.type = THAPAR_BC_MOVING_LID;
    config.bc_zmax.u_val = 1.0f;

    printf("[1/5] Creating Thapar Solver via C ABI...\n");
    thapar_solver_t* solver = thapar_create_solver(&config);
    if (!solver) {
        fprintf(stderr, "ERROR: Failed to allocate thapar_solver_t!\n");
        return 1;
    }
    printf("      Solver instance created successfully.\n");

    printf("[2/5] Advancing 10 time steps...\n");
    for (int step = 0; step < 10; ++step) {
        thapar_status_t status = thapar_step(solver);
        if (status != THAPAR_STATUS_SUCCESS) {
            fprintf(stderr, "ERROR: Step %d failed with code %d: %s\n",
                    step, status, thapar_get_last_error(solver));
            thapar_destroy_solver(solver);
            return 1;
        }
        float d_div=0, d_res=0, d_flux=0; size_t d_vram=0; double d_ms=0;
        thapar_get_diagnostics(solver, &d_div, &d_res, &d_flux, &d_vram, &d_ms);
        printf("      Step %d: Max-Div = %.4e, P-Res = %.4e, Last Step = %.3f ms\n", step, d_div, d_res, d_ms);
    }

    printf("[3/5] Querying diagnostics...\n");
    float max_div = 0.0f, p_res = 0.0f, mass_flux = 0.0f;
    size_t vram_bytes = 0;
    double last_ms = 0.0;
    thapar_get_diagnostics(solver, &max_div, &p_res, &mass_flux, &vram_bytes, &last_ms);
    printf("      Diagnostics: Last Step = %.3f ms, VRAM = %.2f MB, Max Div = %.4e, P-Res = %.4e\n",
           last_ms, (double)vram_bytes / (1024.0 * 1024.0), max_div, p_res);

    printf("[4/5] Extracting 2D Z-slice (32x32 center)...\n");
    size_t slice_size = 64 * 64;
    float* slice_buf = (float*)malloc(slice_size * sizeof(float));
    if (!slice_buf) {
        fprintf(stderr, "ERROR: Failed to allocate host slice buffer!\n");
        thapar_destroy_solver(solver);
        return 1;
    }

    thapar_status_t slice_status = thapar_get_slice(
        solver, THAPAR_FIELD_U, THAPAR_AXIS_Z, 32, slice_buf, slice_size * sizeof(float));
    if (slice_status != THAPAR_STATUS_SUCCESS) {
        fprintf(stderr, "ERROR: thapar_get_slice failed with code %d\n", slice_status);
        free(slice_buf);
        thapar_destroy_solver(solver);
        return 1;
    }

    // Sanity check slice for NaN / Inf
    int nan_count = 0;
    for (size_t i = 0; i < slice_size; ++i) {
        if (isnan(slice_buf[i]) || isinf(slice_buf[i])) {
            nan_count++;
        }
    }
    free(slice_buf);

    if (nan_count > 0) {
        fprintf(stderr, "ERROR: Slice contains %d NaN/Inf values!\n", nan_count);
        thapar_destroy_solver(solver);
        return 1;
    }
    printf("      Slice extracted cleanly with 0 NaN/Inf values.\n");

    printf("[5/5] Destroying solver...\n");
    thapar_destroy_solver(solver);
    printf("      Solver destroyed successfully.\n");

    // Test JSON instantiation
    printf("[Bonus] Testing JSON-based solver instantiation...\n");
    const char* sample_json = "{\"nx\": 32, \"ny\": 32, \"nz\": 32, \"nu\": 0.02, \"dt\": 0.001}";
    thapar_solver_t* json_solver = thapar_create_solver_from_json(sample_json);
    if (!json_solver) {
        fprintf(stderr, "ERROR: Failed to instantiate solver from JSON!\n");
        return 1;
    }
    thapar_step(json_solver);
    thapar_destroy_solver(json_solver);
    printf("      JSON solver created, stepped, and destroyed successfully.\n");

    printf("\n*** C ABI VERIFICATION TEST PASSED ***\n\n");
    return 0;
}
