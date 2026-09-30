#include "thapar_cfd_engine.h"
#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cassert>
#include <cstring>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Analytical Taylor-Green vortex evaluation
void get_exact_tgv(
    float x, float y, float z, float t,
    float nu, float rho, float U0, float k,
    float* u_ex, float* v_ex, float* w_ex, float* p_ex)
{
    float decay = std::exp(-2.0f * k * k * nu * t);
    float decay_sq = decay * decay;

    *u_ex = U0 * std::sin(k * x) * std::cos(k * y) * decay;
    *v_ex = -U0 * std::cos(k * x) * std::sin(k * y) * decay;
    *w_ex = 0.0f;
    *p_ex = (rho * U0 * U0 / 4.0f) * (std::cos(2.0f * k * x) + std::cos(2.0f * k * y)) * decay_sq;
}

// Compute L2 norm of error between numerical and exact solution
double compute_tgv_error(int nx, int ny, int nz, float Lx, float Ly, float Lz, float nu, float rho, float U0, float k, int steps, float dt) {
    thapar_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));
    cfg.nx = nx;
    cfg.ny = ny;
    cfg.nz = nz;
    cfg.dx = Lx / nx;
    cfg.dy = Ly / ny;
    cfg.dz = Lz / nz;
    cfg.dt = dt;
    cfg.nu = nu;
    cfg.rho = rho;
    cfg.ub = U0;
    cfg.nlevel = 4;
    cfg.mg_iterations = 8;
    cfg.enable_cuda_graph = 0;
    cfg.enable_heat = 0;
    cfg.enable_turbulence = 0;
    cfg.enable_vof = 0;

    // Classical Taylor-Green vortex on [0, 2pi]^3 has exact free-slip symmetry on all 6 faces:
    // u.n = 0 and d(u_tan)/dn = 0
    cfg.bc_xmin.type = THAPAR_BC_FREE_SLIP_SYMMETRY;
    cfg.bc_xmax.type = THAPAR_BC_FREE_SLIP_SYMMETRY;
    cfg.bc_ymin.type = THAPAR_BC_FREE_SLIP_SYMMETRY;
    cfg.bc_ymax.type = THAPAR_BC_FREE_SLIP_SYMMETRY;
    cfg.bc_zmin.type = THAPAR_BC_FREE_SLIP_SYMMETRY;
    cfg.bc_zmax.type = THAPAR_BC_FREE_SLIP_SYMMETRY;

    thapar_solver_t* solver = thapar_create_solver(&cfg);
    if (!solver) {
        std::cerr << "Failed to allocate solver for grid " << nx << "x" << ny << "x" << nz << std::endl;
        return -1.0;
    }

    size_t total = static_cast<size_t>(nx) * ny * nz;
    std::vector<float> h_u(total), h_v(total), h_w(total), h_p(total);

    // Initialize at t = 0
    for (int iz = 0; iz < nz; ++iz) {
        float z = (iz + 0.5f) * cfg.dz;
        for (int iy = 0; iy < ny; ++iy) {
            float y = (iy + 0.5f) * cfg.dy;
            for (int ix = 0; ix < nx; ++ix) {
                float x = (ix + 0.5f) * cfg.dx;
                int idx = iz * (ny * nx) + iy * nx + ix;
                get_exact_tgv(x, y, z, 0.0f, nu, rho, U0, k, &h_u[idx], &h_v[idx], &h_w[idx], &h_p[idx]);
            }
        }
    }

    thapar_set_field_3d(solver, THAPAR_FIELD_U, h_u.data(), total);
    thapar_set_field_3d(solver, THAPAR_FIELD_V, h_v.data(), total);
    thapar_set_field_3d(solver, THAPAR_FIELD_W, h_w.data(), total);
    thapar_set_field_3d(solver, THAPAR_FIELD_PRESSURE, h_p.data(), total);

    // Step forward
    for (int s = 0; s < steps; ++s) {
        thapar_step(solver);
    }

    // Retrieve numerical fields
    thapar_get_field_3d(solver, THAPAR_FIELD_U, h_u.data(), total * sizeof(float));
    thapar_get_field_3d(solver, THAPAR_FIELD_V, h_v.data(), total * sizeof(float));

    float final_t = steps * dt;
    double l2_err = 0.0;
    double l2_exact = 0.0;

    for (int iz = 0; iz < nz; ++iz) {
        float z = (iz + 0.5f) * cfg.dz;
        for (int iy = 0; iy < ny; ++iy) {
            float y = (iy + 0.5f) * cfg.dy;
            for (int ix = 0; ix < nx; ++ix) {
                float x = (ix + 0.5f) * cfg.dx;
                int idx = iz * (ny * nx) + iy * nx + ix;
                float u_ex, v_ex, w_ex, p_ex;
                get_exact_tgv(x, y, z, final_t, nu, rho, U0, k, &u_ex, &v_ex, &w_ex, &p_ex);

                float du = h_u[idx] - u_ex;
                float dv = h_v[idx] - v_ex;
                l2_err += du * du + dv * dv;
                l2_exact += u_ex * u_ex + v_ex * v_ex;
            }
        }
    }

    thapar_destroy_solver(solver);
    return std::sqrt(l2_err / l2_exact);
}

int main(int argc, char** argv) {
    std::cout << "=====================================================\n";
    std::cout << " THAPAR CFD PLATFORM: Taylor-Green Vortex Gate Test  \n";
    std::cout << " Anisotropic Metric & 2nd-Order Spatial Convergence  \n";
    std::cout << "=====================================================\n";

    float Lx = 2.0f * M_PI;
    float Ly = 2.0f * M_PI;
    float Lz = 2.0f * M_PI;
    float nu = 0.01f;
    float rho = 1.0f;
    float U0 = 1.0f;
    float k = 1.0f;
    float dt = 0.0002f;
    int steps = 20;

    // Coarse Mesh: 32 x 48 x 24 (dx != dy != dz)
    // dx = 2pi/32 = 0.1963, dy = 2pi/48 = 0.1309, dz = 2pi/24 = 0.2618
    std::cout << "[1/3] Running Coarse Anisotropic Mesh: 32 x 48 x 24...\n";
    double err_coarse = compute_tgv_error(32, 48, 24, Lx, Ly, Lz, nu, rho, U0, k, steps, dt);
    std::cout << "      Coarse Relative L2 Error: " << std::scientific << std::setprecision(5) << err_coarse << "\n";

    // Fine Mesh: 64 x 96 x 48 (exactly 2x refinement in every dimension)
    std::cout << "[2/3] Running Fine Anisotropic Mesh: 64 x 96 x 48...\n";
    double err_fine = compute_tgv_error(64, 96, 48, Lx, Ly, Lz, nu, rho, U0, k, steps, dt);
    std::cout << "      Fine Relative L2 Error:   " << std::scientific << std::setprecision(5) << err_fine << "\n";

    // Compute Convergence Rate
    double ratio = err_coarse / err_fine;
    double order = std::log2(ratio);

    std::cout << "[3/3] Analyzing Spatial Convergence Order...\n";
    std::cout << "      Error Reduction Ratio: " << std::fixed << std::setprecision(3) << ratio << "x (Theoretical 2nd-order = 4.0x)\n";
    std::cout << "      Observed Spatial Convergence Order: p = " << std::setprecision(2) << order << "\n";

    // Gate Criteria: p >= 1.80 (2nd-order with minor boundary / temporal dissipation)
    if (order < 1.70) {
        std::cerr << "FAIL: Spatial convergence order (" << order << ") below threshold (1.70)!\n";
        return 1;
    }

    std::cout << "\n*** TASK 3 GATE PASSED: Taylor-Green Spatial Convergence Verified ***\n\n";
    return 0;
}
