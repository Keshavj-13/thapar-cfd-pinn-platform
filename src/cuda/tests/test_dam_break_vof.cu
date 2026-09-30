#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include "thapar_cfd_engine.h"

int main() {
    std::cout << "=====================================================\n";
    std::cout << " THAPAR CFD PLATFORM: Multiphase VoF Dam Break Test  \n";
    std::cout << " Strict TVD Monotonicity [0, 1] & Mass Conservation  \n";
    std::cout << "=====================================================\n";

    thapar_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));

    cfg.nx = 64;
    cfg.ny = 64;
    cfg.nz = 4;

    float L = 1.0f;
    cfg.dx = L / cfg.nx;
    cfg.dy = L / cfg.ny;
    cfg.dz = 0.1f / cfg.nz;

    cfg.dt = 0.0005f;
    cfg.nu = 0.001f;
    cfg.rho = 1000.0f;

    // Multiphase VoF settings
    cfg.enable_vof = 1;
    cfg.rho1 = 1000.0f;  // Water
    cfg.rho2 = 1.0f;     // Air
    cfg.nu1 = 1e-6f;     // Water viscosity
    cfg.nu2 = 1.5e-5f;   // Air viscosity
    cfg.c_alpha = 1.0f;  // OpenFOAM interface compression factor

    // Gravity downwards
    cfg.gx = 0.0f;
    cfg.gy = -9.81f;
    cfg.gz = 0.0f;

    cfg.nlevel = 4;
    cfg.mg_iterations = 5;

    // Cavity boundary walls: No-slip on all sides, free-slip in Z
    cfg.bc_xmin.type = THAPAR_BC_NO_SLIP_WALL;
    cfg.bc_xmax.type = THAPAR_BC_NO_SLIP_WALL;
    cfg.bc_ymin.type = THAPAR_BC_NO_SLIP_WALL;
    cfg.bc_ymax.type = THAPAR_BC_NO_SLIP_WALL;
    cfg.bc_zmin.type = THAPAR_BC_FREE_SLIP_SYMMETRY;
    cfg.bc_zmax.type = THAPAR_BC_FREE_SLIP_SYMMETRY;

    std::cout << "[1/4] Initializing VoF Dam Break Solver (" << cfg.nx << "x" << cfg.ny << "x" << cfg.nz << ")...\n";
    thapar_solver_t* solver = thapar_create_solver(&cfg);
    if (!solver) {
        std::cerr << "FAILED: Could not create solver\n";
        return 1;
    }

    // Set initial water column in bottom-left corner: x in [0, 0.3L], y in [0, 0.6L]
    int total_cells = cfg.nx * cfg.ny * cfg.nz;
    std::vector<float> init_alpha(total_cells, 0.0f);
    double initial_liquid_mass = 0.0;

    for (int z = 0; z < cfg.nz; ++z) {
        for (int y = 0; y < cfg.ny; ++y) {
            for (int x = 0; x < cfg.nx; ++x) {
                float px = (x + 0.5f) * cfg.dx;
                float py = (y + 0.5f) * cfg.dy;
                int idx = z * (cfg.ny * cfg.nx) + y * cfg.nx + x;
                if (px <= 0.35f * L && py <= 0.65f * L) {
                    init_alpha[idx] = 1.0f;
                    initial_liquid_mass += 1.0;
                }
            }
        }
    }

    thapar_set_field_3d(solver, THAPAR_FIELD_VOF_ALPHA, init_alpha.data(), total_cells);
    std::cout << "      Initial Liquid Cells: " << initial_liquid_mass << " / " << total_cells << "\n";

    std::cout << "[2/4] Advancing Dam Break for 1000 steps (t = 0.50s)...\n";
    std::vector<float> alpha_field(total_cells);
    float min_alpha_global = 1.0f;
    float max_alpha_global = 0.0f;
    double final_liquid_mass = 0.0;

    for (int step = 100; step <= 1000; step += 100) {
        thapar_step_multiple(solver, 100);
        
        thapar_get_field_3d(solver, THAPAR_FIELD_VOF_ALPHA, alpha_field.data(), total_cells * sizeof(float));
        
        float min_a = 1e9f, max_a = -1e9f;
        double current_mass = 0.0;
        for (int i = 0; i < total_cells; ++i) {
            float a = alpha_field[i];
            if (a < min_a) min_a = a;
            if (a > max_a) max_a = a;
            current_mass += a;
        }

        if (min_a < min_alpha_global) min_alpha_global = min_a;
        if (max_a > max_alpha_global) max_alpha_global = max_a;
        final_liquid_mass = current_mass;

        double mass_ratio = current_mass / initial_liquid_mass;
        std::cout << "      Step " << std::setw(4) << step
                  << ": alpha in [" << std::fixed << std::setprecision(5) << min_a << ", " << max_a << "]"
                  << ", Mass Conservation = " << std::setprecision(3) << (mass_ratio * 100.0) << "%\n";
    }

    std::cout << "[3/4] Monotonicity & TVD Boundedness Verification:\n";
    std::cout << "      Global Minimum alpha: " << std::setprecision(6) << min_alpha_global << " (Target: >= 0.0)\n";
    std::cout << "      Global Maximum alpha: " << std::setprecision(6) << max_alpha_global << " (Target: <= 1.0)\n";

    double mass_loss_pct = std::abs(final_liquid_mass - initial_liquid_mass) / initial_liquid_mass * 100.0;
    std::cout << "[4/4] Phase Mass Conservation Verification:\n";
    std::cout << "      Net Mass Conservation Error: " << std::setprecision(3) << mass_loss_pct << "% (Target: <= 1.5%)\n";

    thapar_destroy_solver(solver);

    bool tvd_ok = (min_alpha_global >= -1e-6f) && (max_alpha_global <= 1.0f + 1e-6f);
    bool mass_ok = (mass_loss_pct <= 1.5);

    if (tvd_ok && mass_ok) {
        std::cout << "\n*** TASK GATE PASSED: VoF Dam Break Strict TVD [0, 1] & Mass Conservation Verified ***\n";
        return 0;
    } else {
        std::cerr << "\n*** TASK GATE FAILED: TVD=" << (tvd_ok ? "PASS" : "FAIL") 
                  << ", Mass=" << (mass_ok ? "PASS" : "FAIL") << " ***\n";
        return 1;
    }
}
