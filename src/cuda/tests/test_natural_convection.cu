#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include "thapar_cfd_engine.h"

int main() {
    std::cout << "=====================================================\n";
    std::cout << " THAPAR CFD PLATFORM: Natural Convection Gate Test   \n";
    std::cout << " Ra = 10^5 Differentially Heated Cavity Benchmark    \n";
    std::cout << " Target: de Vahl Davis (1983) Nu_avg = 4.519 +/- 2.0%\n";
    std::cout << "=====================================================\n";

    thapar_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));

    int N = 64;
    cfg.nx = N;
    cfg.ny = N;
    cfg.nz = 4;

    float L = 1.0f;
    cfg.dx = L / cfg.nx;
    cfg.dy = L / cfg.ny;
    cfg.dz = L / cfg.nz;

    // Dimensionless parameters for Ra = 10^5, Pr = 0.71
    cfg.nu = 0.01f;
    cfg.rho = 1.0f;
    cfg.thermal_diffusivity = cfg.nu / 0.71f; // ~0.0140845f
    cfg.dt = 0.0005f;

    cfg.gy = -10.0f;
    cfg.gx = 0.0f;
    cfg.gz = 0.0f;
    cfg.beta_thermal = (1e5f * cfg.nu * cfg.thermal_diffusivity) / (10.0f * 1.0f);
    cfg.t_ref = 0.5f; // Symmetric reference temperature

    cfg.enable_heat = 1;
    cfg.nlevel = 4;
    cfg.mg_iterations = 5;

    // Boundaries:
    // Left (X-Min): Isothermal Hot Wall (T = 1.0), No-Slip
    cfg.bc_xmin.type = THAPAR_BC_ISOTHERMAL_WALL;
    cfg.bc_xmin.t_val = 1.0f;

    // Right (X-Max): Isothermal Cold Wall (T = 0.0), No-Slip
    cfg.bc_xmax.type = THAPAR_BC_ISOTHERMAL_WALL;
    cfg.bc_xmax.t_val = 0.0f;

    // Bottom (Y-Min) & Top (Y-Max): Adiabatic, No-Slip
    cfg.bc_ymin.type = THAPAR_BC_NO_SLIP_WALL;
    cfg.bc_ymax.type = THAPAR_BC_NO_SLIP_WALL;

    // Z-Min & Z-Max: Free-Slip Symmetry (pseudo-2D)
    cfg.bc_zmin.type = THAPAR_BC_FREE_SLIP_SYMMETRY;
    cfg.bc_zmax.type = THAPAR_BC_FREE_SLIP_SYMMETRY;

    std::cout << "[1/4] Initializing Ra=10^5 solver on A100 (" << cfg.nx << "x" << cfg.ny << "x" << cfg.nz << ")...\n";
    std::cout << "      nu = " << cfg.nu << ", alpha_th = " << cfg.thermal_diffusivity 
              << ", beta = " << cfg.beta_thermal << ", t_ref = " << cfg.t_ref << ", dt = " << cfg.dt << "\n";

    thapar_solver_t* solver = thapar_create_solver(&cfg);
    if (!solver) {
        std::cerr << "FAILED: Could not create solver\n";
        return 1;
    }

    // Initialize linear temperature profile: T(x) = 1.0 - x/L
    std::vector<float> init_T(cfg.nx * cfg.ny * cfg.nz);
    for (int z = 0; z < cfg.nz; ++z) {
        for (int y = 0; y < cfg.ny; ++y) {
            for (int x = 0; x < cfg.nx; ++x) {
                float x_coord = (x + 0.5f) * cfg.dx;
                init_T[z * (cfg.ny * cfg.nx) + y * cfg.nx + x] = 1.0f - (x_coord / L);
            }
        }
    }
    thapar_set_field_3d(solver, THAPAR_FIELD_TEMPERATURE, init_T.data(), init_T.size());

    std::cout << "[2/4] Advancing natural convection to asymptotic steady state (20,000 steps = 10.0s)...\n";
    std::vector<float> T_field(cfg.nx * cfg.ny * cfg.nz);
    int z_mid = cfg.nz / 2;
    double Nu_cur = 0.0;

    for (int step = 2000; step <= 20000; step += 2000) {
        thapar_step_multiple(solver, 2000);
        float max_div, p_res;
        double step_ms;
        thapar_get_diagnostics(solver, &max_div, &p_res, nullptr, nullptr, &step_ms);
        
        thapar_get_field_3d(solver, THAPAR_FIELD_TEMPERATURE, T_field.data(), T_field.size() * sizeof(float));
        double nu_sum = 0.0;
        for (int y = 0; y < cfg.ny; ++y) {
            float T1 = T_field[z_mid * (cfg.ny * cfg.nx) + y * cfg.nx + 0];
            float T2 = T_field[z_mid * (cfg.ny * cfg.nx) + y * cfg.nx + 1];
            float Nu_y = (8.0f * 1.0f - 9.0f * T1 + T2) / (3.0f * cfg.dx);
            nu_sum += Nu_y;
        }
        Nu_cur = nu_sum / cfg.ny;

        std::cout << "      Step " << std::setw(5) << step
                  << ": Nu_avg = " << std::fixed << std::setprecision(4) << Nu_cur
                  << ", Div = " << std::scientific << std::setprecision(3) << max_div
                  << ", P-Res = " << p_res
                  << ", StepTime = " << std::fixed << std::setprecision(2) << step_ms << " ms\n";
    }

    std::cout << "[3/4] Final Evaluation at Hot Wall (X=0):\n";
    std::cout << "[4/4] Validation Against de Vahl Davis (1983) Benchmark:\n";
    std::cout << "      Observed Average Nusselt Number Nu_avg: " << std::fixed << std::setprecision(4) << Nu_cur << "\n";
    std::cout << "      Benchmark de Vahl Davis (1983) Target:   4.519\n";

    double target_Nu = 4.519;
    double rel_error_pct = std::abs(Nu_cur - target_Nu) / target_Nu * 100.0;
    std::cout << "      Relative Discrepancy: " << std::setprecision(2) << rel_error_pct << "%\n";

    thapar_destroy_solver(solver);

    if (rel_error_pct <= 2.0) {
        std::cout << "\n*** TASK GATE PASSED: Natural Convection Nu = " << Nu_cur 
                  << " within " << std::setprecision(2) << rel_error_pct << "% of de Vahl Davis (tol <= 2.0%) ***\n";
        return 0;
    } else {
        std::cerr << "\n*** TASK GATE FAILED: Nu discrepancy " << rel_error_pct << "% > 2.0% ***\n";
        return 1;
    }
}
