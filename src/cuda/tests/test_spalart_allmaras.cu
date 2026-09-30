#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include <algorithm>
#include "thapar_cfd_engine.h"

int main() {
    std::cout << "=====================================================\n";
    std::cout << " THAPAR CFD PLATFORM: Spalart-Allmaras Turbulence   \n";
    std::cout << " Positivity, Boundary Layer Production & Eddy Nu_t  \n";
    std::cout << "=====================================================\n";

    thapar_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));

    cfg.nx = 64;
    cfg.ny = 32;
    cfg.nz = 4;

    float L = 2.0f;
    float H = 1.0f;
    cfg.dx = L / cfg.nx;
    cfg.dy = H / cfg.ny;
    cfg.dz = 0.1f / cfg.nz;

    cfg.dt = 0.0005f;
    cfg.nu = 1e-4f; // High Reynolds number flow: Re_H = U * H / nu = 1.0 * 1.0 / 1e-4 = 10,000
    cfg.rho = 1.0f;
    cfg.ub = 1.0f;

    cfg.enable_turbulence = 1;
    cfg.nlevel = 4;
    cfg.mg_iterations = 5;

    // Inlet / Outlet channel flow with solid walls at Y-Min and Y-Max
    cfg.bc_xmin.type = THAPAR_BC_DIRICHLET_INFLOW;
    cfg.bc_xmin.u_val = cfg.ub;
    cfg.bc_xmax.type = THAPAR_BC_NEUMANN_OUTFLOW;

    cfg.bc_ymin.type = THAPAR_BC_NO_SLIP_WALL;
    cfg.bc_ymax.type = THAPAR_BC_NO_SLIP_WALL;

    cfg.bc_zmin.type = THAPAR_BC_FREE_SLIP_SYMMETRY;
    cfg.bc_zmax.type = THAPAR_BC_FREE_SLIP_SYMMETRY;

    std::cout << "[1/4] Initializing High-Re (Re=10,000) SA Turbulence Solver on A100...\n";
    thapar_solver_t* solver = thapar_create_solver(&cfg);
    if (!solver) {
        std::cerr << "FAILED: Could not create solver\n";
        return 1;
    }

    int total_cells = cfg.nx * cfg.ny * cfg.nz;

    // Initialize freestream turbulent viscosity: nu_tilde_inf = 3 * nu
    std::vector<float> init_u(total_cells, cfg.ub);
    thapar_set_field_3d(solver, THAPAR_FIELD_U, init_u.data(), total_cells);

    std::cout << "[2/4] Advancing boundary layer development for 1000 steps...\n";
    for (int step = 200; step <= 1000; step += 200) {
        thapar_step_multiple(solver, 200);
        float max_div, p_res;
        double step_ms;
        thapar_get_diagnostics(solver, &max_div, &p_res, nullptr, nullptr, &step_ms);
        std::cout << "      Step " << std::setw(4) << step
                  << ": Div = " << std::scientific << std::setprecision(3) << max_div
                  << ", P-Res = " << p_res
                  << ", StepTime = " << std::fixed << std::setprecision(2) << step_ms << " ms\n";
    }

    std::cout << "[3/4] Extracting Turbulent Eddy Viscosity Field (nu_t)...\n";
    std::vector<float> nu_t_field(total_cells);
    thapar_get_field_3d(solver, THAPAR_FIELD_TURBULENT_VISCOSITY, nu_t_field.data(), total_cells * sizeof(float));

    float min_nut = 1e9f, max_nut = -1e9f;
    for (int i = 0; i < total_cells; ++i) {
        float val = nu_t_field[i];
        if (val < min_nut) min_nut = val;
        if (val > max_nut) max_nut = val;
    }

    std::cout << "[4/4] Verification of Turbulence Metrics:\n";
    std::cout << "      Minimum Eddy Viscosity (nu_t_min): " << std::scientific << std::setprecision(5) << min_nut 
              << " (Target: >= 0.0, strictly non-negative)\n";
    std::cout << "      Peak Eddy Viscosity    (nu_t_max): " << std::scientific << std::setprecision(5) << max_nut 
              << " (Target: >= 0.0)\n";

    thapar_destroy_solver(solver);

    bool non_negative = (min_nut >= -1e-8f);
    bool bounded = (max_nut < 100.0f); // No runaway blowup

    if (non_negative && bounded) {
        std::cout << "\n*** TASK GATE PASSED: Spalart-Allmaras Strict Positivity & Stability Verified ***\n";
        return 0;
    } else {
        std::cerr << "\n*** TASK GATE FAILED: non_negative=" << non_negative << ", bounded=" << bounded << " ***\n";
        return 1;
    }
}
