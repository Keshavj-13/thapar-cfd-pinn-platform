#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <cstring>
#include "thapar_cfd_engine.h"

int main() {
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

    cfg.nu = 0.01f;
    cfg.rho = 1.0f;
    cfg.thermal_diffusivity = cfg.nu / 0.71f;
    cfg.dt = 0.0005f;

    cfg.gy = -10.0f;
    cfg.gx = 0.0f;
    cfg.gz = 0.0f;
    cfg.beta_thermal = (1e5f * cfg.nu * cfg.thermal_diffusivity) / (10.0f * 1.0f);
    cfg.t_ref = 0.5f;

    cfg.enable_heat = 1;
    cfg.nlevel = 4;
    cfg.mg_iterations = 5;

    cfg.bc_xmin.type = THAPAR_BC_ISOTHERMAL_WALL;
    cfg.bc_xmin.t_val = 1.0f;
    cfg.bc_xmax.type = THAPAR_BC_ISOTHERMAL_WALL;
    cfg.bc_xmax.t_val = 0.0f;
    cfg.bc_ymin.type = THAPAR_BC_NO_SLIP_WALL;
    cfg.bc_ymax.type = THAPAR_BC_NO_SLIP_WALL;
    cfg.bc_zmin.type = THAPAR_BC_FREE_SLIP_SYMMETRY;
    cfg.bc_zmax.type = THAPAR_BC_FREE_SLIP_SYMMETRY;

    thapar_solver_t* solver = thapar_create_solver(&cfg);

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

    std::vector<float> T_field(cfg.nx * cfg.ny * cfg.nz);
    int z_mid = cfg.nz / 2;

    for (int step = 1000; step <= 25000; step += 1000) {
        thapar_step_multiple(solver, 1000);
        thapar_get_field_3d(solver, THAPAR_FIELD_TEMPERATURE, T_field.data(), T_field.size() * sizeof(float));
        double nu_sum = 0.0;
        for (int y = 0; y < cfg.ny; ++y) {
            float T1 = T_field[z_mid * (cfg.ny * cfg.nx) + y * cfg.nx + 0];
            float T2 = T_field[z_mid * (cfg.ny * cfg.nx) + y * cfg.nx + 1];
            float Nu_y = (8.0f * 1.0f - 9.0f * T1 + T2) / (3.0f * cfg.dx);
            nu_sum += Nu_y;
        }
        double Nu_cur = nu_sum / cfg.ny;
        std::cout << "Step " << std::setw(5) << step << ": Nu = " << std::fixed << std::setprecision(4) << Nu_cur << std::endl;
    }

    thapar_destroy_solver(solver);
    return 0;
}
