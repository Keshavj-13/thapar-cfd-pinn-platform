#ifndef MULTIPHYSICS_ENGINE_CUH
#define MULTIPHYSICS_ENGINE_CUH

#include <cuda_runtime.h>
#include "domain_metrics.cuh"

// ============================================================================
// STAGE 1: FUSED HYDRODYNAMIC & THERMAL PREDICTOR MACRO-KERNEL
// Launch Bounds: (256, 2) -> Max 64 registers/thread (Ampere SM 8.0 >= 50% occupancy)
// Computes u*, v*, w*, and T* with Boussinesq buoyancy and Brinkman obstacle damping
// ============================================================================
__global__ void __launch_bounds__(256, 2) fused_hydrodynamic_thermal_predictor_kernel(
    const float* __restrict__ uu, // Padded (D+2) x (H+2) x (W+2)
    const float* __restrict__ vv,
    const float* __restrict__ ww,
    const float* __restrict__ tt, // Padded temperature (if enabled)
    const float* __restrict__ nu_t, // Turbulent eddy viscosity (if enabled)
    const float* __restrict__ sigma, // Brinkman obstacle damping
    float* __restrict__ u_star, // Output interior D x H x W
    float* __restrict__ v_star,
    float* __restrict__ w_star,
    float* __restrict__ t_star,
    int D, int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < W && y < H && z < D) {
        int pad_W = W + 2;
        int pad_H = H + 2;
        int pad_c = (z + 1) * (pad_H * pad_W) + (y + 1) * pad_W + (x + 1);

        // Grid metrics from device constant memory
        float inv_2dx = c_metrics.inv_2dx;
        float inv_2dy = c_metrics.inv_2dy;
        float inv_2dz = c_metrics.inv_2dz;
        float inv_dx2 = c_metrics.inv_dx2;
        float inv_dy2 = c_metrics.inv_dy2;
        float inv_dz2 = c_metrics.inv_dz2;
        float dt      = c_metrics.dt;
        float nu_mol  = c_metrics.nu;

        // Interior index
        int out_idx = z * (H * W) + y * W + x;

        // Effective viscosity (molecular + turbulent)
        float nu_eff = nu_mol;
        if (c_metrics.enable_turbulence && nu_t != nullptr) {
            nu_eff += nu_t[out_idx];
        }

        // Center values
        float u_c = uu[pad_c];
        float v_c = vv[pad_c];
        float w_c = ww[pad_c];

        // 6-Neighbor stencil indices
        int pad_xm = pad_c - 1;         int pad_xp = pad_c + 1;
        int pad_ym = pad_c - pad_W;     int pad_yp = pad_c + pad_W;
        int pad_zm = pad_c - (pad_H * pad_W); int pad_zp = pad_c + (pad_H * pad_W);

        // Velocity 1st derivatives (Central Differencing)
        float du_dx = (uu[pad_xp] - uu[pad_xm]) * inv_2dx;
        float du_dy = (uu[pad_yp] - uu[pad_ym]) * inv_2dy;
        float du_dz = (uu[pad_zp] - uu[pad_zm]) * inv_2dz;

        float dv_dx = (vv[pad_xp] - vv[pad_xm]) * inv_2dx;
        float dv_dy = (vv[pad_yp] - vv[pad_ym]) * inv_2dy;
        float dv_dz = (vv[pad_zp] - vv[pad_zm]) * inv_2dz;

        float dw_dx = (ww[pad_xp] - ww[pad_xm]) * inv_2dx;
        float dw_dy = (ww[pad_yp] - ww[pad_ym]) * inv_2dy;
        float dw_dz = (ww[pad_zp] - ww[pad_zm]) * inv_2dz;

        // Velocity Laplacians
        float lap_u = (uu[pad_xp] - 2.0f * u_c + uu[pad_xm]) * inv_dx2 +
                      (uu[pad_yp] - 2.0f * u_c + uu[pad_ym]) * inv_dy2 +
                      (uu[pad_zp] - 2.0f * u_c + uu[pad_zm]) * inv_dz2;

        float lap_v = (vv[pad_xp] - 2.0f * v_c + vv[pad_xm]) * inv_dx2 +
                      (vv[pad_yp] - 2.0f * v_c + vv[pad_ym]) * inv_dy2 +
                      (vv[pad_zp] - 2.0f * v_c + vv[pad_zm]) * inv_dz2;

        float lap_w = (ww[pad_xp] - 2.0f * w_c + ww[pad_xm]) * inv_dx2 +
                      (ww[pad_yp] - 2.0f * w_c + ww[pad_ym]) * inv_dy2 +
                      (ww[pad_zp] - 2.0f * w_c + ww[pad_zm]) * inv_dz2;

        // Advective transport: (u . grad) u
        float adv_u = u_c * du_dx + v_c * du_dy + w_c * du_dz;
        float adv_v = u_c * dv_dx + v_c * dv_dy + w_c * dv_dz;
        float adv_w = u_c * dw_dx + v_c * dw_dy + w_c * dw_dz;

        // Boussinesq Buoyancy Coupling: a_buoy = g * beta * (T - T_ref)
        float buoy_x = 0.0f, buoy_y = 0.0f, buoy_z = 0.0f;
        if (c_metrics.enable_heat && tt != nullptr) {
            float t_c = tt[pad_c];
            float delta_T = t_c - c_metrics.t_ref;
            float beta_eff = c_metrics.beta_thermal;
            buoy_x = -c_metrics.gx * beta_eff * delta_T;
            buoy_y = -c_metrics.gy * beta_eff * delta_T;
            buoy_z = -c_metrics.gz * beta_eff * delta_T;

            // Thermal energy transport: dT/dt + (u . grad) T = alpha_th * lap(T)
            float dt_dx = (tt[pad_xp] - tt[pad_xm]) * inv_2dx;
            float dt_dy = (tt[pad_yp] - tt[pad_ym]) * inv_2dy;
            float dt_dz = (tt[pad_zp] - tt[pad_zm]) * inv_2dz;
            float adv_t = u_c * dt_dx + v_c * dt_dy + w_c * dt_dz;

            float lap_t = (tt[pad_xp] - 2.0f * t_c + tt[pad_xm]) * inv_dx2 +
                          (tt[pad_yp] - 2.0f * t_c + tt[pad_ym]) * inv_dy2 +
                          (tt[pad_zp] - 2.0f * t_c + tt[pad_zm]) * inv_dz2;

            float t_new = t_c + dt * (-adv_t + c_metrics.thermal_diffusivity * lap_t);
            if (t_star != nullptr) t_star[out_idx] = t_new;
        }

        // Brinkman obstacle penalization: 1 / (1 + dt * sigma)
        float sig = (sigma != nullptr) ? sigma[out_idx] : 0.0f;
        float solid_damp = 1.0f / (1.0f + dt * sig);

        // Momentum Predictor updates
        u_star[out_idx] = (u_c + dt * (-adv_u + nu_eff * lap_u + buoy_x)) * solid_damp;
        v_star[out_idx] = (v_c + dt * (-adv_v + nu_eff * lap_v + buoy_y)) * solid_damp;
        w_star[out_idx] = (w_c + dt * (-adv_w + nu_eff * lap_w + buoy_z)) * solid_damp;
    }
}

// ============================================================================
// STAGE 3: FUSED SCALAR TRANSPORT PREDICTOR MACRO-KERNEL
// Launch Bounds: (256, 2) -> Max 64 registers/thread
// Solves: Spalart-Allmaras (nu_tilde) and VoF (alpha) with OpenFOAM interface compression
// Updates dynamic mixture properties: rho(alpha), nu(alpha)
// ============================================================================
__global__ void __launch_bounds__(256, 2) fused_scalar_transport_predictor_kernel(
    const float* __restrict__ uu, // Velocity fields (padded)
    const float* __restrict__ vv,
    const float* __restrict__ ww,
    const float* __restrict__ nu_tilde_pad, // Spalart-Allmaras padded
    const float* __restrict__ alpha_pad,    // VoF phase padded
    const float* __restrict__ wall_dist,    // Precomputed wall distance field d(x)
    float* __restrict__ nu_tilde_star,      // Output intermediate nu_tilde
    float* __restrict__ nu_t_out,           // Output eddy viscosity nu_t
    float* __restrict__ alpha_star,         // Output intermediate alpha
    float* __restrict__ rho_mix,            // Output mixture density
    float* __restrict__ nu_mix,             // Output mixture viscosity
    int D, int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < W && y < H && z < D) {
        int pad_W = W + 2;
        int pad_H = H + 2;
        int pad_c = (z + 1) * (pad_H * pad_W) + (y + 1) * pad_W + (x + 1);
        int out_idx = z * (H * W) + y * W + x;

        float inv_2dx = c_metrics.inv_2dx;
        float inv_2dy = c_metrics.inv_2dy;
        float inv_2dz = c_metrics.inv_2dz;
        float inv_dx2 = c_metrics.inv_dx2;
        float inv_dy2 = c_metrics.inv_dy2;
        float inv_dz2 = c_metrics.inv_dz2;
        float dt      = c_metrics.dt;
        float nu_mol  = c_metrics.nu;

        // Center velocities
        float u_c = uu[pad_c];
        float v_c = vv[pad_c];
        float w_c = ww[pad_c];

        int pad_xm = pad_c - 1;         int pad_xp = pad_c + 1;
        int pad_ym = pad_c - pad_W;     int pad_yp = pad_c + pad_W;
        int pad_zm = pad_c - (pad_H * pad_W); int pad_zp = pad_c + (pad_H * pad_W);

        // --------------------------------------------------------------------
        // 1. Spalart-Allmaras One-Equation Turbulence Model
        // --------------------------------------------------------------------
        if (c_metrics.enable_turbulence && nu_tilde_pad != nullptr) {
            float nutilde_c = nu_tilde_pad[pad_c];

            // Advection: (u . grad) nu_tilde
            float dnt_dx = (nu_tilde_pad[pad_xp] - nu_tilde_pad[pad_xm]) * inv_2dx;
            float dnt_dy = (nu_tilde_pad[pad_yp] - nu_tilde_pad[pad_ym]) * inv_2dy;
            float dnt_dz = (nu_tilde_pad[pad_zp] - nu_tilde_pad[pad_zm]) * inv_2dz;
            float adv_nt = u_c * dnt_dx + v_c * dnt_dy + w_c * dnt_dz;

            // Laplacian: lap(nu_tilde)
            float lap_nt = (nu_tilde_pad[pad_xp] - 2.0f * nutilde_c + nu_tilde_pad[pad_xm]) * inv_dx2 +
                           (nu_tilde_pad[pad_yp] - 2.0f * nutilde_c + nu_tilde_pad[pad_ym]) * inv_dy2 +
                           (nu_tilde_pad[pad_zp] - 2.0f * nutilde_c + nu_tilde_pad[pad_zm]) * inv_dz2;

            // Gradient magnitude squared: |grad(nu_tilde)|^2
            float grad_nt_sq = dnt_dx * dnt_dx + dnt_dy * dnt_dy + dnt_dz * dnt_dz;

            // Vorticity magnitude S = |curl(u)|
            float dw_dy = (ww[pad_yp] - ww[pad_ym]) * inv_2dy;
            float dv_dz = (vv[pad_zp] - vv[pad_zm]) * inv_2dz;
            float du_dz = (uu[pad_zp] - uu[pad_zm]) * inv_2dz;
            float dw_dx = (ww[pad_xp] - ww[pad_xm]) * inv_2dx;
            float dv_dx = (vv[pad_xp] - vv[pad_xm]) * inv_2dx;
            float du_dy = (uu[pad_yp] - uu[pad_ym]) * inv_2dy;

            float vor_x = dw_dy - dv_dz;
            float vor_y = du_dz - dw_dx;
            float vor_z = dv_dx - du_dy;
            float S_mag = sqrtf(vor_x * vor_x + vor_y * vor_y + vor_z * vor_z + 1e-12f);

            // Wall distance d
            float d_w = (wall_dist != nullptr) ? fmaxf(wall_dist[out_idx], 1e-5f) : 1e-2f;

            // SA Model Damping Functions: chi = nu_tilde / nu
            float chi = fmaxf(nutilde_c / fmaxf(nu_mol, 1e-8f), 0.0f);
            float chi3 = chi * chi * chi;
            float cv1_3 = c_metrics.sa_cv1_3;
            float fv1 = chi3 / (chi3 + cv1_3 + 1e-12f);
            float fv2 = 1.0f - (chi / (1.0f + chi * fv1 + 1e-12f));

            // Modified vorticity S_tilde
            float S_tilde = S_mag + (nutilde_c / (c_metrics.sa_kappa2 * d_w * d_w + 1e-12f)) * fv2;
            S_tilde = fmaxf(S_tilde, 1e-6f);

            // Production term P = c_b1 * S_tilde * nu_tilde
            float prod = c_metrics.sa_cb1 * S_tilde * nutilde_c;

            // Destruction term D = c_w1 * f_w * (nu_tilde / d)^2
            float r_sa = fminf(nutilde_c / (S_tilde * c_metrics.sa_kappa2 * d_w * d_w + 1e-12f), 10.0f);
            float g_sa = r_sa + c_metrics.sa_cw2 * (powf(r_sa, 6.0f) - r_sa);
            float cw3_6 = powf(c_metrics.sa_cw3, 6.0f);
            float fw = g_sa * powf((1.0f + cw3_6) / (powf(g_sa, 6.0f) + cw3_6 + 1e-12f), 1.0f / 6.0f);
            float cw1 = c_metrics.sa_cb1 * c_metrics.sa_sigma_inv + (1.0f + c_metrics.sa_cb2) * c_metrics.sa_sigma_inv;
            float dest = cw1 * fw * ((nutilde_c * nutilde_c) / (d_w * d_w + 1e-12f));

            // Diffusion term
            float diff = c_metrics.sa_sigma_inv * ((nu_mol + nutilde_c) * lap_nt + c_metrics.sa_cb2 * grad_nt_sq);

            // Time advancement with positivity enforcement
            float nutilde_new = nutilde_c + dt * (-adv_nt + prod - dest + diff);
            nutilde_new = fmaxf(nutilde_new, 0.0f);

            if (nu_tilde_star != nullptr) nu_tilde_star[out_idx] = nutilde_new;
            if (nu_t_out != nullptr) nu_t_out[out_idx] = nutilde_new * fv1;
        }

        // --------------------------------------------------------------------
        // 2. Multiphase Volume of Fluid (VoF) with OpenFOAM Interface Compression
        // --------------------------------------------------------------------
        if (c_metrics.enable_vof && alpha_pad != nullptr) {
            float alpha_c = alpha_pad[pad_c];

            // Primary phase gradients (Van Leer limited)
            float da_dx = (alpha_pad[pad_xp] - alpha_pad[pad_xm]) * inv_2dx;
            float da_dy = (alpha_pad[pad_yp] - alpha_pad[pad_ym]) * inv_2dy;
            float da_dz = (alpha_pad[pad_zp] - alpha_pad[pad_zm]) * inv_2dz;
            float grad_alpha_mag = sqrtf(da_dx * da_dx + da_dy * da_dy + da_dz * da_dz + 1e-8f);

            // Standard advective flux: (u . grad) alpha
            float adv_alpha = u_c * da_dx + v_c * da_dy + w_c * da_dz;

            // OpenFOAM Interface Compression Velocity: u_r = c_alpha * |u| * (grad(alpha) / |grad(alpha)|)
            float u_mag = sqrtf(u_c * u_c + v_c * v_c + w_c * w_c + 1e-8f);
            float ur_coeff = c_metrics.c_alpha * u_mag / grad_alpha_mag;
            float ur_x = ur_coeff * da_dx;
            float ur_y = ur_coeff * da_dy;
            float ur_z = ur_coeff * da_dz;

            // Interface compression flux: div( u_r * alpha * (1 - alpha) )
            float comp_xp = alpha_pad[pad_xp] * (1.0f - alpha_pad[pad_xp]);
            float comp_xm = alpha_pad[pad_xm] * (1.0f - alpha_pad[pad_xm]);
            float comp_yp = alpha_pad[pad_yp] * (1.0f - alpha_pad[pad_yp]);
            float comp_ym = alpha_pad[pad_ym] * (1.0f - alpha_pad[pad_ym]);
            float comp_zp = alpha_pad[pad_zp] * (1.0f - alpha_pad[pad_zp]);

            float div_comp = (comp_xp * ur_x - comp_xm * ur_x) * inv_2dx +
                             (comp_yp * ur_y - comp_ym * ur_y) * inv_2dy +
                             (comp_zp * ur_z - comp_zp * ur_z) * inv_2dz;

            // Time advancement with TVD clamping [0.0, 1.0]
            float alpha_new = alpha_c - dt * (adv_alpha + div_comp);
            alpha_new = fminf(fmaxf(alpha_new, 0.0f), 1.0f);

            if (alpha_star != nullptr) alpha_star[out_idx] = alpha_new;

            // Update dynamic mixture density and viscosity
            if (rho_mix != nullptr) {
                rho_mix[out_idx] = alpha_new * c_metrics.rho1 + (1.0f - alpha_new) * c_metrics.rho2;
            }
            if (nu_mix != nullptr) {
                nu_mix[out_idx] = alpha_new * c_metrics.nu1 + (1.0f - alpha_new) * c_metrics.nu2;
            }
        }
    }
}

// ============================================================================
// STAGE 4: FUSED VELOCITY CORRECTOR & MULTIPHASE PROJECTION
// Launch Bounds: (256, 2)
// Projects: u^{n+1} = (u* - dt/rho(alpha) * grad(p)) / (1 + dt*sigma)
// ============================================================================
__global__ void __launch_bounds__(256, 2) fused_velocity_corrector_multiphase_kernel(
    const float* __restrict__ u_star,
    const float* __restrict__ v_star,
    const float* __restrict__ w_star,
    const float* __restrict__ pp,      // Padded pressure
    const float* __restrict__ rho_mix, // Local density (if VoF active)
    const float* __restrict__ sigma,   // Brinkman damping
    float* __restrict__ u_out,
    float* __restrict__ v_out,
    float* __restrict__ w_out,
    int D, int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < W && y < H && z < D) {
        int pad_W = W + 2;
        int pad_H = H + 2;
        int pad_c = (z + 1) * (pad_H * pad_W) + (y + 1) * pad_W + (x + 1);
        int idx = z * (H * W) + y * W + x;

        float inv_2dx = c_metrics.inv_2dx;
        float inv_2dy = c_metrics.inv_2dy;
        float inv_2dz = c_metrics.inv_2dz;
        float dt      = c_metrics.dt;

        // Density: local mixture density if multiphase, else primary rho
        float local_rho = c_metrics.rho;
        if (c_metrics.enable_vof && rho_mix != nullptr) {
            local_rho = fmaxf(rho_mix[idx], 1e-4f);
        }
        float inv_rho = 1.0f / local_rho;

        // Pressure gradients
        float dp_dx = (pp[pad_c + 1] - pp[pad_c - 1]) * inv_2dx;
        float dp_dy = (pp[pad_c + pad_W] - pp[pad_c - pad_W]) * inv_2dy;
        float dp_dz = (pp[pad_c + pad_H * pad_W] - pp[pad_c - pad_H * pad_W]) * inv_2dz;

        // Brinkman damping factor
        float sig = (sigma != nullptr) ? sigma[idx] : 0.0f;
        float solid_damp = 1.0f / (1.0f + dt * sig);

        // Velocity projection
        u_out[idx] = (u_star[idx] - dt * inv_rho * dp_dx) * solid_damp;
        v_out[idx] = (v_star[idx] - dt * inv_rho * dp_dy) * solid_damp;
        w_out[idx] = (w_star[idx] - dt * inv_rho * dp_dz) * solid_damp;
    }
}

#endif /* MULTIPHYSICS_ENGINE_CUH */
