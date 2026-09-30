#include "thapar_cfd_engine.h"
#include "gpu_array_3d.cuh"
#define DEFINE_DOMAIN_METRICS
#include "domain_metrics.cuh"
#include "boundary_manager.cuh"
#include "stencil_engine.cuh"
#include "multigrid_engine.cuh"
#include "multiphysics_engine.cuh"

#include <vector>
#include <cmath>
#include <chrono>
#include <cstring>
#include <string>
#include <sstream>
#include <algorithm>
#include <cctype>

// Precomputes minimum distance to solid domain boundaries for SA turbulence model
__global__ void compute_wall_distance_kernel(
    float* __restrict__ wall_dist,
    float dx, float dy, float dz,
    int D, int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < W && y < H && z < D) {
        float dist_x = fminf((x + 0.5f) * dx, (W - 0.5f - x) * dx);
        float dist_y = fminf((y + 0.5f) * dy, (H - 0.5f - y) * dy);
        float dist_z = fminf((z + 0.5f) * dz, (D - 0.5f - z) * dz);
        float d = fminf(dist_x, fminf(dist_y, dist_z));
        wall_dist[z * (H * W) + y * W + x] = fmaxf(d, 1e-6f);
    }
}

// Element-wise addition: dst[i] += src[i]
__global__ void add_field_kernel(
    float* __restrict__ dst,
    const float* __restrict__ src,
    int total_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_cells) {
        dst[idx] += src[idx];
    }
}

// Remove mean from field to enforce discrete solvability or gauge invariance
__global__ void remove_mean_kernel(
    float* __restrict__ field,
    float mean_val,
    int total_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_cells) {
        field[idx] -= mean_val;
    }
}

// Compute velocity magnitude: mag[i] = sqrt(u^2 + v^2 + w^2)
__global__ void compute_velocity_magnitude_kernel(
    const float* __restrict__ u,
    const float* __restrict__ v,
    const float* __restrict__ w,
    float* __restrict__ mag,
    int total_cells)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total_cells) {
        float uu = u[idx];
        float vv = v[idx];
        float ww = w[idx];
        mag[idx] = sqrtf(uu * uu + vv * vv + ww * ww);
    }
}

// 2D slice extraction across any axis
__global__ void extract_slice_2d_kernel(
    const float* __restrict__ d_src,
    float* __restrict__ d_dst,
    int axis, int slice_idx,
    int W, int H, int D)
{
    if (axis == 0) { // X slice in Y-Z plane, size H x D
        int y = blockIdx.x * blockDim.x + threadIdx.x;
        int z = blockIdx.y * blockDim.y + threadIdx.y;
        if (y < H && z < D) {
            d_dst[z * H + y] = d_src[z * (H * W) + y * W + slice_idx];
        }
    } else if (axis == 1) { // Y slice in X-Z plane, size W x D
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int z = blockIdx.y * blockDim.y + threadIdx.y;
        if (x < W && z < D) {
            d_dst[z * W + x] = d_src[z * (H * W) + slice_idx * W + x];
        }
    } else if (axis == 2) { // Z slice in X-Y plane, size W x H
        int x = blockIdx.x * blockDim.x + threadIdx.x;
        int y = blockIdx.y * blockDim.y + threadIdx.y;
        if (x < W && y < H) {
            d_dst[y * W + x] = d_src[slice_idx * (H * W) + y * W + x];
        }
    }
}

struct thapar_solver_s {
    thapar_config_t config;
    DomainMetrics metrics;
    std::string last_error;

    // 1. Primary Physical Fields (D x H x W)
    GPUArray3D<float> u, v, w, p;

    // 2. Ghost-Padded Boundary Arrays ((D+2) x (H+2) x (W+2))
    GPUArray3D<float> uu, vv, ww, pp;

    // 3. Intermediate Predictor Fields
    GPUArray3D<float> u_star, v_star, w_star;
    GPUArray3D<float> rhs_div;

    // 4. Immersed Boundary Obstacle Mask (sigma)
    GPUArray3D<float> sigma;

    // 5. Multiphysics Fields
    // Thermal Energy (T)
    GPUArray3D<float> T, TT, t_star;
    // Spalart-Allmaras Turbulence
    GPUArray3D<float> nu_tilde, nu_tilde_pad, nu_tilde_star, nu_t, wall_dist;
    // Multiphase Volume of Fluid
    GPUArray3D<float> alpha, alpha_pad, alpha_star, rho_mix, nu_mix;

    // 6. Diagnostics / Scratch Buffers
    GPUArray3D<float> scratch_3d;
    GPUArray3D<float> slice_2d_buf;

    // 7. Multigrid Hierarchy Arrays
    std::vector<GPUArray3D<float>> r_levels;   // Residuals
    std::vector<GPUArray3D<float>> e_levels;   // Error corrections
    std::vector<GPUArray3D<float>> tmp_levels; // Smoother temporary buffers
    std::vector<int> level_d, level_h, level_w;

    // 8. CUDA Execution & Hardware Graph
    cudaStream_t stream = nullptr;
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;
    bool graph_captured = false;

    // 9. Performance & Verification Diagnostics
    double last_step_ms = 0.0;
    float max_divergence = 0.0f;
    float last_pressure_residual = 0.0f;
    float mass_flux_error = 0.0f;
    size_t total_vram_bytes = 0;

    thapar_solver_s(const thapar_config_t* cfg) : config(*cfg) {
        CUDA_CHECK(cudaStreamCreate(&stream));
        init_metrics();
        allocate_resident_memory();
    }

    ~thapar_solver_s() {
        if (graph_exec) cudaGraphExecDestroy(graph_exec);
        if (graph) cudaGraphDestroy(graph);
        if (stream) cudaStreamDestroy(stream);
    }

    void init_metrics() {
        metrics.nx = config.nx;
        metrics.ny = config.ny;
        metrics.nz = config.nz;
        metrics.pad_nx = config.nx + 2;
        metrics.pad_ny = config.ny + 2;
        metrics.pad_nz = config.nz + 2;

        metrics.dx = config.dx;
        metrics.dy = config.dy;
        metrics.dz = config.dz;
        metrics.dt = config.dt;
        metrics.nu = config.nu;
        metrics.rho = config.rho;
        metrics.ub = config.ub;

        metrics.inv_dx = 1.0f / config.dx;
        metrics.inv_dy = 1.0f / config.dy;
        metrics.inv_dz = 1.0f / config.dz;

        metrics.inv_2dx = 0.5f / config.dx;
        metrics.inv_2dy = 0.5f / config.dy;
        metrics.inv_2dz = 0.5f / config.dz;

        metrics.inv_dx2 = 1.0f / (config.dx * config.dx);
        metrics.inv_dy2 = 1.0f / (config.dy * config.dy);
        metrics.inv_dz2 = 1.0f / (config.dz * config.dz);

        metrics.diag = 2.0f * (metrics.inv_dx2 + metrics.inv_dy2 + metrics.inv_dz2);
        metrics.inv_diag = 1.0f / metrics.diag;
        metrics.jacobi_omega = 0.8f; // Damped Jacobi smoothing factor

        metrics.nlevel = config.nlevel > 0 ? config.nlevel : 6;
        metrics.mg_iterations = config.mg_iterations > 0 ? config.mg_iterations : 5;

        // Multiphysics parameters
        metrics.enable_heat = config.enable_heat;
        metrics.enable_turbulence = config.enable_turbulence;
        metrics.enable_vof = config.enable_vof;

        metrics.thermal_diffusivity = config.thermal_diffusivity > 0.0f ? config.thermal_diffusivity : 0.01f;
        metrics.beta_thermal = config.beta_thermal;
        metrics.t_ref = config.t_ref;
        metrics.gx = config.gx;
        metrics.gy = config.gy;
        metrics.gz = config.gz;

        metrics.sa_cb1 = 0.1355f;
        metrics.sa_cb2 = 0.622f;
        metrics.sa_sigma_inv = 1.0f / 0.66667f;
        metrics.sa_cv1_3 = 7.1f * 7.1f * 7.1f;
        metrics.sa_cw2 = 0.3f;
        metrics.sa_cw3 = 2.0f;
        metrics.sa_kappa2 = 0.41f * 0.41f;

        metrics.rho1 = config.rho1 > 0.0f ? config.rho1 : 1000.0f;
        metrics.rho2 = config.rho2 > 0.0f ? config.rho2 : 1.0f;
        metrics.nu1 = config.nu1 > 0.0f ? config.nu1 : 1e-6f;
        metrics.nu2 = config.nu2 > 0.0f ? config.nu2 : 1.5e-5f;
        metrics.sigma_tension = config.sigma_tension;
        metrics.c_alpha = config.c_alpha > 0.0f ? config.c_alpha : 1.0f;

        // Upload to device constant memory
        CUDA_CHECK(cudaMemcpyToSymbolAsync(c_metrics, &metrics, sizeof(DomainMetrics), 0, cudaMemcpyHostToDevice, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    void allocate_resident_memory() {
        int W = config.nx, H = config.ny, D = config.nz;
        int pW = W + 2, pH = H + 2, pD = D + 2;

        // Primary Fields
        u.allocate(W, H, D);
        v.allocate(W, H, D);
        w.allocate(W, H, D);
        p.allocate(W, H, D);

        // Ghost-padded
        uu.allocate(pW, pH, pD);
        vv.allocate(pW, pH, pD);
        ww.allocate(pW, pH, pD);
        pp.allocate(pW, pH, pD);

        // Predictors & Scratch
        u_star.allocate(W, H, D);
        v_star.allocate(W, H, D);
        w_star.allocate(W, H, D);
        rhs_div.allocate(W, H, D);
        sigma.allocate(W, H, D);
        scratch_3d.allocate(W, H, D);

        // Thermal Fields
        T.allocate(W, H, D);
        TT.allocate(pW, pH, pD);
        t_star.allocate(W, H, D);

        // SA Turbulence Fields
        nu_tilde.allocate(W, H, D);
        nu_tilde_pad.allocate(pW, pH, pD);
        nu_tilde_star.allocate(W, H, D);
        nu_t.allocate(W, H, D);
        wall_dist.allocate(W, H, D);

        // Multiphase VoF Fields
        alpha.allocate(W, H, D);
        alpha_pad.allocate(pW, pH, pD);
        alpha_star.allocate(W, H, D);
        rho_mix.allocate(W, H, D);
        nu_mix.allocate(W, H, D);

        // Precompute normal wall distance field
        dim3 b3d(16, 8, 2);
        dim3 g3d((W + 15) / 16, (H + 7) / 8, (D + 1) / 2);
        compute_wall_distance_kernel<<<g3d, b3d, 0, stream>>>(
            wall_dist.data(), config.dx, config.dy, config.dz, D, H, W);

        // Max 2D slice buffer (max of W*H, W*D, H*D)
        int max_slice = std::max({W * H, W * D, H * D});
        slice_2d_buf.allocate(max_slice, 1, 1);

        // Multigrid levels
        int cur_w = W, cur_h = H, cur_d = D;
        total_vram_bytes = 0;

        for (int l = 0; l < metrics.nlevel; ++l) {
            level_w.push_back(cur_w);
            level_h.push_back(cur_h);
            level_d.push_back(cur_d);

            GPUArray3D<float> r(cur_w, cur_h, cur_d);
            GPUArray3D<float> e(cur_w, cur_h, cur_d);
            GPUArray3D<float> tmp(cur_w, cur_h, cur_d);

            total_vram_bytes += r.get_bytes() + e.get_bytes() + tmp.get_bytes();

            r_levels.push_back(std::move(r));
            e_levels.push_back(std::move(e));
            tmp_levels.push_back(std::move(tmp));

            cur_w = std::max(2, cur_w / 2);
            cur_h = std::max(2, cur_h / 2);
            cur_d = std::max(2, cur_d / 2);
        }

        total_vram_bytes += (u.get_bytes() * 4 + uu.get_bytes() * 4 + u_star.get_bytes() * 6 +
                             T.get_bytes() * 3 + nu_tilde.get_bytes() * 5 + alpha.get_bytes() * 5 +
                             slice_2d_buf.get_bytes());
    }

    void launch_boundary_conditions() {
        int W = config.nx, H = config.ny, D = config.nz;
        dim3 block3d(16, 8, 2);
        dim3 grid3d((W + 15) / 16, (H + 7) / 8, (D + 1) / 2);

        // 1. Copy interior to padded
        copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(u.data(), uu.data(), D, H, W);
        copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(v.data(), vv.data(), D, H, W);
        copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(w.data(), ww.data(), D, H, W);
        copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(p.data(), pp.data(), D, H, W);

        // 2. Apply physical boundary patches (Enforces strict normal no-penetration u_n=0)
        int threads = 256;
        int blocks = (std::max({(D + 2) * (H + 2), (D + 2) * (W + 2), (H + 2) * (W + 2)}) + threads - 1) / threads;

        apply_patch_boundaries_velocity_kernel<<<blocks, threads, 0, stream>>>(
            uu.data(), 0, config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W);

        apply_patch_boundaries_velocity_kernel<<<blocks, threads, 0, stream>>>(
            vv.data(), 1, config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W);

        apply_patch_boundaries_velocity_kernel<<<blocks, threads, 0, stream>>>(
            ww.data(), 2, config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W);

        apply_patch_boundaries_pressure_kernel<<<blocks, threads, 0, stream>>>(
            pp.data(), config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W);

        // 3. Scalar boundaries
        if (config.enable_heat) {
            copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(T.data(), TT.data(), D, H, W);
            apply_patch_boundaries_scalar_kernel<<<blocks, threads, 0, stream>>>(
                TT.data(), config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W, true);
        }
        if (config.enable_turbulence) {
            copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(nu_tilde.data(), nu_tilde_pad.data(), D, H, W);
            apply_patch_boundaries_scalar_kernel<<<blocks, threads, 0, stream>>>(
                nu_tilde_pad.data(), config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W, false);
        }
        if (config.enable_vof) {
            copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(alpha.data(), alpha_pad.data(), D, H, W);
            apply_patch_boundaries_scalar_kernel<<<blocks, threads, 0, stream>>>(
                alpha_pad.data(), config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W, false);
        }
    }

    void execute_single_step_pipeline() {
        int W = config.nx, H = config.ny, D = config.nz;
        dim3 block3d(16, 8, 2);
        dim3 grid3d((W + 15) / 16, (H + 7) / 8, (D + 1) / 2);

        // Step 1: Update boundaries on primary state (u, v, w, p, scalars)
        launch_boundary_conditions();

        // Step 2: Stage 1 - Fused Hydrodynamic & Thermal Predictor
        fused_hydrodynamic_thermal_predictor_kernel<<<grid3d, block3d, 0, stream>>>(
            uu.data(), vv.data(), ww.data(),
            config.enable_heat ? TT.data() : nullptr,
            config.enable_turbulence ? nu_t.data() : nullptr,
            sigma.data(),
            u_star.data(), v_star.data(), w_star.data(),
            config.enable_heat ? t_star.data() : nullptr,
            D, H, W);

        if (config.enable_heat) {
            CUDA_CHECK(cudaMemcpyAsync(T.data(), t_star.data(), T.get_bytes(), cudaMemcpyDeviceToDevice, stream));
        }

        // Step 3: Apply boundaries to intermediate velocity u*
        copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(u_star.data(), uu.data(), D, H, W);
        copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(v_star.data(), vv.data(), D, H, W);
        copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(w_star.data(), ww.data(), D, H, W);

        int threads = 256;
        int blocks = (std::max({(D + 2) * (H + 2), (D + 2) * (W + 2), (H + 2) * (W + 2)}) + threads - 1) / threads;
        apply_patch_boundaries_velocity_kernel<<<blocks, threads, 0, stream>>>(
            uu.data(), 0, config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W);
        apply_patch_boundaries_velocity_kernel<<<blocks, threads, 0, stream>>>(
            vv.data(), 1, config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W);
        apply_patch_boundaries_velocity_kernel<<<blocks, threads, 0, stream>>>(
            ww.data(), 2, config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W);

        // Step 4: Evaluate Divergence RHS: b = (rho / dt) * div(u*)
        compute_divergence_rhs_kernel<<<grid3d, block3d, 0, stream>>>(
            uu.data(), vv.data(), ww.data(), rhs_div.data(), D, H, W);

        // Step 5: Geometric Multigrid (GMG) V-Cycles for Pressure Poisson (Stage 2)
        execute_multigrid_solver();

        // Step 6: Apply boundaries to converged pressure
        copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(p.data(), pp.data(), D, H, W);
        apply_patch_boundaries_pressure_kernel<<<blocks, threads, 0, stream>>>(
            pp.data(), config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W);

        // Step 7: Divergence-free Velocity Projection (Stage 4)
        fused_velocity_corrector_multiphase_kernel<<<grid3d, block3d, 0, stream>>>(
            u_star.data(), v_star.data(), w_star.data(), pp.data(),
            config.enable_vof ? rho_mix.data() : nullptr,
            sigma.data(),
            u.data(), v.data(), w.data(), D, H, W);

        // Step 8: Stage 3 - Fused Scalar Transport Predictor (Spalart-Allmaras & VoF)
        if (config.enable_turbulence || config.enable_vof) {
            // Apply velocity boundaries on corrected velocity u for scalar advection
            copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(u.data(), uu.data(), D, H, W);
            copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(v.data(), vv.data(), D, H, W);
            copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(w.data(), ww.data(), D, H, W);

            apply_patch_boundaries_velocity_kernel<<<blocks, threads, 0, stream>>>(
                uu.data(), 0, config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W);
            apply_patch_boundaries_velocity_kernel<<<blocks, threads, 0, stream>>>(
                vv.data(), 1, config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W);
            apply_patch_boundaries_velocity_kernel<<<blocks, threads, 0, stream>>>(
                ww.data(), 2, config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W);

            if (config.enable_turbulence) {
                copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(nu_tilde.data(), nu_tilde_pad.data(), D, H, W);
                apply_patch_boundaries_scalar_kernel<<<blocks, threads, 0, stream>>>(
                    nu_tilde_pad.data(), config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W, false);
            }
            if (config.enable_vof) {
                copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(alpha.data(), alpha_pad.data(), D, H, W);
                apply_patch_boundaries_scalar_kernel<<<blocks, threads, 0, stream>>>(
                    alpha_pad.data(), config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W, false);
            }

            fused_scalar_transport_predictor_kernel<<<grid3d, block3d, 0, stream>>>(
                uu.data(), vv.data(), ww.data(),
                config.enable_turbulence ? nu_tilde_pad.data() : nullptr,
                config.enable_vof ? alpha_pad.data() : nullptr,
                config.enable_turbulence ? wall_dist.data() : nullptr,
                config.enable_turbulence ? nu_tilde_star.data() : nullptr,
                config.enable_turbulence ? nu_t.data() : nullptr,
                config.enable_vof ? alpha_star.data() : nullptr,
                config.enable_vof ? rho_mix.data() : nullptr,
                config.enable_vof ? nu_mix.data() : nullptr,
                D, H, W);

            if (config.enable_turbulence) {
                CUDA_CHECK(cudaMemcpyAsync(nu_tilde.data(), nu_tilde_star.data(), nu_tilde.get_bytes(), cudaMemcpyDeviceToDevice, stream));
            }
            if (config.enable_vof) {
                CUDA_CHECK(cudaMemcpyAsync(alpha.data(), alpha_star.data(), alpha.get_bytes(), cudaMemcpyDeviceToDevice, stream));
            }
        }
    }

    void execute_multigrid_solver() {
        int W = config.nx, H = config.ny, D = config.nz;
        int nlevels = metrics.nlevel;

        float inv_dx2 = metrics.inv_dx2;
        float inv_dy2 = metrics.inv_dy2;
        float inv_dz2 = metrics.inv_dz2;
        float inv_diag = metrics.inv_diag;

        dim3 b0(16, 8, 2);
        dim3 g0((W + 15) / 16, (H + 7) / 8, (D + 1) / 2);
        int total_cells = W * H * D;
        int th = 256;
        int bl = (total_cells + th - 1) / th;

        // 1. Enforce discrete Poisson solvability: sum(rhs) == 0
        {
            std::vector<float> h_rhs(total_cells);
            rhs_div.copy_to_host(h_rhs.data(), stream);
            cudaStreamSynchronize(stream);
            double sum_rhs = 0.0;
            for (int i = 0; i < total_cells; ++i) sum_rhs += h_rhs[i];
            float mean_rhs = static_cast<float>(sum_rhs / total_cells);
            remove_mean_kernel<<<bl, th, 0, stream>>>(rhs_div.data(), mean_rhs, total_cells);
        }

        for (int iter = 0; iter < metrics.mg_iterations; ++iter) {
            // Level 0: Pre-smoothing on fine pressure (3 Red-Black Gauss-Seidel sweeps)
            for (int s = 0; s < 3; ++s) {
                mg_red_black_gauss_seidel_kernel<<<g0, b0, 0, stream>>>(
                    p.data(), rhs_div.data(), inv_dx2, inv_dy2, inv_dz2, inv_diag, 0, D, H, W);
                mg_red_black_gauss_seidel_kernel<<<g0, b0, 0, stream>>>(
                    p.data(), rhs_div.data(), inv_dx2, inv_dy2, inv_dz2, inv_diag, 1, D, H, W);
            }

            // Level 0: Compute fine residual r_0 = rhs_div - A*p
            compute_mg_residual_kernel<<<g0, b0, 0, stream>>>(
                p.data(), rhs_div.data(), r_levels[0].data(),
                inv_dx2, inv_dy2, inv_dz2, D, H, W);

            // Downward V-Cycle: Full Weighting Restriction
            float cur_inv_dx2 = inv_dx2, cur_inv_dy2 = inv_dy2, cur_inv_dz2 = inv_dz2;
            for (int l = 1; l < nlevels; ++l) {
                int cW = level_w[l], cH = level_h[l], cD = level_d[l];
                int fW = level_w[l - 1], fH = level_h[l - 1], fD = level_d[l - 1];

                dim3 b_c(16, 8, 2);
                dim3 g_c((cW + 15) / 16, (cH + 7) / 8, (cD + 1) / 2);

                restriction_3d_cell_average_kernel<<<g_c, b_c, 0, stream>>>(
                    r_levels[l - 1].data(), r_levels[l].data(),
                    cD, cH, cW, fD, fH, fW);

                cur_inv_dx2 *= 0.25f; // dx doubles on coarse level
                cur_inv_dy2 *= 0.25f;
                cur_inv_dz2 *= 0.25f;
            }

            // Coarsest Level: Exact Conjugate Gradient Solver
            int l_c = nlevels - 1;
            int cW = level_w[l_c], cH = level_h[l_c], cD = level_d[l_c];
            size_t shared_bytes = (cD * cH * cW * 4 + 32) * sizeof(float);
            coarse_grid_exact_cg_kernel<<<1, 128, shared_bytes, stream>>>(
                r_levels[l_c].data(), e_levels[l_c].data(),
                cur_inv_dx2, cur_inv_dy2, cur_inv_dz2, cD, cH, cW, 50, 1e-5f);

            // Upward V-Cycle: Trilinear Prolongation + RBGS Smoothing
            for (int l = nlevels - 2; l >= 0; --l) {
                int fW = level_w[l], fH = level_h[l], fD = level_d[l];
                int cW = level_w[l + 1], cH = level_h[l + 1], cD = level_d[l + 1];

                cur_inv_dx2 *= 4.0f;
                cur_inv_dy2 *= 4.0f;
                cur_inv_dz2 *= 4.0f;
                float l_inv_diag = 1.0f / (2.0f * (cur_inv_dx2 + cur_inv_dy2 + cur_inv_dz2));

                dim3 b_f(16, 8, 2);
                dim3 g_f((fW + 15) / 16, (fH + 7) / 8, (fD + 1) / 2);

                // Zero fine error buffer before accumulating prolongation
                e_levels[l].zero(stream);

                // Trilinear Prolongation: e_fine += P(e_coarse)
                prolongation_3d_trilinear_accumulate_kernel<<<g_f, b_f, 0, stream>>>(
                    e_levels[l + 1].data(), e_levels[l].data(),
                    fD, fH, fW, cD, cH, cW);

                // Smoothing passes (3 Red-Black Gauss-Seidel iterations)
                for (int s = 0; s < 3; ++s) {
                    mg_red_black_gauss_seidel_kernel<<<g_f, b_f, 0, stream>>>(
                        e_levels[l].data(), r_levels[l].data(),
                        cur_inv_dx2, cur_inv_dy2, cur_inv_dz2, l_inv_diag, 0, fD, fH, fW);
                    mg_red_black_gauss_seidel_kernel<<<g_f, b_f, 0, stream>>>(
                        e_levels[l].data(), r_levels[l].data(),
                        cur_inv_dx2, cur_inv_dy2, cur_inv_dz2, l_inv_diag, 1, fD, fH, fW);
                }
            }

            // Correct fine pressure: p = p + e_0 (multigrid coarse-grid correction)
            add_field_kernel<<<bl, th, 0, stream>>>(p.data(), e_levels[0].data(), total_cells);

            // Post-smoothing pass on fine pressure (3 Red-Black Gauss-Seidel sweeps)
            for (int s = 0; s < 3; ++s) {
                mg_red_black_gauss_seidel_kernel<<<g0, b0, 0, stream>>>(
                    p.data(), rhs_div.data(), inv_dx2, inv_dy2, inv_dz2, inv_diag, 0, D, H, W);
                mg_red_black_gauss_seidel_kernel<<<g0, b0, 0, stream>>>(
                    p.data(), rhs_div.data(), inv_dx2, inv_dy2, inv_dz2, inv_diag, 1, D, H, W);
            }
        }

        // Remove gauge drift (keep p bounded with zero mean)
        {
            std::vector<float> h_p(total_cells);
            p.copy_to_host(h_p.data(), stream);
            cudaStreamSynchronize(stream);
            double sum_p = 0.0;
            for (int i = 0; i < total_cells; ++i) sum_p += h_p[i];
            float mean_p = static_cast<float>(sum_p / total_cells);
            remove_mean_kernel<<<bl, th, 0, stream>>>(p.data(), mean_p, total_cells);
        }
    }

    void compute_diagnostics_internal() {
        int W = config.nx, H = config.ny, D = config.nz;
        int total = W * H * D;
        dim3 block3d(16, 8, 2);
        dim3 grid3d((W + 15) / 16, (H + 7) / 8, (D + 1) / 2);

        // 1. Copy u, v, w to padded to evaluate div(u)
        copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(u.data(), uu.data(), D, H, W);
        copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(v.data(), vv.data(), D, H, W);
        copy_interior_to_padded_kernel<<<grid3d, block3d, 0, stream>>>(w.data(), ww.data(), D, H, W);

        int threads = 256;
        int blocks = (std::max({(D + 2) * (H + 2), (D + 2) * (W + 2), (H + 2) * (W + 2)}) + threads - 1) / threads;
        apply_patch_boundaries_velocity_kernel<<<blocks, threads, 0, stream>>>(
            uu.data(), 0, config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W);
        apply_patch_boundaries_velocity_kernel<<<blocks, threads, 0, stream>>>(
            vv.data(), 1, config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W);
        apply_patch_boundaries_velocity_kernel<<<blocks, threads, 0, stream>>>(
            ww.data(), 2, config.bc_xmin, config.bc_xmax, config.bc_ymin, config.bc_ymax, config.bc_zmin, config.bc_zmax, D, H, W);

        compute_divergence_rhs_kernel<<<grid3d, block3d, 0, stream>>>(
            uu.data(), vv.data(), ww.data(), scratch_3d.data(), D, H, W);

        std::vector<float> h_div(total);
        scratch_3d.copy_to_host(h_div.data(), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        float max_div = 0.0f;
        float factor = metrics.dt / metrics.rho;
        for (int i = 0; i < total; ++i) {
            float div = std::abs(h_div[i] * factor);
            if (div > max_div) max_div = div;
        }
        max_divergence = max_div;

        // 2. Pressure residual
        std::vector<float> h_res(total);
        r_levels[0].copy_to_host(h_res.data(), stream);
        CUDA_CHECK(cudaStreamSynchronize(stream));

        double sum_sq = 0.0;
        for (int i = 0; i < total; ++i) {
            sum_sq += static_cast<double>(h_res[i]) * h_res[i];
        }
        last_pressure_residual = static_cast<float>(std::sqrt(sum_sq / total));
    }

    void step() {
        auto t0 = std::chrono::high_resolution_clock::now();

        if (config.enable_cuda_graph) {
            if (!graph_captured) {
                // Warmup pass
                execute_single_step_pipeline();
                CUDA_CHECK(cudaStreamSynchronize(stream));

                // Capture into hardware CUDA Graph
                CUDA_CHECK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
                execute_single_step_pipeline();
                CUDA_CHECK(cudaStreamEndCapture(stream, &graph));
                CUDA_CHECK(cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0));
                graph_captured = true;
            }
            CUDA_CHECK(cudaGraphLaunch(graph_exec, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
        } else {
            execute_single_step_pipeline();
            CUDA_CHECK(cudaStreamSynchronize(stream));
        }

        auto t1 = std::chrono::high_resolution_clock::now();
        last_step_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    }
};

// Simple JSON extraction helper
static std::string extract_json_value(const std::string& json, const std::string& key) {
    size_t pos = json.find("\"" + key + "\"");
    if (pos == std::string::npos) return "";
    pos = json.find(':', pos);
    if (pos == std::string::npos) return "";
    pos++;
    while (pos < json.size() && std::isspace(json[pos])) pos++;
    if (pos >= json.size()) return "";

    if (json[pos] == '"') {
        size_t end = json.find('"', pos + 1);
        if (end == std::string::npos) return "";
        return json.substr(pos + 1, end - pos - 1);
    } else {
        size_t end = pos;
        while (end < json.size() && (std::isalnum(json[end]) || json[end] == '.' || json[end] == '-' || json[end] == '+')) end++;
        return json.substr(pos, end - pos);
    }
}

// C ABI Implementation

thapar_solver_t* thapar_create_solver(const thapar_config_t* config) {
    if (!config) return nullptr;
    try {
        return new thapar_solver_s(config);
    } catch (const std::exception& e) {
        fprintf(stderr, "Failed to create Thapar CFD Solver: %s\n", e.what());
        return nullptr;
    }
}

thapar_solver_t* thapar_create_solver_from_json(const char* sim_spec_json) {
    if (!sim_spec_json) return nullptr;
    std::string str(sim_spec_json);
    thapar_config_t cfg;
    std::memset(&cfg, 0, sizeof(cfg));

    auto parse_int = [&](const std::string& key, int default_v) -> int {
        std::string v = extract_json_value(str, key);
        return v.empty() ? default_v : std::atoi(v.c_str());
    };
    auto parse_float = [&](const std::string& key, float default_v) -> float {
        std::string v = extract_json_value(str, key);
        return v.empty() ? default_v : static_cast<float>(std::atof(v.c_str()));
    };

    cfg.nx = parse_int("nx", 128);
    cfg.ny = parse_int("ny", 128);
    cfg.nz = parse_int("nz", 128);
    cfg.dx = parse_float("dx", 1.0f / cfg.nx);
    cfg.dy = parse_float("dy", 1.0f / cfg.ny);
    cfg.dz = parse_float("dz", 1.0f / cfg.nz);
    cfg.dt = parse_float("dt", 0.001f);
    cfg.nu = parse_float("nu", 0.01f);
    cfg.rho = parse_float("rho", 1.0f);
    cfg.ub = parse_float("ub", 1.0f);
    cfg.nlevel = parse_int("nlevel", 5);
    cfg.mg_iterations = parse_int("mg_iterations", 5);
    cfg.enable_cuda_graph = parse_int("enable_cuda_graph", 0);
    cfg.enable_heat = parse_int("enable_heat", 0);
    cfg.enable_turbulence = parse_int("enable_turbulence", 0);
    cfg.enable_vof = parse_int("enable_vof", 0);
    cfg.thermal_diffusivity = parse_float("thermal_diffusivity", 0.01f);
    cfg.beta_thermal = parse_float("beta_thermal", 0.003f);
    cfg.t_ref = parse_float("t_ref", 0.0f);
    cfg.gx = parse_float("gx", 0.0f);
    cfg.gy = parse_float("gy", 0.0f);
    cfg.gz = parse_float("gz", 0.0f);
    cfg.rho1 = parse_float("rho1", 1000.0f);
    cfg.rho2 = parse_float("rho2", 1.0f);
    cfg.nu1 = parse_float("nu1", 1e-6f);
    cfg.nu2 = parse_float("nu2", 1.5e-5f);
    cfg.sigma_tension = parse_float("sigma_tension", 0.07f);
    cfg.c_alpha = parse_float("c_alpha", 1.0f);

    // Per-face boundary conditions: each of the 6 outer domain patches
    // (xmin, xmax, ymin, ymax, zmin, zmax) may be independently specified
    // from the sim_spec IR as "bc_<face>_type" (one of no_slip_wall,
    // free_slip_symmetry, dirichlet_inflow, neumann_outflow, moving_lid,
    // periodic, isothermal_wall) plus "bc_<face>_u/v/w/p/t" values. Faces
    // not present in the IR fall back to the classic lid-driven-cavity
    // default (Z-Max moving lid, all other five no-slip) so existing
    // callers/tests are unaffected.
    auto parse_bc_type = [&](const std::string& key, thapar_bc_type_t default_v) -> thapar_bc_type_t {
        std::string v = extract_json_value(str, key);
        if (v.empty()) return default_v;
        if (v == "no_slip_wall") return THAPAR_BC_NO_SLIP_WALL;
        if (v == "free_slip_symmetry") return THAPAR_BC_FREE_SLIP_SYMMETRY;
        if (v == "dirichlet_inflow") return THAPAR_BC_DIRICHLET_INFLOW;
        if (v == "neumann_outflow") return THAPAR_BC_NEUMANN_OUTFLOW;
        if (v == "moving_lid") return THAPAR_BC_MOVING_LID;
        if (v == "periodic") return THAPAR_BC_PERIODIC;
        if (v == "isothermal_wall") return THAPAR_BC_ISOTHERMAL_WALL;
        return default_v;
    };
    auto parse_bc = [&](const std::string& face, thapar_bc_type_t default_type,
                         float default_u, float default_v, float default_w) -> thapar_patch_bc_t {
        thapar_patch_bc_t bc;
        std::memset(&bc, 0, sizeof(bc));
        bc.type = parse_bc_type("bc_" + face + "_type", default_type);
        bc.u_val = parse_float("bc_" + face + "_u", default_u);
        bc.v_val = parse_float("bc_" + face + "_v", default_v);
        bc.w_val = parse_float("bc_" + face + "_w", default_w);
        bc.p_val = parse_float("bc_" + face + "_p", 0.0f);
        bc.t_val = parse_float("bc_" + face + "_t", cfg.t_ref);
        return bc;
    };

    // Default: Lid-driven cavity (Z-Max moving lid, all other no-slip)
    cfg.bc_xmin = parse_bc("xmin", THAPAR_BC_NO_SLIP_WALL, 0.0f, 0.0f, 0.0f);
    cfg.bc_xmax = parse_bc("xmax", THAPAR_BC_NO_SLIP_WALL, 0.0f, 0.0f, 0.0f);
    cfg.bc_ymin = parse_bc("ymin", THAPAR_BC_NO_SLIP_WALL, 0.0f, 0.0f, 0.0f);
    cfg.bc_ymax = parse_bc("ymax", THAPAR_BC_NO_SLIP_WALL, 0.0f, 0.0f, 0.0f);
    cfg.bc_zmin = parse_bc("zmin", THAPAR_BC_NO_SLIP_WALL, 0.0f, 0.0f, 0.0f);
    cfg.bc_zmax = parse_bc("zmax", THAPAR_BC_MOVING_LID, cfg.ub, 0.0f, 0.0f);

    return thapar_create_solver(&cfg);
}

thapar_status_t thapar_step(thapar_solver_t* solver) {
    if (!solver) return THAPAR_STATUS_INVALID_ARGUMENT;
    try {
        solver->step();
        return THAPAR_STATUS_SUCCESS;
    } catch (const std::exception& e) {
        solver->last_error = e.what();
        return THAPAR_STATUS_CUDA_ERROR;
    }
}

thapar_status_t thapar_step_multiple(thapar_solver_t* solver, int num_steps) {
    if (!solver || num_steps <= 0) return THAPAR_STATUS_INVALID_ARGUMENT;
    for (int i = 0; i < num_steps; ++i) {
        thapar_status_t status = thapar_step(solver);
        if (status != THAPAR_STATUS_SUCCESS) return status;
    }
    return THAPAR_STATUS_SUCCESS;
}

thapar_status_t thapar_get_slice(
    thapar_solver_t* solver,
    thapar_field_type_t field_type,
    thapar_axis_t axis,
    int index,
    float* out_buffer,
    size_t buffer_size)
{
    if (!solver || !out_buffer) return THAPAR_STATUS_INVALID_ARGUMENT;

    int W = solver->config.nx;
    int H = solver->config.ny;
    int D = solver->config.nz;
    int total = W * H * D;

    float* d_src = nullptr;
    if (field_type == THAPAR_FIELD_U) d_src = solver->u.data();
    else if (field_type == THAPAR_FIELD_V) d_src = solver->v.data();
    else if (field_type == THAPAR_FIELD_W) d_src = solver->w.data();
    else if (field_type == THAPAR_FIELD_PRESSURE) d_src = solver->p.data();
    else if (field_type == THAPAR_FIELD_TEMPERATURE) d_src = solver->T.data();
    else if (field_type == THAPAR_FIELD_TURBULENT_VISCOSITY) d_src = solver->nu_t.data();
    else if (field_type == THAPAR_FIELD_VOF_ALPHA) d_src = solver->alpha.data();
    else if (field_type == THAPAR_FIELD_OBSTACLE_MASK) d_src = solver->sigma.data();
    else if (field_type == THAPAR_FIELD_VELOCITY_MAGNITUDE) {
        int th = 256;
        int bl = (total + th - 1) / th;
        compute_velocity_magnitude_kernel<<<bl, th, 0, solver->stream>>>(
            solver->u.data(), solver->v.data(), solver->w.data(), solver->scratch_3d.data(), total);
        d_src = solver->scratch_3d.data();
    } else return THAPAR_STATUS_INVALID_ARGUMENT;

    // Check slice bounds and sizes
    size_t slice_elements = 0;
    dim3 b2d(16, 16);
    dim3 g2d(1, 1);

    if (axis == THAPAR_AXIS_X) { // Y-Z plane: H x D
        if (index < 0 || index >= W) return THAPAR_STATUS_INVALID_ARGUMENT;
        slice_elements = static_cast<size_t>(H) * D;
        g2d = dim3((H + 15) / 16, (D + 15) / 16);
    } else if (axis == THAPAR_AXIS_Y) { // X-Z plane: W x D
        if (index < 0 || index >= H) return THAPAR_STATUS_INVALID_ARGUMENT;
        slice_elements = static_cast<size_t>(W) * D;
        g2d = dim3((W + 15) / 16, (D + 15) / 16);
    } else if (axis == THAPAR_AXIS_Z) { // X-Y plane: W x H
        if (index < 0 || index >= D) return THAPAR_STATUS_INVALID_ARGUMENT;
        slice_elements = static_cast<size_t>(W) * H;
        g2d = dim3((W + 15) / 16, (H + 15) / 16);
    } else return THAPAR_STATUS_INVALID_ARGUMENT;

    if (buffer_size < slice_elements * sizeof(float)) return THAPAR_STATUS_INVALID_ARGUMENT;

    // Fast-path for contiguous Z slice
    if (axis == THAPAR_AXIS_Z && field_type != THAPAR_FIELD_VELOCITY_MAGNITUDE) {
        size_t offset = static_cast<size_t>(index) * (W * H);
        CUDA_CHECK(cudaMemcpyAsync(out_buffer, d_src + offset, slice_elements * sizeof(float), cudaMemcpyDeviceToHost, solver->stream));
        CUDA_CHECK(cudaStreamSynchronize(solver->stream));
        return THAPAR_STATUS_SUCCESS;
    }

    // Kernel extraction into device 2D scratch buffer then copy to host
    extract_slice_2d_kernel<<<g2d, b2d, 0, solver->stream>>>(
        d_src, solver->slice_2d_buf.data(), static_cast<int>(axis), index, W, H, D);
    CUDA_CHECK(cudaMemcpyAsync(out_buffer, solver->slice_2d_buf.data(), slice_elements * sizeof(float), cudaMemcpyDeviceToHost, solver->stream));
    CUDA_CHECK(cudaStreamSynchronize(solver->stream));

    return THAPAR_STATUS_SUCCESS;
}

thapar_status_t thapar_get_field_3d(
    thapar_solver_t* solver,
    thapar_field_type_t field_type,
    float* out_buffer,
    size_t buffer_size)
{
    if (!solver || !out_buffer) return THAPAR_STATUS_INVALID_ARGUMENT;
    size_t total_elements = static_cast<size_t>(solver->config.nx) * solver->config.ny * solver->config.nz;
    if (buffer_size < total_elements * sizeof(float)) return THAPAR_STATUS_INVALID_ARGUMENT;

    float* d_src = nullptr;
    if (field_type == THAPAR_FIELD_U) d_src = solver->u.data();
    else if (field_type == THAPAR_FIELD_V) d_src = solver->v.data();
    else if (field_type == THAPAR_FIELD_W) d_src = solver->w.data();
    else if (field_type == THAPAR_FIELD_PRESSURE) d_src = solver->p.data();
    else if (field_type == THAPAR_FIELD_TEMPERATURE) d_src = solver->T.data();
    else if (field_type == THAPAR_FIELD_TURBULENT_VISCOSITY) d_src = solver->nu_t.data();
    else if (field_type == THAPAR_FIELD_VOF_ALPHA) d_src = solver->alpha.data();
    else if (field_type == THAPAR_FIELD_OBSTACLE_MASK) d_src = solver->sigma.data();
    else return THAPAR_STATUS_INVALID_ARGUMENT;

    CUDA_CHECK(cudaMemcpyAsync(out_buffer, d_src, total_elements * sizeof(float), cudaMemcpyDeviceToHost, solver->stream));
    CUDA_CHECK(cudaStreamSynchronize(solver->stream));
    return THAPAR_STATUS_SUCCESS;
}

thapar_status_t thapar_set_field_3d(
    thapar_solver_t* solver,
    thapar_field_type_t field_type,
    const float* host_buffer,
    size_t count)
{
    if (!solver || !host_buffer) return THAPAR_STATUS_INVALID_ARGUMENT;
    size_t total_elements = static_cast<size_t>(solver->config.nx) * solver->config.ny * solver->config.nz;
    if (count != total_elements) return THAPAR_STATUS_INVALID_ARGUMENT;

    float* d_dst = nullptr;
    if (field_type == THAPAR_FIELD_U) d_dst = solver->u.data();
    else if (field_type == THAPAR_FIELD_V) d_dst = solver->v.data();
    else if (field_type == THAPAR_FIELD_W) d_dst = solver->w.data();
    else if (field_type == THAPAR_FIELD_PRESSURE) d_dst = solver->p.data();
    else if (field_type == THAPAR_FIELD_TEMPERATURE) d_dst = solver->T.data();
    else if (field_type == THAPAR_FIELD_TURBULENT_VISCOSITY) d_dst = solver->nu_t.data();
    else if (field_type == THAPAR_FIELD_VOF_ALPHA) d_dst = solver->alpha.data();
    else if (field_type == THAPAR_FIELD_OBSTACLE_MASK) d_dst = solver->sigma.data();
    else return THAPAR_STATUS_INVALID_ARGUMENT;

    CUDA_CHECK(cudaMemcpyAsync(d_dst, host_buffer, total_elements * sizeof(float), cudaMemcpyHostToDevice, solver->stream));
    CUDA_CHECK(cudaStreamSynchronize(solver->stream));
    return THAPAR_STATUS_SUCCESS;
}

thapar_status_t thapar_set_obstacle_mask(
    thapar_solver_t* solver,
    const float* host_mask,
    size_t count)
{
    if (!solver || !host_mask) return THAPAR_STATUS_INVALID_ARGUMENT;
    size_t total = static_cast<size_t>(solver->config.nx) * solver->config.ny * solver->config.nz;
    if (count != total) return THAPAR_STATUS_INVALID_ARGUMENT;

    solver->sigma.copy_from_host(host_mask, solver->stream);
    CUDA_CHECK(cudaStreamSynchronize(solver->stream));
    return THAPAR_STATUS_SUCCESS;
}

thapar_status_t thapar_get_diagnostics(
    thapar_solver_t* solver,
    float* out_max_divergence,
    float* out_pressure_residual,
    float* out_mass_flux_error,
    size_t* out_vram_allocated_bytes,
    double* out_last_step_ms)
{
    if (!solver) return THAPAR_STATUS_INVALID_ARGUMENT;
    try {
        solver->compute_diagnostics_internal();
    } catch (...) {}

    if (out_max_divergence) *out_max_divergence = solver->max_divergence;
    if (out_pressure_residual) *out_pressure_residual = solver->last_pressure_residual;
    if (out_mass_flux_error) *out_mass_flux_error = solver->mass_flux_error;
    if (out_vram_allocated_bytes) *out_vram_allocated_bytes = solver->total_vram_bytes;
    if (out_last_step_ms) *out_last_step_ms = solver->last_step_ms;
    return THAPAR_STATUS_SUCCESS;
}

void thapar_destroy_solver(thapar_solver_t* solver) {
    if (solver) {
        delete solver;
    }
}

const char* thapar_get_last_error(thapar_solver_t* solver) {
    if (!solver) return "Null solver pointer";
    return solver->last_error.c_str();
}
