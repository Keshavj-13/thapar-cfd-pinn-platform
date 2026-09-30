#ifndef MULTIGRID_ENGINE_CUH
#define MULTIGRID_ENGINE_CUH

#include <cuda_runtime.h>
#include "domain_metrics.cuh"

// 3D 27-Point Full Weighting Restriction Kernel (r_coarse = R * r_fine)
// Evaluates 2x2x2 fine cell average into each coarse cell
__global__ void restriction_3d_cell_average_kernel(
    const float* __restrict__ fine,
    float* __restrict__ coarse,
    int cD, int cH, int cW,
    int fD, int fH, int fW)
{
    int cx = blockIdx.x * blockDim.x + threadIdx.x;
    int cy = blockIdx.y * blockDim.y + threadIdx.y;
    int cz = blockIdx.z * blockDim.z + threadIdx.z;

    if (cx < cW && cy < cH && cz < cD) {
        int fx = cx * 2;
        int fy = cy * 2;
        int fz = cz * 2;

        float sum = 0.0f;
        #pragma unroll
        for (int kz = 0; kz < 2; ++kz) {
            int iz = fz + kz;
            if (iz < fD) {
                #pragma unroll
                for (int ky = 0; ky < 2; ++ky) {
                    int iy = fy + ky;
                    if (iy < fH) {
                        #pragma unroll
                        for (int kx = 0; kx < 2; ++kx) {
                            int ix = fx + kx;
                            if (ix < fW) {
                                sum += fine[iz * (fH * fW) + iy * fW + ix];
                            }
                        }
                    }
                }
            }
        }
        int c_idx = cz * (cH * cW) + cy * cW + cx;
        coarse[c_idx] = sum * 0.125f; // Box filter (conservation preserving)
    }
}

// 3D Trilinear Prolongation Kernel with Accumulative Correction:
// fine = fine + P(coarse)
// Order m_p = 2, preserving linear error fields
__global__ void prolongation_3d_trilinear_accumulate_kernel(
    const float* __restrict__ coarse,
    float* __restrict__ fine,
    int fD, int fH, int fW,
    int cD, int cH, int cW)
{
    int fx = blockIdx.x * blockDim.x + threadIdx.x;
    int fy = blockIdx.y * blockDim.y + threadIdx.y;
    int fz = blockIdx.z * blockDim.z + threadIdx.z;

    if (fx < fW && fy < fH && fz < fD) {
        // Continuous coordinate in coarse cell units
        // Coarse cell centers are at c + 0.5. Fine centers are at (f + 0.5)/2.
        float cx_f = (fx + 0.5f) * 0.5f - 0.5f;
        float cy_f = (fy + 0.5f) * 0.5f - 0.5f;
        float cz_f = (fz + 0.5f) * 0.5f - 0.5f;

        int c0_x = floorf(cx_f);
        int c0_y = floorf(cy_f);
        int c0_z = floorf(cz_f);

        int c1_x = c0_x + 1;
        int c1_y = c0_y + 1;
        int c1_z = c0_z + 1;

        float wx1 = cx_f - c0_x; float wx0 = 1.0f - wx1;
        float wy1 = cy_f - c0_y; float wy0 = 1.0f - wy1;
        float wz1 = cz_f - c0_z; float wz0 = 1.0f - wz1;

        // Clamp coordinates to coarse domain bounds
        int x0 = max(0, min(cW - 1, c0_x));
        int x1 = max(0, min(cW - 1, c1_x));
        int y0 = max(0, min(cH - 1, c0_y));
        int y1 = max(0, min(cH - 1, c1_y));
        int z0 = max(0, min(cD - 1, c0_z));
        int z1 = max(0, min(cD - 1, c1_z));

        // Sample 8 corners
        float c000 = coarse[z0 * (cH * cW) + y0 * cW + x0];
        float c100 = coarse[z0 * (cH * cW) + y0 * cW + x1];
        float c010 = coarse[z0 * (cH * cW) + y1 * cW + x0];
        float c110 = coarse[z0 * (cH * cW) + y1 * cW + x1];

        float c001 = coarse[z1 * (cH * cW) + y0 * cW + x0];
        float c101 = coarse[z1 * (cH * cW) + y0 * cW + x1];
        float c011 = coarse[z1 * (cH * cW) + y1 * cW + x0];
        float c111 = coarse[z1 * (cH * cW) + y1 * cW + x1];

        // Trilinear interpolation
        float interp = wz0 * (wy0 * (wx0 * c000 + wx1 * c100) + wy1 * (wx0 * c010 + wx1 * c110)) +
                       wz1 * (wy0 * (wx0 * c001 + wx1 * c101) + wy1 * (wx0 * c011 + wx1 * c111));

        // Accumulative correction: fine += P(coarse)
        int f_idx = fz * (fH * fW) + fy * fW + fx;
        fine[f_idx] += interp;
    }
}

// 7-Point Laplacian Smoother with Damped Jacobi (omega = 0.8)
// L(w) = (w_{i+1} - 2w_i + w_{i-1})/dx^2 + ...
__global__ void mg_damped_jacobi_smoother_kernel(
    const float* __restrict__ w_in,
    const float* __restrict__ r_in,
    float* __restrict__ w_out,
    float inv_dx2, float inv_dy2, float inv_dz2,
    float inv_diag, float omega,
    int D, int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < W && y < H && z < D) {
        int idx = z * (H * W) + y * W + x;

        float w_c = w_in[idx];
        float w_xm = (x > 0)     ? w_in[idx - 1]       : w_c; // Homogeneous Neumann
        float w_xp = (x < W - 1) ? w_in[idx + 1]       : w_c;
        float w_ym = (y > 0)     ? w_in[idx - W]       : w_c;
        float w_yp = (y < H - 1) ? w_in[idx + W]       : w_c;
        float w_zm = (z > 0)     ? w_in[idx - (H * W)] : w_c;
        float w_zp = (z < D - 1) ? w_in[idx + (H * W)] : w_c;

        // Positive-definite operator A = -nabla^2
        float Aw = (2.0f * w_c - w_xp - w_xm) * inv_dx2 +
                   (2.0f * w_c - w_yp - w_ym) * inv_dy2 +
                   (2.0f * w_c - w_zp - w_zm) * inv_dz2;

        float r_val = r_in[idx];
        // Damped Jacobi update: w_new = w + omega * (r - Aw) * inv_diag
        w_out[idx] = w_c + omega * (r_val - Aw) * inv_diag;
    }
}

// Highly Optimal 3D Red-Black Gauss-Seidel Smoother (smoothing factor mu = 1/9 = 0.111)
__global__ void mg_red_black_gauss_seidel_kernel(
    float* __restrict__ w,
    const float* __restrict__ r,
    float inv_dx2, float inv_dy2, float inv_dz2,
    float inv_diag, int target_color,
    int D, int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < W && y < H && z < D) {
        if (((x + y + z) & 1) != target_color) return;

        int idx = z * (H * W) + y * W + x;
        float w_c  = w[idx];
        float w_xm = (x > 0)     ? w[idx - 1]       : w_c;
        float w_xp = (x < W - 1) ? w[idx + 1]       : w_c;
        float w_ym = (y > 0)     ? w[idx - W]       : w_c;
        float w_yp = (y < H - 1) ? w[idx + W]       : w_c;
        float w_zm = (z > 0)     ? w[idx - (H * W)] : w_c;
        float w_zp = (z < D - 1) ? w[idx + (H * W)] : w_c;

        float off_diag_sum = (w_xp + w_xm) * inv_dx2 +
                             (w_yp + w_ym) * inv_dy2 +
                             (w_zp + w_zm) * inv_dz2;
        float r_val = r[idx];
        w[idx] = (r_val + off_diag_sum) * inv_diag;
    }
}

// Compute Residual Kernel: r = b - A*w
__global__ void compute_mg_residual_kernel(
    const float* __restrict__ w,
    const float* __restrict__ b,
    float* __restrict__ r,
    float inv_dx2, float inv_dy2, float inv_dz2,
    int D, int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < W && y < H && z < D) {
        int idx = z * (H * W) + y * W + x;

        float w_c  = w[idx];
        float w_xm = (x > 0)     ? w[idx - 1]       : w_c;
        float w_xp = (x < W - 1) ? w[idx + 1]       : w_c;
        float w_ym = (y > 0)     ? w[idx - W]       : w_c;
        float w_yp = (y < H - 1) ? w[idx + W]       : w_c;
        float w_zm = (z > 0)     ? w[idx - (H * W)] : w_c;
        float w_zp = (z < D - 1) ? w[idx + (H * W)] : w_c;

        float Aw = (2.0f * w_c - w_xp - w_xm) * inv_dx2 +
                   (2.0f * w_c - w_yp - w_ym) * inv_dy2 +
                   (2.0f * w_c - w_zp - w_zm) * inv_dz2;

        r[idx] = b[idx] - Aw;
    }
}

// Coarse Grid Conjugate Gradient (CG) Solver Kernel for Level L_coarsest
// Since coarsest grid has small dimensions (e.g. 4x4x4 = 64 cells),
// a single thread block solves A_c * e_c = r_c to exact convergence in shared memory!
__global__ void coarse_grid_exact_cg_kernel(
    const float* __restrict__ r_in,
    float* __restrict__ e_out,
    float inv_dx2, float inv_dy2, float inv_dz2,
    int D, int H, int W,
    int max_iters, float tol)
{
    // Coarsest grid assumed small: total_cells <= 512
    int total_cells = D * H * W;
    int tid = threadIdx.x;

    extern __shared__ float sh_mem[];
    float* sh_p  = sh_mem;                   // size total_cells
    float* sh_r  = sh_p + total_cells;       // size total_cells
    float* sh_Ap = sh_r + total_cells;       // size total_cells
    float* sh_e  = sh_Ap + total_cells;      // size total_cells
    float* sh_red = sh_e + total_cells;      // reduction scratch (size 32)

    // Project out null space mean from r_in
    float sum_r = 0.0f;
    for (int i = tid; i < total_cells; i += blockDim.x) sum_r += r_in[i];
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) sum_r += __shfl_down_sync(0xffffffff, sum_r, offset);
    if (tid % 32 == 0) sh_red[tid / 32] = sum_r;
    __syncthreads();
    if (tid == 0) {
        float total_sum = 0.0f;
        for (int w = 0; w < (blockDim.x + 31) / 32; ++w) total_sum += sh_red[w];
        sh_red[0] = total_sum / total_cells;
    }
    __syncthreads();
    float r_mean = sh_red[0];

    // 1. Initialize e = 0, r = r_in - r_mean, p = r
    for (int i = tid; i < total_cells; i += blockDim.x) {
        float r_val = r_in[i] - r_mean;
        sh_e[i] = 0.0f;
        sh_r[i] = r_val;
        sh_p[i] = r_val;
    }
    __syncthreads();

    // Initial r_dot_r
    float r_dot_r = 0.0f;
    for (int i = tid; i < total_cells; i += blockDim.x) {
        r_dot_r += sh_r[i] * sh_r[i];
    }
    
    // Warp reduce
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        r_dot_r += __shfl_down_sync(0xffffffff, r_dot_r, offset);
    }
    if (tid % 32 == 0) sh_red[tid / 32] = r_dot_r;
    __syncthreads();

    float total_rsold = 0.0f;
    if (tid == 0) {
        for (int w = 0; w < (blockDim.x + 31) / 32; ++w) total_rsold += sh_red[w];
        sh_red[0] = total_rsold;
    }
    __syncthreads();
    total_rsold = sh_red[0];

    if (total_rsold < tol * tol) {
        for (int i = tid; i < total_cells; i += blockDim.x) e_out[i] = 0.0f;
        return;
    }

    // CG Iterations
    for (int iter = 0; iter < max_iters; ++iter) {
        // Evaluate Ap = A * p
        for (int i = tid; i < total_cells; i += blockDim.x) {
            int cz = i / (H * W);
            int rem = i % (H * W);
            int cy = rem / W;
            int cx = rem % W;

            float p_c  = sh_p[i];
            float p_xm = (cx > 0)     ? sh_p[i - 1]       : p_c;
            float p_xp = (cx < W - 1) ? sh_p[i + 1]       : p_c;
            float p_ym = (cy > 0)     ? sh_p[i - W]       : p_c;
            float p_yp = (cy < H - 1) ? sh_p[i + W]       : p_c;
            float p_zm = (cz > 0)     ? sh_p[i - (H * W)] : p_c;
            float p_zp = (cz < D - 1) ? sh_p[i + (H * W)] : p_c;

            sh_Ap[i] = (2.0f * p_c - p_xp - p_xm) * inv_dx2 +
                       (2.0f * p_c - p_yp - p_ym) * inv_dy2 +
                       (2.0f * p_c - p_zp - p_zm) * inv_dz2;
        }
        __syncthreads();

        // p_dot_Ap
        float p_Ap = 0.0f;
        for (int i = tid; i < total_cells; i += blockDim.x) {
            p_Ap += sh_p[i] * sh_Ap[i];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            p_Ap += __shfl_down_sync(0xffffffff, p_Ap, offset);
        }
        if (tid % 32 == 0) sh_red[tid / 32] = p_Ap;
        __syncthreads();

        float total_pAp = 0.0f;
        if (tid == 0) {
            for (int w = 0; w < (blockDim.x + 31) / 32; ++w) total_pAp += sh_red[w];
            sh_red[0] = total_pAp;
        }
        __syncthreads();
        total_pAp = sh_red[0];

        if (fabsf(total_pAp) < 1e-12f) break;
        float alpha = total_rsold / total_pAp;

        // e = e + alpha * p, r = r - alpha * Ap
        for (int i = tid; i < total_cells; i += blockDim.x) {
            sh_e[i] += alpha * sh_p[i];
            sh_r[i] -= alpha * sh_Ap[i];
        }
        __syncthreads();

        // New r_dot_r
        float rsnew = 0.0f;
        for (int i = tid; i < total_cells; i += blockDim.x) {
            rsnew += sh_r[i] * sh_r[i];
        }
        #pragma unroll
        for (int offset = 16; offset > 0; offset /= 2) {
            rsnew += __shfl_down_sync(0xffffffff, rsnew, offset);
        }
        if (tid % 32 == 0) sh_red[tid / 32] = rsnew;
        __syncthreads();

        float total_rsnew = 0.0f;
        if (tid == 0) {
            for (int w = 0; w < (blockDim.x + 31) / 32; ++w) total_rsnew += sh_red[w];
            sh_red[0] = total_rsnew;
        }
        __syncthreads();
        total_rsnew = sh_red[0];

        if (sqrtf(total_rsnew) < tol) break;

        float beta = total_rsnew / total_rsold;
        total_rsold = total_rsnew;

        for (int i = tid; i < total_cells; i += blockDim.x) {
            sh_p[i] = sh_r[i] + beta * sh_p[i];
        }
        __syncthreads();
    }

    // Write back e_out
    for (int i = tid; i < total_cells; i += blockDim.x) {
        e_out[i] = sh_e[i];
    }
}

#endif /* MULTIGRID_ENGINE_CUH */
