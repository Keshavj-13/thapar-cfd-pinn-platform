#ifndef BOUNDARY_MANAGER_CUH
#define BOUNDARY_MANAGER_CUH

#include <cuda_runtime.h>
#include "domain_metrics.cuh"
#include "thapar_cfd_engine.h"

// 3D Interior Copy Kernel: Copies interior (D x H x W) to center of padded array
__global__ void copy_interior_to_padded_kernel(
    const float* __restrict__ src,
    float* __restrict__ dst,
    int D, int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < W && y < H && z < D) {
        int src_idx = z * (H * W) + y * W + x;
        int dst_idx = (z + 1) * ((H + 2) * (W + 2)) + (y + 1) * (W + 2) + (x + 1);
        dst[dst_idx] = src[src_idx];
    }
}

// Generalized Boundary Application Kernel for Velocity Field u, v, or w
// Component: 0 = u, 1 = v, 2 = w
__global__ void apply_patch_boundaries_velocity_kernel(
    float* __restrict__ padded_field,
    int component, // 0 for u, 1 for v, 2 for w
    thapar_patch_bc_t bc_xmin, thapar_patch_bc_t bc_xmax,
    thapar_patch_bc_t bc_ymin, thapar_patch_bc_t bc_ymax,
    thapar_patch_bc_t bc_zmin, thapar_patch_bc_t bc_zmax,
    int D, int H, int W)
{
    int pad_W = W + 2;
    int pad_H = H + 2;
    int pad_D = D + 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    // 1. X-Min & X-Max faces (Normal axis = X)
    // Grid: pad_D * pad_H
    for (int i = idx; i < pad_D * pad_H; i += total_threads) {
        int z = i / pad_H;
        int y = i % pad_H;
        if (z < pad_D && y < pad_H) {
            int slice_offset = z * (pad_H * pad_W) + y * pad_W;
            
            // X-Min (Ghost x=0, Interior x=1)
            if (component == 0) { // Normal velocity u
                if (bc_xmin.type == THAPAR_BC_NO_SLIP_WALL || bc_xmin.type == THAPAR_BC_ISOTHERMAL_WALL || bc_xmin.type == THAPAR_BC_FREE_SLIP_SYMMETRY) {
                    padded_field[slice_offset + 0] = -padded_field[slice_offset + 1]; // u_wall = 0
                } else if (bc_xmin.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_field[slice_offset + 0] = 2.0f * bc_xmin.u_val - padded_field[slice_offset + 1];
                } else if (bc_xmin.type == THAPAR_BC_NEUMANN_OUTFLOW) {
                    padded_field[slice_offset + 0] = padded_field[slice_offset + 1];
                } else {
                    padded_field[slice_offset + 0] = -padded_field[slice_offset + 1];
                }
            } else { // Tangential velocity v or w
                float target_val = (component == 1) ? bc_xmin.v_val : bc_xmin.w_val;
                if (bc_xmin.type == THAPAR_BC_NO_SLIP_WALL || bc_xmin.type == THAPAR_BC_ISOTHERMAL_WALL) {
                    padded_field[slice_offset + 0] = -padded_field[slice_offset + 1]; // v_wall = 0
                } else if (bc_xmin.type == THAPAR_BC_FREE_SLIP_SYMMETRY || bc_xmin.type == THAPAR_BC_NEUMANN_OUTFLOW) {
                    padded_field[slice_offset + 0] = padded_field[slice_offset + 1]; // dv/dx = 0
                } else if (bc_xmin.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_field[slice_offset + 0] = 2.0f * target_val - padded_field[slice_offset + 1];
                }
            }

            // X-Max (Ghost x=W+1, Interior x=W)
            if (component == 0) { // Normal velocity u
                if (bc_xmax.type == THAPAR_BC_NEUMANN_OUTFLOW) {
                    padded_field[slice_offset + (W + 1)] = padded_field[slice_offset + W]; // du/dx = 0
                } else if (bc_xmax.type == THAPAR_BC_NO_SLIP_WALL || bc_xmax.type == THAPAR_BC_ISOTHERMAL_WALL || bc_xmax.type == THAPAR_BC_FREE_SLIP_SYMMETRY) {
                    padded_field[slice_offset + (W + 1)] = -padded_field[slice_offset + W]; // u_wall = 0
                } else if (bc_xmax.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_field[slice_offset + (W + 1)] = 2.0f * bc_xmax.u_val - padded_field[slice_offset + W];
                }
            } else { // Tangential velocity v or w
                float target_val = (component == 1) ? bc_xmax.v_val : bc_xmax.w_val;
                if (bc_xmax.type == THAPAR_BC_NEUMANN_OUTFLOW || bc_xmax.type == THAPAR_BC_FREE_SLIP_SYMMETRY) {
                    padded_field[slice_offset + (W + 1)] = padded_field[slice_offset + W];
                } else if (bc_xmax.type == THAPAR_BC_NO_SLIP_WALL || bc_xmax.type == THAPAR_BC_ISOTHERMAL_WALL) {
                    padded_field[slice_offset + (W + 1)] = -padded_field[slice_offset + W];
                } else if (bc_xmax.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_field[slice_offset + (W + 1)] = 2.0f * target_val - padded_field[slice_offset + W];
                }
            }
        }
    }

    // 2. Y-Min & Y-Max faces (Normal axis = Y)
    // Grid: pad_D * pad_W
    for (int i = idx; i < pad_D * pad_W; i += total_threads) {
        int z = i / pad_W;
        int x = i % pad_W;
        if (z < pad_D && x < pad_W) {
            int z_offset = z * (pad_H * pad_W) + x;

            // Y-Min (Ghost y=0, Interior y=1)
            if (component == 1) { // Normal velocity v
                // CRITICAL FIX: Normal velocity must be strictly anti-symmetric (v_wall = 0)
                if (bc_ymin.type == THAPAR_BC_NO_SLIP_WALL || bc_ymin.type == THAPAR_BC_ISOTHERMAL_WALL || bc_ymin.type == THAPAR_BC_FREE_SLIP_SYMMETRY) {
                    padded_field[z_offset + 0 * pad_W] = -padded_field[z_offset + 1 * pad_W];
                } else if (bc_ymin.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_field[z_offset + 0 * pad_W] = 2.0f * bc_ymin.v_val - padded_field[z_offset + 1 * pad_W];
                } else {
                    padded_field[z_offset + 0 * pad_W] = -padded_field[z_offset + 1 * pad_W];
                }
            } else { // Tangential velocity u or w
                float target_val = (component == 0) ? bc_ymin.u_val : bc_ymin.w_val;
                if (bc_ymin.type == THAPAR_BC_NO_SLIP_WALL || bc_ymin.type == THAPAR_BC_ISOTHERMAL_WALL) {
                    padded_field[z_offset + 0 * pad_W] = -padded_field[z_offset + 1 * pad_W];
                } else if (bc_ymin.type == THAPAR_BC_FREE_SLIP_SYMMETRY || bc_ymin.type == THAPAR_BC_NEUMANN_OUTFLOW) {
                    padded_field[z_offset + 0 * pad_W] = padded_field[z_offset + 1 * pad_W];
                } else if (bc_ymin.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_field[z_offset + 0 * pad_W] = 2.0f * target_val - padded_field[z_offset + 1 * pad_W];
                }
            }

            // Y-Max (Ghost y=H+1, Interior y=H)
            if (component == 1) { // Normal velocity v
                if (bc_ymax.type == THAPAR_BC_NO_SLIP_WALL || bc_ymax.type == THAPAR_BC_ISOTHERMAL_WALL || bc_ymax.type == THAPAR_BC_FREE_SLIP_SYMMETRY) {
                    padded_field[z_offset + (H + 1) * pad_W] = -padded_field[z_offset + H * pad_W];
                } else if (bc_ymax.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_field[z_offset + (H + 1) * pad_W] = 2.0f * bc_ymax.v_val - padded_field[z_offset + H * pad_W];
                } else {
                    padded_field[z_offset + (H + 1) * pad_W] = -padded_field[z_offset + H * pad_W];
                }
            } else { // Tangential velocity u or w
                float target_val = (component == 0) ? bc_ymax.u_val : bc_ymax.w_val;
                if (bc_ymax.type == THAPAR_BC_NO_SLIP_WALL || bc_ymax.type == THAPAR_BC_ISOTHERMAL_WALL) {
                    padded_field[z_offset + (H + 1) * pad_W] = -padded_field[z_offset + H * pad_W];
                } else if (bc_ymax.type == THAPAR_BC_FREE_SLIP_SYMMETRY || bc_ymax.type == THAPAR_BC_NEUMANN_OUTFLOW) {
                    padded_field[z_offset + (H + 1) * pad_W] = padded_field[z_offset + H * pad_W];
                } else if (bc_ymax.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_field[z_offset + (H + 1) * pad_W] = 2.0f * target_val - padded_field[z_offset + H * pad_W];
                }
            }
        }
    }

    // 3. Z-Min & Z-Max faces (Normal axis = Z)
    // Grid: pad_H * pad_W
    for (int i = idx; i < pad_H * pad_W; i += total_threads) {
        int y = i / pad_W;
        int x = i % pad_W;
        if (y < pad_H && x < pad_W) {
            int base_offset = y * pad_W + x;

            // Z-Min (Ghost z=0, Interior z=1)
            if (component == 2) { // Normal velocity w
                if (bc_zmin.type == THAPAR_BC_NO_SLIP_WALL || bc_zmin.type == THAPAR_BC_ISOTHERMAL_WALL || bc_zmin.type == THAPAR_BC_FREE_SLIP_SYMMETRY) {
                    padded_field[0 * (pad_H * pad_W) + base_offset] = -padded_field[1 * (pad_H * pad_W) + base_offset];
                } else if (bc_zmin.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_field[0 * (pad_H * pad_W) + base_offset] = 2.0f * bc_zmin.w_val - padded_field[1 * (pad_H * pad_W) + base_offset];
                } else {
                    padded_field[0 * (pad_H * pad_W) + base_offset] = -padded_field[1 * (pad_H * pad_W) + base_offset];
                }
            } else { // Tangential velocity u or v
                float target_val = (component == 0) ? bc_zmin.u_val : bc_zmin.v_val;
                if (bc_zmin.type == THAPAR_BC_NO_SLIP_WALL || bc_zmin.type == THAPAR_BC_ISOTHERMAL_WALL) {
                    padded_field[0 * (pad_H * pad_W) + base_offset] = -padded_field[1 * (pad_H * pad_W) + base_offset];
                } else if (bc_zmin.type == THAPAR_BC_FREE_SLIP_SYMMETRY || bc_zmin.type == THAPAR_BC_NEUMANN_OUTFLOW) {
                    padded_field[0 * (pad_H * pad_W) + base_offset] = padded_field[1 * (pad_H * pad_W) + base_offset];
                } else if (bc_zmin.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_field[0 * (pad_H * pad_W) + base_offset] = 2.0f * target_val - padded_field[1 * (pad_H * pad_W) + base_offset];
                }
            }

            // Z-Max (Ghost z=D+1, Interior z=D)
            if (component == 2) { // Normal velocity w
                if (bc_zmax.type == THAPAR_BC_MOVING_LID || bc_zmax.type == THAPAR_BC_NO_SLIP_WALL || bc_zmax.type == THAPAR_BC_ISOTHERMAL_WALL || bc_zmax.type == THAPAR_BC_FREE_SLIP_SYMMETRY) {
                    padded_field[(D + 1) * (pad_H * pad_W) + base_offset] = -padded_field[D * (pad_H * pad_W) + base_offset];
                } else if (bc_zmax.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_field[(D + 1) * (pad_H * pad_W) + base_offset] = 2.0f * bc_zmax.w_val - padded_field[D * (pad_H * pad_W) + base_offset];
                } else {
                    padded_field[(D + 1) * (pad_H * pad_W) + base_offset] = -padded_field[D * (pad_H * pad_W) + base_offset];
                }
            } else { // Tangential velocity u or v
                float target_val = (component == 0) ? bc_zmax.u_val : bc_zmax.v_val;
                if (bc_zmax.type == THAPAR_BC_MOVING_LID) {
                    padded_field[(D + 1) * (pad_H * pad_W) + base_offset] = 2.0f * target_val - padded_field[D * (pad_H * pad_W) + base_offset];
                } else if (bc_zmax.type == THAPAR_BC_NO_SLIP_WALL || bc_zmax.type == THAPAR_BC_ISOTHERMAL_WALL) {
                    padded_field[(D + 1) * (pad_H * pad_W) + base_offset] = -padded_field[D * (pad_H * pad_W) + base_offset];
                } else if (bc_zmax.type == THAPAR_BC_FREE_SLIP_SYMMETRY || bc_zmax.type == THAPAR_BC_NEUMANN_OUTFLOW) {
                    padded_field[(D + 1) * (pad_H * pad_W) + base_offset] = padded_field[D * (pad_H * pad_W) + base_offset];
                }
            }
        }
    }
}

// Pressure Boundary Application Kernel (Homogeneous Neumann on all walls, Dirichlet zero on outflow)
__global__ void apply_patch_boundaries_pressure_kernel(
    float* __restrict__ pp,
    thapar_patch_bc_t bc_xmin, thapar_patch_bc_t bc_xmax,
    thapar_patch_bc_t bc_ymin, thapar_patch_bc_t bc_ymax,
    thapar_patch_bc_t bc_zmin, thapar_patch_bc_t bc_zmax,
    int D, int H, int W)
{
    int pad_W = W + 2;
    int pad_H = H + 2;
    int pad_D = D + 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    // X faces
    for (int i = idx; i < pad_D * pad_H; i += total_threads) {
        int z = i / pad_H;
        int y = i % pad_H;
        if (z < pad_D && y < pad_H) {
            int offset = z * (pad_H * pad_W) + y * pad_W;
            pp[offset + 0] = pp[offset + 1];
            pp[offset + (W + 1)] = pp[offset + W];
        }
    }

    // Y faces
    for (int i = idx; i < pad_D * pad_W; i += total_threads) {
        int z = i / pad_W;
        int x = i % pad_W;
        if (z < pad_D && x < pad_W) {
            int offset = z * (pad_H * pad_W) + x;
            pp[offset + 0 * pad_W] = pp[offset + 1 * pad_W];
            pp[offset + (H + 1) * pad_W] = pp[offset + H * pad_W];
        }
    }

    // Z faces
    for (int i = idx; i < pad_H * pad_W; i += total_threads) {
        int y = i / pad_W;
        int x = i % pad_W;
        if (y < pad_H && x < pad_W) {
            int offset = y * pad_W + x;
            pp[0 * (pad_H * pad_W) + offset] = pp[1 * (pad_H * pad_W) + offset];
            pp[(D + 1) * (pad_H * pad_W) + offset] = pp[D * (pad_H * pad_W) + offset];
        }
    }
}

// Scalar Boundary Application Kernel (Temperature, Spalart-Allmaras, VoF)
__global__ void apply_patch_boundaries_scalar_kernel(
    float* __restrict__ padded_scalar,
    thapar_patch_bc_t bc_xmin, thapar_patch_bc_t bc_xmax,
    thapar_patch_bc_t bc_ymin, thapar_patch_bc_t bc_ymax,
    thapar_patch_bc_t bc_zmin, thapar_patch_bc_t bc_zmax,
    int D, int H, int W,
    bool is_temperature)
{
    int pad_W = W + 2;
    int pad_H = H + 2;
    int pad_D = D + 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_threads = blockDim.x * gridDim.x;

    // X faces
    for (int i = idx; i < pad_D * pad_H; i += total_threads) {
        int z = i / pad_H;
        int y = i % pad_H;
        if (z < pad_D && y < pad_H) {
            int offset = z * (pad_H * pad_W) + y * pad_W;
            // X-Min
            if (is_temperature) {
                if (bc_xmin.type == THAPAR_BC_ISOTHERMAL_WALL || bc_xmin.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_scalar[offset + 0] = 2.0f * bc_xmin.t_val - padded_scalar[offset + 1];
                } else {
                    padded_scalar[offset + 0] = padded_scalar[offset + 1];
                }
            } else {
                if (bc_xmin.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_scalar[offset + 0] = 2.0f * bc_xmin.u_val - padded_scalar[offset + 1];
                } else {
                    padded_scalar[offset + 0] = padded_scalar[offset + 1];
                }
            }

            // X-Max
            if (is_temperature) {
                if (bc_xmax.type == THAPAR_BC_ISOTHERMAL_WALL || bc_xmax.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_scalar[offset + (W + 1)] = 2.0f * bc_xmax.t_val - padded_scalar[offset + W];
                } else {
                    padded_scalar[offset + (W + 1)] = padded_scalar[offset + W];
                }
            } else {
                if (bc_xmax.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_scalar[offset + (W + 1)] = 2.0f * bc_xmax.u_val - padded_scalar[offset + W];
                } else {
                    padded_scalar[offset + (W + 1)] = padded_scalar[offset + W];
                }
            }
        }
    }

    // Y faces
    for (int i = idx; i < pad_D * pad_W; i += total_threads) {
        int z = i / pad_W;
        int x = i % pad_W;
        if (z < pad_D && x < pad_W) {
            int offset = z * (pad_H * pad_W) + x;
            // Y-Min
            if (is_temperature) {
                if (bc_ymin.type == THAPAR_BC_ISOTHERMAL_WALL || bc_ymin.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_scalar[offset + 0 * pad_W] = 2.0f * bc_ymin.t_val - padded_scalar[offset + 1 * pad_W];
                } else {
                    padded_scalar[offset + 0 * pad_W] = padded_scalar[offset + 1 * pad_W];
                }
            } else {
                if (bc_ymin.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_scalar[offset + 0 * pad_W] = 2.0f * bc_ymin.u_val - padded_scalar[offset + 1 * pad_W];
                } else {
                    padded_scalar[offset + 0 * pad_W] = padded_scalar[offset + 1 * pad_W];
                }
            }

            // Y-Max
            if (is_temperature) {
                if (bc_ymax.type == THAPAR_BC_ISOTHERMAL_WALL || bc_ymax.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_scalar[offset + (H + 1) * pad_W] = 2.0f * bc_ymax.t_val - padded_scalar[offset + H * pad_W];
                } else {
                    padded_scalar[offset + (H + 1) * pad_W] = padded_scalar[offset + H * pad_W];
                }
            } else {
                if (bc_ymax.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_scalar[offset + (H + 1) * pad_W] = 2.0f * bc_ymax.u_val - padded_scalar[offset + H * pad_W];
                } else {
                    padded_scalar[offset + (H + 1) * pad_W] = padded_scalar[offset + H * pad_W];
                }
            }
        }
    }

    // Z faces
    for (int i = idx; i < pad_H * pad_W; i += total_threads) {
        int y = i / pad_W;
        int x = i % pad_W;
        if (y < pad_H && x < pad_W) {
            int offset = y * pad_W + x;
            // Z-Min
            if (is_temperature) {
                if (bc_zmin.type == THAPAR_BC_ISOTHERMAL_WALL || bc_zmin.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_scalar[0 * (pad_H * pad_W) + offset] = 2.0f * bc_zmin.t_val - padded_scalar[1 * (pad_H * pad_W) + offset];
                } else {
                    padded_scalar[0 * (pad_H * pad_W) + offset] = padded_scalar[1 * (pad_H * pad_W) + offset];
                }
            } else {
                if (bc_zmin.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_scalar[0 * (pad_H * pad_W) + offset] = 2.0f * bc_zmin.u_val - padded_scalar[1 * (pad_H * pad_W) + offset];
                } else {
                    padded_scalar[0 * (pad_H * pad_W) + offset] = padded_scalar[1 * (pad_H * pad_W) + offset];
                }
            }

            // Z-Max
            if (is_temperature) {
                if (bc_zmax.type == THAPAR_BC_ISOTHERMAL_WALL || bc_zmax.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_scalar[(D + 1) * (pad_H * pad_W) + offset] = 2.0f * bc_zmax.t_val - padded_scalar[D * (pad_H * pad_W) + offset];
                } else {
                    padded_scalar[(D + 1) * (pad_H * pad_W) + offset] = padded_scalar[D * (pad_H * pad_W) + offset];
                }
            } else {
                if (bc_zmax.type == THAPAR_BC_DIRICHLET_INFLOW) {
                    padded_scalar[(D + 1) * (pad_H * pad_W) + offset] = 2.0f * bc_zmax.u_val - padded_scalar[D * (pad_H * pad_W) + offset];
                } else {
                    padded_scalar[(D + 1) * (pad_H * pad_W) + offset] = padded_scalar[D * (pad_H * pad_W) + offset];
                }
            }
        }
    }
}

#endif /* BOUNDARY_MANAGER_CUH */
