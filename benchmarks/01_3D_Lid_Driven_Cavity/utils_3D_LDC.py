#!/usr/bin/env python3
"""
utile_3D_LDC.py — Calibrated 3D CFD Utilities for 3D Lid-Driven Cavity (LDC)
Inherited from AI4PDEs, with the first-derivative multiplier calibration (unit gain)
and optimized boundary conditions for accurate 3D LDC simulation in Jupyter Notebooks.

Key Modifications:
1. Removed the extra '* 0.5' multiplier on w2, w3, w4 in get_weights_linear_3D(dx)
   to restore unit derivative gain (positive weights sum to 0.500 -> 1/(2*dx)).
2. Provides pure 1D finite-difference stencil alternative get_weights_1D_3D(dx)
   which eliminates 27-point transverse numerical viscosity.
3. Provides cell-centered boundary conditions for 3D LDC.
"""

import numpy as np
import torch
import torch.nn.functional as F

# =============================================================================
# 1. 3D Tensor Allocation
# =============================================================================
def create_tensors_3D(nx, ny, nz, device=None):
    """
    Allocate all required 3D flow tensors on CPU or GPU.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_shape = (1, 1, nz, ny, nx)
    input_shape_pad = (1, 1, nz + 2, ny + 2, nx + 2)

    values_u = torch.zeros(input_shape, device=device)
    values_v = torch.zeros(input_shape, device=device)
    values_w = torch.zeros(input_shape, device=device)
    values_p = torch.zeros(input_shape, device=device)

    values_uu = torch.zeros(input_shape_pad, device=device)
    values_vv = torch.zeros(input_shape_pad, device=device)
    values_ww = torch.zeros(input_shape_pad, device=device)
    values_pp = torch.zeros(input_shape_pad, device=device)

    b_uu = torch.zeros(input_shape_pad, device=device)
    b_vv = torch.zeros(input_shape_pad, device=device)
    b_ww = torch.zeros(input_shape_pad, device=device)

    print('All the required 3D tensors have been created successfully!')
    print('===========================================================')
    print(f'Device:    {device}')
    print('values_u  => u velocity [first step]  - (1,1,nz,ny,nx)')
    print('values_v  => v velocity [first step]  - (1,1,nz,ny,nx)')
    print('values_w  => w velocity [first step]  - (1,1,nz,ny,nx)')
    print('values_p  => pressure                 - (1,1,nz,ny,nx)')
    print('b_uu      => u velocity [second step] - (1,1,nz+2,ny+2,nx+2)')
    print('b_vv      => v velocity [second step] - (1,1,nz+2,ny+2,nx+2)')
    print('b_ww      => w velocity [second step] - (1,1,nz+2,ny+2,nx+2)')
    print('values_uu => u velocity [padded]      - (1,1,nz+2,ny+2,nx+2)')
    print('values_vv => v velocity [padded]      - (1,1,nz+2,ny+2,nx+2)')
    print('values_ww => w velocity [padded]      - (1,1,nz+2,ny+2,nx+2)')
    print('values_pp => pressure   [padded]      - (1,1,nz+2,ny+2,nx+2)')
    print('===========================================================')
    return values_u, values_v, values_w, values_p, values_uu, values_vv, values_ww, values_pp, b_uu, b_vv, b_ww


# =============================================================================
# 2. Calibrated 3D Differencing Stencils (Unit Gain)
# =============================================================================
def get_weights_linear_3D(dx):
    """
    Constructs 3D second-order (Laplacian) and first-order (gradient/advection) stencils.
    
    CALIBRATION FIX:
    Original AI4PDEs had `w2[0,0,:,:,:] = -p_div_x / dx * 0.5`.
    Because p_div_x positive weights already summed to 0.500 across all 3 slices,
    the extra `* 0.5` halved convective momentum to 0.25 (Re_eff = Re / 2).
    Here, the multiplier is calibrated to unit gain (-p_div / dx).
    """
    # 27-Point Laplacian w1 and Poisson matrix wA
    pd1 = torch.tensor([[2/26, 3/26, 2/26],
                        [3/26, 6/26, 3/26],
                        [2/26, 3/26, 2/26]])
    pd2 = torch.tensor([[3/26, 6/26, 3/26],
                        [6/26, -88/26, 6/26],
                        [3/26, 6/26, 3/26]])
    pd3 = torch.tensor([[2/26, 3/26, 2/26],
                        [3/26, 6/26, 3/26],
                        [2/26, 3/26, 2/26]])

    w1 = torch.zeros([1, 1, 3, 3, 3])
    wA = torch.zeros([1, 1, 3, 3, 3])
    w1[0, 0, 0, :, :] = pd1 / (dx**2)
    w1[0, 0, 1, :, :] = pd2 / (dx**2)
    w1[0, 0, 2, :, :] = pd3 / (dx**2)

    wA[0, 0, 0, :, :] = -pd1 / (dx**2)
    wA[0, 0, 1, :, :] = -pd2 / (dx**2)
    wA[0, 0, 2, :, :] = -pd3 / (dx**2)

    # Gradient filters (Sum of positive elements = 0.500)
    p_div_x1 = torch.tensor([[-0.014, 0.0, 0.014],
                            [-0.056, 0.0, 0.056],
                            [-0.014, 0.0, 0.014]])
    p_div_x2 = torch.tensor([[-0.056, 0.0, 0.056],
                            [-0.220, 0.0, 0.220],
                            [-0.056, 0.0, 0.056]])
    p_div_x3 = torch.tensor([[-0.014, 0.0, 0.014],
                            [-0.056, 0.0, 0.056],
                            [-0.014, 0.0, 0.014]])

    p_div_y1 = torch.tensor([[ 0.014,  0.056,  0.014],
                            [ 0.000,  0.000,  0.000],
                            [-0.014, -0.056, -0.014]])
    p_div_y2 = torch.tensor([[ 0.056,  0.220,  0.056],
                            [ 0.000,  0.000,  0.000],
                            [-0.056, -0.220, -0.056]])
    p_div_y3 = torch.tensor([[ 0.014,  0.056,  0.014],
                            [ 0.000,  0.000,  0.000],
                            [-0.014, -0.056, -0.014]])

    p_div_z1 = torch.tensor([[0.014, 0.056, 0.014],
                            [0.056, 0.220, 0.056],
                            [0.014, 0.056, 0.014]])
    p_div_z2 = torch.tensor([[0.000, 0.000, 0.000],
                            [0.000, 0.000, 0.000],
                            [0.000, 0.000, 0.000]])
    p_div_z3 = torch.tensor([[-0.014, -0.056, -0.014],
                            [-0.056, -0.220, -0.056],
                            [-0.014, -0.056, -0.014]])

    w2 = torch.zeros([1, 1, 3, 3, 3])
    w3 = torch.zeros([1, 1, 3, 3, 3])
    w4 = torch.zeros([1, 1, 3, 3, 3])

    # =========================================================================
    # CALIBRATED MULTIPLIERS (Removed extra * 0.5 to enforce unit gain)
    # =========================================================================
    w2[0, 0, 0, :, :] = -p_div_x1 / dx
    w2[0, 0, 1, :, :] = -p_div_x2 / dx
    w2[0, 0, 2, :, :] = -p_div_x3 / dx

    w3[0, 0, 0, :, :] = -p_div_y1 / dx
    w3[0, 0, 1, :, :] = -p_div_y2 / dx
    w3[0, 0, 2, :, :] = -p_div_y3 / dx

    w4[0, 0, 0, :, :] = -p_div_z1 / dx
    w4[0, 0, 1, :, :] = -p_div_z2 / dx
    w4[0, 0, 2, :, :] = -p_div_z3 / dx

    # Multigrid restriction filter
    w_res = torch.zeros([1, 1, 2, 2, 2])
    w_res[0, 0, :, :, :] = 0.125
    diag = np.array(wA)[0, 0, 1, 1, 1]

    print('All the required 3D filters have been created successfully!')
    print('===========================================================')
    print('w1    => second order derivative  - (1,1,3,3,3)')
    print('w2    => first order derivative x - (1,1,3,3,3) [CALIBRATED UNIT GAIN]')
    print('w3    => first order derivative y - (1,1,3,3,3) [CALIBRATED UNIT GAIN]')
    print('w4    => first order derivative z - (1,1,3,3,3) [CALIBRATED UNIT GAIN]')
    print('wA    => second order derivative  - (1,1,3,3,3)')
    print('w_res => Restriction operation    - (1,1,2,2,2)')
    print(f'diag  => Diagonal component of wA - ({diag:.4f})')
    print('===========================================================')
    return w1, w2, w3, w4, wA, w_res, diag


def get_weights_1D_3D(dx):
    """
    Pure 1D Central Differencing Stencils (Zero Transverse Viscosity).
    Recommended for high-Re boundary layers (Re >= 400).
    Computes exact du/dx = (u[i+1] - u[i-1]) / (2*dx) with no transverse averaging.
    """
    pd1 = torch.tensor([[2/26, 3/26, 2/26],
                        [3/26, 6/26, 3/26],
                        [2/26, 3/26, 2/26]])
    pd2 = torch.tensor([[3/26, 6/26, 3/26],
                        [6/26, -88/26, 6/26],
                        [3/26, 6/26, 3/26]])
    pd3 = torch.tensor([[2/26, 3/26, 2/26],
                        [3/26, 6/26, 3/26],
                        [2/26, 3/26, 2/26]])

    w1 = torch.zeros([1, 1, 3, 3, 3])
    wA = torch.zeros([1, 1, 3, 3, 3])
    w1[0, 0, 0, :, :] = pd1 / (dx**2)
    w1[0, 0, 1, :, :] = pd2 / (dx**2)
    w1[0, 0, 2, :, :] = pd3 / (dx**2)

    wA[0, 0, 0, :, :] = -pd1 / (dx**2)
    wA[0, 0, 1, :, :] = -pd2 / (dx**2)
    wA[0, 0, 2, :, :] = -pd3 / (dx**2)

    w2 = torch.zeros([1, 1, 3, 3, 3])
    w3 = torch.zeros([1, 1, 3, 3, 3])
    w4 = torch.zeros([1, 1, 3, 3, 3])

    # Exact 1D central difference along each respective axis
    # Convolution convention in PyTorch: (f * g)[i] = sum_j f[i-j]*g[j]
    # For (u[i+1] - u[i-1]) / (2*dx), stencil kernel values are:
    # index 0: -0.5/dx, index 2: +0.5/dx
    w2[0, 0, 1, 1, 0] = -0.5 / dx
    w2[0, 0, 1, 1, 2] =  0.5 / dx

    w3[0, 0, 1, 0, 1] = -0.5 / dx
    w3[0, 0, 1, 2, 1] =  0.5 / dx

    w4[0, 0, 0, 1, 1] = -0.5 / dx
    w4[0, 0, 2, 1, 1] =  0.5 / dx

    w_res = torch.zeros([1, 1, 2, 2, 2])
    w_res[0, 0, :, :, :] = 0.125
    diag = np.array(wA)[0, 0, 1, 1, 1]
    return w1, w2, w3, w4, wA, w_res, diag


# =============================================================================
# 3. 3D Lid-Driven Cavity Boundary Conditions
# =============================================================================
def boundary_condition_3D_u_cell(values_u, values_uu, ub, regularized=False):
    """
    True Cell-Centered Boundary Conditions for 3D LDC (u-velocity).
    Wall is at the cell face halfway between ghost and interior cells:
    No-slip wall: (u_ghost + u_inner) / 2 = 0  =>  u_ghost = -u_inner
    Top moving lid: (u_ghost + u_inner) / 2 = ub => u_ghost = 2*ub - u_inner
    """
    nz = values_u.shape[2]
    ny = values_u.shape[3]
    nx = values_u.shape[4]

    # Map inner domain
    values_uu[0, 0, 1:nz+1, 1:ny+1, 1:nx+1] = values_u[0, 0, :, :, :]

    # Left & Right walls (x = 0 and x = nx+1)
    values_uu[0, 0, :, :, 0] = -values_uu[0, 0, :, :, 1]
    values_uu[0, 0, :, :, nx+1] = -values_uu[0, 0, :, :, nx]

    # Front & Back walls (y = 0 and y = ny+1)
    values_uu[0, 0, :, 0, :] = -values_uu[0, 0, :, 1, :]
    values_uu[0, 0, :, ny+1, :] = -values_uu[0, 0, :, ny, :]

    # Bottom wall (z = 0)
    values_uu[0, 0, 0, :, :] = -values_uu[0, 0, 1, :, :]

    # Top moving lid (z = nz+1)
    if regularized:
        # 16th-order polynomial regularization to damp corner singularities
        y_coords = (torch.arange(ny, device=values_u.device, dtype=values_u.dtype) + 0.5) / ny
        x_coords = (torch.arange(nx, device=values_u.device, dtype=values_u.dtype) + 0.5) / nx
        ty = 2.0 * y_coords - 1.0
        tx = 2.0 * x_coords - 1.0
        fy = torch.clamp(1.0 - ty**16, min=0.0)
        fx = torch.clamp(1.0 - tx**16, min=0.0)
        reg_profile = fx.unsqueeze(0) * fy.unsqueeze(1)  # (ny, nx)
        values_uu[0, 0, nz+1, 1:ny+1, 1:nx+1] = 2.0 * ub * reg_profile - values_uu[0, 0, nz, 1:ny+1, 1:nx+1]
    else:
        values_uu[0, 0, nz+1, 1:ny+1, 1:nx+1] = 2.0 * ub - values_uu[0, 0, nz, 1:ny+1, 1:nx+1]

    return values_uu


def boundary_condition_3D_v_cell(values_v, values_vv, ub=0.0):
    """
    True Cell-Centered Boundary Conditions for 3D LDC (v-velocity).
    All 6 bounding faces are stationary no-slip walls (v_wall = 0).
    """
    nz = values_v.shape[2]
    ny = values_v.shape[3]
    nx = values_v.shape[4]

    values_vv[0, 0, 1:nz+1, 1:ny+1, 1:nx+1] = values_v[0, 0, :, :, :]

    values_vv[0, 0, :, :, 0] = -values_vv[0, 0, :, :, 1]
    values_vv[0, 0, :, :, nx+1] = -values_vv[0, 0, :, :, nx]
    values_vv[0, 0, :, 0, :] = -values_vv[0, 0, :, 1, :]
    values_vv[0, 0, :, ny+1, :] = -values_vv[0, 0, :, ny, :]
    values_vv[0, 0, 0, :, :] = -values_vv[0, 0, 1, :, :]
    values_vv[0, 0, nz+1, :, :] = -values_vv[0, 0, nz, :, :]
    return values_vv


def boundary_condition_3D_w_cell(values_w, values_ww, ub=0.0):
    """
    True Cell-Centered Boundary Conditions for 3D LDC (w-velocity).
    All 6 bounding faces are stationary no-slip walls (w_wall = 0).
    """
    return boundary_condition_3D_v_cell(values_w, values_ww, ub)


def boundary_condition_3D_p_cell(values_p, values_pp):
    """
    Zero-Gradient Neumann Boundary Condition for Pressure:
    dp/dn = 0  =>  p_ghost = p_inner
    """
    nz = values_p.shape[2]
    ny = values_p.shape[3]
    nx = values_p.shape[4]

    values_pp[0, 0, 1:nz+1, 1:ny+1, 1:nx+1] = values_p[0, 0, :, :, :]

    values_pp[0, 0, :, :, 0] = values_pp[0, 0, :, :, 1]
    values_pp[0, 0, :, :, nx+1] = values_pp[0, 0, :, :, nx]
    values_pp[0, 0, :, 0, :] = values_pp[0, 0, :, 1, :]
    values_pp[0, 0, :, ny+1, :] = values_pp[0, 0, :, ny, :]
    values_pp[0, 0, 0, :, :] = values_pp[0, 0, 1, :, :]
    values_pp[0, 0, nz+1, :, :] = values_pp[0, 0, nz, :, :]
    return values_pp


# Aliases matching standard AI4PDEs function naming convention:
boundary_condition_3D_u = boundary_condition_3D_u_cell
boundary_condition_3D_v = boundary_condition_3D_v_cell
boundary_condition_3D_w = boundary_condition_3D_w_cell
boundary_condition_3D_p = boundary_condition_3D_p_cell

def boundary_condition_3D_k(k_u):
    return F.pad(k_u, (1, 1, 1, 1, 1, 1), mode='constant', value=0)

def boundary_condition_3D_cw(w):
    return F.pad(w, (1, 1, 1, 1, 1, 1), mode='constant', value=0)
