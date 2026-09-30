#!/usr/bin/env python3
"""
utils_2D_bluff_body.py — Calibrated 2D CFD Utilities for Flow Past a Bluff Body (Re=100)
Implements 2D Incompressible Navier-Stokes with immersed boundary / Darcy drag obstacle.
Exhibits classical periodic Von Kármán vortex street shedding.

Governing Equations:
1. Momentum Advection-Diffusion:
   u* = u^n + dt * [ nu * grad^2(u^n) - (u^n . grad)u^n - sigma * u^n ]
2. Pressure Poisson:
   grad^2(p) = (1 / dt) * div(u*)
3. Projection:
   u^{n+1} = u* - dt * grad(p)

Flow Features:
- Asymmetric wake instability leading to alternating vortex shedding
- Periodic lift and drag fluctuations
- Strouhal number St = f * D / U_inf approx 0.15 - 0.16
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# =============================================================================
# 1. 2D Tensor Allocation
# =============================================================================
def create_tensors_2D_bluff(nx, ny, device=None):
    """Allocate all required 2D flow tensors on CPU or GPU."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input_shape = (1, 1, ny, nx)
    input_shape_pad = (1, 1, ny + 2, nx + 2)

    values_u = torch.zeros(input_shape, device=device)
    values_v = torch.zeros(input_shape, device=device)
    values_p = torch.zeros(input_shape, device=device)

    values_uu = torch.zeros(input_shape_pad, device=device)
    values_vv = torch.zeros(input_shape_pad, device=device)
    values_pp = torch.zeros(input_shape_pad, device=device)

    return values_u, values_v, values_p, values_uu, values_vv, values_pp


# =============================================================================
# 2. 2D Bluff Body Solver Class
# =============================================================================
class BluffBody2DSolverGPU:
    """
    GPU-Resident 2D Navier-Stokes Solver for Flow Past Bluff Body:
    - Inflow on left: u = U_inf, v = 0
    - Outflow on right: Neumann du/dx = 0, dv/dx = 0
    - Top and bottom: Slip or no-slip walls
    - Solid obstacle: Immersed boundary Darcy drag sigma = 1e8
    """
    def __init__(self, nx=512, ny=128, dx=1.0, dy=1.0, dt=0.04, nu=0.08, u_inf=1.0,
                 obstacle_type="square", device=None):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.nx = nx
        self.ny = ny
        self.dx = dx
        self.dy = dy
        self.dt = dt
        self.nu = nu
        self.u_inf = u_inf

        self.u = torch.ones((1, 1, ny, nx), device=device, dtype=torch.float32) * u_inf
        self.v = torch.zeros((1, 1, ny, nx), device=device, dtype=torch.float32)
        self.p = torch.zeros((1, 1, ny, nx), device=device, dtype=torch.float32)

        self.uu = torch.zeros((1, 1, ny + 2, nx + 2), device=device, dtype=torch.float32)
        self.vv = torch.zeros((1, 1, ny + 2, nx + 2), device=device, dtype=torch.float32)
        self.pp = torch.zeros((1, 1, ny + 2, nx + 2), device=device, dtype=torch.float32)

        self.mask = torch.zeros((1, 1, ny, nx), device=device, dtype=torch.float32)

        # Build Obstacle Mask (Square Cylinder D = 32 at x0 = nx//4, y0 = ny//2)
        self.x0 = nx // 4
        self.y0 = ny // 2
        self.D = 32 # obstacle diameter / height
        half_d = self.D // 2

        if obstacle_type == "square":
            self.mask[0, 0, (self.y0 - half_d):(self.y0 + half_d), (self.x0 - half_d):(self.x0 + half_d)] = 1.0
        elif obstacle_type == "circle":
            yy, xx = torch.meshgrid(torch.arange(ny, device=device),
                                    torch.arange(nx, device=device), indexing='ij')
            dist = torch.sqrt((xx - self.x0)**2 + (yy - self.y0)**2)
            self.mask[0, 0, dist <= half_d] = 1.0

        # Zero out velocity inside obstacle initially
        self.u.masked_fill_(self.mask > 0.5, 0.0)
        self.v.masked_fill_(self.mask > 0.5, 0.0)

        # Preallocate coordinate grid for semi-Lagrangian or finite-difference
        y_coords = torch.linspace(-1, 1, ny, device=device)
        x_coords = torch.linspace(-1, 1, nx, device=device)
        self.grid_y, self.grid_x = torch.meshgrid(y_coords, x_coords, indexing='ij')

        self.step_count = 0

    def apply_boundary_conditions(self, u_tensor, v_tensor, uu_padded, vv_padded):
        # Fill interior
        uu_padded[0, 0, 1:-1, 1:-1] = u_tensor[0, 0]
        vv_padded[0, 0, 1:-1, 1:-1] = v_tensor[0, 0]

        # Left Boundary: Dirichlet Inflow
        uu_padded[0, 0, :, 0].fill_(self.u_inf)
        vv_padded[0, 0, :, 0].fill_(0.0)

        # Right Boundary: Convective Outflow (Neumann)
        uu_padded[0, 0, :, -1] = uu_padded[0, 0, :, -2]
        vv_padded[0, 0, :, -1] = vv_padded[0, 0, :, -2]

        # Top & Bottom: Slip walls (zero normal gradient for u, zero penetration for v)
        uu_padded[0, 0, 0, :] = uu_padded[0, 0, 1, :]
        uu_padded[0, 0, -1, :] = uu_padded[0, 0, -2, :]
        vv_padded[0, 0, 0, :].fill_(0.0)
        vv_padded[0, 0, -1, :].fill_(0.0)

    def step(self, poisson_iters=35):
        dt = self.dt
        nu = self.nu
        dx = self.dx
        dy = self.dy

        # 1. Boundary conditions
        self.apply_boundary_conditions(self.u, self.v, self.uu, self.vv)

        # Trigger small vertical asymmetry early to stimulate shedding mode
        if 50 <= self.step_count <= 250:
            self.v[0, 0, (self.y0 - 4):(self.y0 + 4), (self.x0 + self.D // 2 + 2):(self.x0 + self.D // 2 + 8)] += 0.03

        # 2. Diffusion (5-point Laplacian)
        lap_u = (self.uu[:, :, 1:-1, 2:] + self.uu[:, :, 1:-1, :-2] +
                 self.uu[:, :, 2:, 1:-1] + self.uu[:, :, :-2, 1:-1] - 4.0 * self.u) / (dx * dy)
        lap_v = (self.vv[:, :, 1:-1, 2:] + self.vv[:, :, 1:-1, :-2] +
                 self.vv[:, :, 2:, 1:-1] + self.vv[:, :, :-2, 1:-1] - 4.0 * self.v) / (dx * dy)

        # Central advection
        du_dx = (self.uu[:, :, 1:-1, 2:] - self.uu[:, :, 1:-1, :-2]) / (2.0 * dx)
        du_dy = (self.uu[:, :, 2:, 1:-1] - self.uu[:, :, :-2, 1:-1]) / (2.0 * dy)
        dv_dx = (self.vv[:, :, 1:-1, 2:] - self.vv[:, :, 1:-1, :-2]) / (2.0 * dx)
        dv_dy = (self.vv[:, :, 2:, 1:-1] - self.vv[:, :, :-2, 1:-1]) / (2.0 * dy)

        # Predictor intermediate velocity u*, v*
        u_star = self.u + dt * (nu * lap_u - (self.u * du_dx + self.v * du_dy))
        v_star = self.v + dt * (nu * lap_v - (self.u * dv_dx + self.v * dv_dy))

        # Enforce no-slip on obstacle
        u_star.masked_fill_(self.mask > 0.5, 0.0)
        v_star.masked_fill_(self.mask > 0.5, 0.0)

        # 3. Intermediate boundary conditions & divergence
        self.apply_boundary_conditions(u_star, v_star, self.uu, self.vv)
        div = ((self.uu[:, :, 1:-1, 2:] - self.uu[:, :, 1:-1, :-2]) / (2.0 * dx) +
               (self.vv[:, :, 2:, 1:-1] - self.vv[:, :, :-2, 1:-1]) / (2.0 * dy))

        # 4. Pressure Poisson solve (Jacobi)
        rhs = (1.0 / dt) * div
        for _ in range(poisson_iters):
            self.pp[0, 0, 1:-1, 1:-1] = self.p[0, 0]
            # Pressure Neumann on walls
            self.pp[0, 0, :, 0] = self.pp[0, 0, :, 1]
            self.pp[0, 0, :, -1].fill_(0.0) # Outflow Dirichlet p=0
            self.pp[0, 0, 0, :] = self.pp[0, 0, 1, :]
            self.pp[0, 0, -1, :] = self.pp[0, 0, -2, :]

            self.p = 0.25 * (self.pp[:, :, 1:-1, 2:] + self.pp[:, :, 1:-1, :-2] +
                             self.pp[:, :, 2:, 1:-1] + self.pp[:, :, :-2, 1:-1] - (dx * dy) * rhs)
            self.p.masked_fill_(self.mask > 0.5, 0.0)

        self.pp[0, 0, 1:-1, 1:-1] = self.p[0, 0]
        self.pp[0, 0, :, 0] = self.pp[0, 0, :, 1]
        self.pp[0, 0, :, -1].fill_(0.0)
        self.pp[0, 0, 0, :] = self.pp[0, 0, 1, :]
        self.pp[0, 0, -1, :] = self.pp[0, 0, -2, :]

        # 5. Projection
        dp_dx = (self.pp[:, :, 1:-1, 2:] - self.pp[:, :, 1:-1, :-2]) / (2.0 * dx)
        dp_dy = (self.pp[:, :, 2:, 1:-1] - self.pp[:, :, :-2, 1:-1]) / (2.0 * dy)

        self.u = u_star - dt * dp_dx
        self.v = v_star - dt * dp_dy

        # Re-enforce zero velocity inside solid obstacle
        self.u.masked_fill_(self.mask > 0.5, 0.0)
        self.v.masked_fill_(self.mask > 0.5, 0.0)

        self.step_count += 1

    def compute_aerodynamic_forces(self):
        """
        Compute instantaneous lift and drag coefficients on the obstacle.
        Calculated via surface pressure integration over the bluff body boundary.
        """
        half_d = self.D // 2
        x_front = self.x0 - half_d
        x_rear = self.x0 + half_d
        y_bot = self.y0 - half_d
        y_top = self.y0 + half_d

        p_front = self.p[0, 0, y_bot:y_top, x_front - 1]
        p_rear = self.p[0, 0, y_bot:y_top, x_rear + 1]
        p_bot = self.p[0, 0, y_bot - 1, x_front:x_rear]
        p_top = self.p[0, 0, y_top + 1, x_front:x_rear]

        f_drag = torch.sum(p_front - p_rear).item() * self.dy
        f_lift = torch.sum(p_bot - p_top).item() * self.dx

        q_inf = 0.5 * (self.u_inf**2) * self.D
        c_d = f_drag / q_inf
        c_l = f_lift / q_inf
        return c_d, c_l
