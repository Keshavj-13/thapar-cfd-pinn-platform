#!/usr/bin/env python3
"""
utils_3D_buildings.py — Calibrated 3D CFD Utilities for 3D Flow Past Buildings / Urban Canopy
Inherited from AI4PDEs, with unit-gain derivative stencils and GPU-resident CUDA Graph execution.

Governing Equations:
1. Momentum Advection-Diffusion:
   u* = u^n + dt * [ nu * grad^2(u^n) - (u^n . grad)u^n - sigma * u^n ]
2. Pressure Poisson Equation:
   grad^2(p) = (1 / dt) * div(u*)
3. Pressure Projection & Incompressibility:
   u^{n+1} = u* - dt * grad(p)

Geometry & Domain:
- Realistic urban canopy building obstacle block array loaded from Mesh_buildings_highres.npy
- Darcy drag tensor sigma = 10^8 inside solid buildings, 0 in atmospheric flow passages
- Ground no-slip (y=0), top slip (y=ny), lateral slip (z=0, nz), inflow u=ub (x=0), outflow Neumann (x=nx)
"""

import os
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Add CUDA extension path for cfd_cuda
CUDA_PATH = "/workspace/cuda-optim/src/cuda"
if CUDA_PATH not in sys.path:
    sys.path.insert(0, CUDA_PATH)

try:
    import cfd_cuda
    HAS_CFD_CUDA = True
except ImportError:
    HAS_CFD_CUDA = False

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False


# =============================================================================
# 1. 3D Tensor Allocation
# =============================================================================
def create_tensors_3D(nx, ny, nz, device=None):
    """Allocate all required 3D flow tensors on CPU or GPU."""
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

    return (values_u, values_v, values_w, values_p,
            values_uu, values_vv, values_ww, values_pp,
            b_uu, b_vv, b_ww)


# =============================================================================
# 2. Calibrated Discrete Stencils
# =============================================================================
def get_weights_1D_3D(dx):
    """Return calibrated directional 1D finite-difference stencils."""
    w1 = np.zeros((1, 1, 3, 3, 3), dtype='float32')
    w2 = np.zeros((1, 1, 3, 3, 3), dtype='float32')
    w3 = np.zeros((1, 1, 3, 3, 3), dtype='float32')
    w4 = np.zeros((1, 1, 3, 3, 3), dtype='float32')
    wA = np.zeros((1, 1, 3, 3, 3), dtype='float32')
    w_res = np.zeros((1, 1, 3, 3, 3), dtype='float32')
    diag = 6.0

    inv_dx2 = 1.0 / (dx * dx)
    w1[0, 0, 1, 1, 2] = inv_dx2
    w1[0, 0, 1, 1, 0] = inv_dx2
    w1[0, 0, 1, 2, 1] = inv_dx2
    w1[0, 0, 1, 0, 1] = inv_dx2
    w1[0, 0, 2, 1, 1] = inv_dx2
    w1[0, 0, 0, 1, 1] = inv_dx2
    w1[0, 0, 1, 1, 1] = -6.0 * inv_dx2

    w2[0, 0, 1, 1, 2] = 0.5 / dx
    w2[0, 0, 1, 1, 0] = -0.5 / dx
    w3[0, 0, 1, 2, 1] = 0.5 / dx
    w3[0, 0, 1, 0, 1] = -0.5 / dx
    w4[0, 0, 2, 1, 1] = 0.5 / dx
    w4[0, 0, 0, 1, 1] = -0.5 / dx

    wA[0, 0, :, :, :] = 1.0 / 27.0
    w_res[0, 0, :, :, :] = -w1[0, 0, :, :, :] / diag

    return [torch.from_numpy(w).float() for w in [w1, w2, w3, w4, wA, w_res]] + [diag]


# =============================================================================
# 3. Urban Canopy Building Mesh Loader
# =============================================================================
def load_buildings_sigma(mesh_path, nx, ny, nz, device):
    """
    Load urban canopy geometry and construct inverse permeability tensor sigma.
    Solid obstacle voxels receive sigma = 1e8, fluid cells receive sigma = 0.
    Coordinate mapping:
      mesh_raw axis 0 is z (vertical height, 128)
      mesh_raw axis 1 is y (spanwise width, 1024)
      mesh_raw axis 2 is x (streamwise length, 1024)
    Output shape: (1, 1, nz, ny, nx) where nz is height, ny is spanwise, nx is streamwise.
    """
    if not os.path.exists(mesh_path):
        raise FileNotFoundError(f"Mesh file not found at: {mesh_path}")

    mesh_raw = np.load(mesh_path, mmap_mode="r")
    start_y = 350
    start_x = 200
    if mesh_raw.ndim == 5:
        crop = mesh_raw[0, 0, :nz, start_y:start_y+ny, start_x:start_x+nx]
    else:
        crop = mesh_raw[:nz, start_y:start_y+ny, start_x:start_x+nx]

    # In mesh_raw: 1 indicates solid buildings, 0 indicates open air
    sigma_np = np.where(crop == 1, 1e8, 0.0).astype(np.float32)
    sigma = torch.from_numpy(sigma_np).unsqueeze(0).unsqueeze(0).to(device)

    obstacle_count = int((sigma > 0).sum().item())
    total_cells = nx * ny * nz
    solid_fraction = obstacle_count / total_cells * 100.0
    print(f"Urban Canopy Mesh Loaded: {obstacle_count:,} solid cells ({solid_fraction:.2f}% building density)")
    return sigma


# =============================================================================
# 4. Tier-3 High-Performance CUDA Solver Wrapper for Buildings
# =============================================================================
class BuildingsTier3CUDASolver:
    """
    Tier-3 GPU-Resident Multigrid & CUDA Graph Solver for Flow Past Buildings:
    - Native boundary conditions for urban canopy (bc_mode=0)
    - Full CUDA Graph capture and hardware-rate replay
    """
    def __init__(self, nx=512, ny=128, nz=128, dx=1.0, dy=1.0, dz=1.0,
                 dt=0.5, nu=0.32, ub=-1.0, diag=88.0/26.0,
                 nlevel=8, iteration=5, sigma=None,
                 enable_cuda_graph=True):
        if not HAS_CFD_CUDA:
            raise RuntimeError("cfd_cuda module not found! Ensure it is compiled.")

        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.dy = dy
        self.dz = dz
        self.dt = dt
        self.nu = nu
        self.ub = ub
        self.diag = float(diag)
        self.nlevel = nlevel
        self.iteration = iteration
        self.enable_cuda_graph = enable_cuda_graph
        self.bc_mode = 0 # Urban Canopy / Buildings Mode

        if sigma is None:
            sigma = torch.zeros((1, 1, nz, ny, nx), device="cuda", dtype=torch.float32)

        self.engine = cfd_cuda.GPUResidentSolverEngine(
            nx, ny, nz, dx, dy, dz, dt, nu, ub, self.diag, nlevel, iteration, sigma, self.bc_mode
        )

        self.cuda_graph = None
        if self.enable_cuda_graph:
            self._init_graph()

    def _init_graph(self):
        """Warmup and capture timestep into CUDA Graph."""
        self.cuda_graph = torch.cuda.CUDAGraph()
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            self.engine.step()
            self.cuda_graph.capture_begin()
            self.engine.step()
            self.cuda_graph.capture_end()
        torch.cuda.current_stream().wait_stream(s)

        device = torch.device("cuda")
        self.engine.set_fields(
            torch.zeros((1, 1, self.nz, self.ny, self.nx), device=device),
            torch.zeros((1, 1, self.nz, self.ny, self.nx), device=device),
            torch.zeros((1, 1, self.nz, self.ny, self.nx), device=device),
            torch.zeros((1, 1, self.nz, self.ny, self.nx), device=device)
        )

    def step(self):
        if self.cuda_graph is not None:
            self.cuda_graph.replay()
        else:
            self.engine.step()

    def get_fields(self):
        return self.engine.get_fields()
