#!/usr/bin/env python3
"""
tumor_multiphysics_models.py — Progressive Multi-Species Biological & Tumor Microenvironment Models
Built on top of custom_pde_engine.py.

Stages of Biological Complexity:
1. Stage 1: Coupled Morphogenesis & Chemotaxis (Gray-Scott Turing Spots + Keller-Segel Cellular Drift)
2. Stage 2: 3D Avascular Tumor Spheroid with Hypoxic-Necrotic Zonation (Greenspan 1972 Benchmark)
3. Stage 3: 3D Vascularized Glioblastoma Microenvironment (Angiogenic Sprouting, Darcy IFP Hypertension, & ECM Remodeling)
"""

import math
import numpy as np
import torch
from custom_pde_engine import PDEProblemSpec, FieldSpec, ChemotaxisSpec, DarcySpec


# =============================================================================
# Stage 1: Morphogenesis & Chemotaxis (Turing Patterning + Cellular Drift)
# =============================================================================
def build_morphogenesis_chemotaxis_problem(F: float = 0.034, k: float = 0.065) -> PDEProblemSpec:
    """
    Stage 1: Coupled reaction-diffusion-chemotaxis model.
    A chemical signaling field (u, v) undergoes autocatalytic pattern formation (Gray-Scott),
    while a mobile cellular population (cells) migrates toward the activator peaks via chemotaxis:
      du/dt = D_u * grad^2(u) - u*v^2 + F*(1 - u)
      dv/dt = D_v * grad^2(v) + u*v^2 - (F + k)*v
      dn/dt = D_n * grad^2(n) - div(chi * n * grad(v)) + mu * n * (1 - n)
    """
    def kinetics(state, t):
        u = state["activator"]
        v = state["inhibitor"]
        n = state["cells"]

        uv2 = u * (v ** 2)
        r_u = -uv2 + F * (1.0 - u)
        r_v = uv2 - (F + k) * v
        # Logistic cellular division
        r_n = 0.02 * n * (1.0 - n)

        return {"activator": r_u, "inhibitor": r_v, "cells": r_n}

    fields = {
        "activator": FieldSpec(
            name="activator", diffusivity=0.16,
            bc_type={"x": "neumann", "y": "neumann", "z": "neumann"}
        ),
        "inhibitor": FieldSpec(
            name="inhibitor", diffusivity=0.08,
            bc_type={"x": "neumann", "y": "neumann", "z": "neumann"}
        ),
        "cells": FieldSpec(
            name="cells", diffusivity=0.03,
            bc_type={"x": "neumann", "y": "neumann", "z": "neumann"},
            chemotaxis=[ChemotaxisSpec(target_field="inhibitor", chi=0.45, saturation_k=0.5)],
            clip_min=0.0, clip_max=2.0
        )
    }

    return PDEProblemSpec(
        name="Stage1_Morphogenesis_Chemotaxis",
        description="Coupled Gray-Scott Turing Reaction-Diffusion with Keller-Segel Chemotactic Cellular Migration",
        fields=fields,
        reaction_kinetics=kinetics
    )


# =============================================================================
# Stage 2: 3D Avascular Tumor Spheroid with Hypoxic-Necrotic Zonation
# Canonical Greenspan (1972) / McElwain & Ponzo (1977) Benchmark
# =============================================================================
def build_avascular_spheroid_problem(
    mu_p: float = 0.25,        # Proliferation rate of viable cells (1/day)
    mu_d: float = 0.10,        # Necrotic disintegration / core shrinkage rate
    gamma_c: float = 0.60,     # Oxygen consumption rate
    c_hypox: float = 0.45,     # Critical oxygen threshold for hypoxia / mitotic arrest
    c_necro: float = 0.15,     # Critical oxygen threshold for necrosis / anoxic lysis
    K_m: float = 0.10          # Michaelis-Menten half-saturation constant for nutrient
) -> PDEProblemSpec:
    """
    Stage 2: 3D Avascular Tumor Spheroid.
    Captures the three canonical concentric radial layers:
      1. Outer Proliferating Rim (c > c_hypox): Exponential mitotic growth, high nutrient consumption.
      2. Intermediate Hypoxic Quiescent Layer (c_necro < c <= c_hypox): Growth-arrested viable cells.
      3. Central Necrotic Core (c <= c_necro): Lysed cellular debris and apoptotic fragmentation.

    Validates against Greenspan (1972) steady-state necrotic radius analytical benchmark:
      R_n = R * sqrt( 1 - 2*D_c*(c_ext - c_necro) / (gamma * R^2) )
    """
    def kinetics(state, t):
        p = state["proliferating"]
        h = state["hypoxic"]
        n = state["necrotic"]
        c = state["oxygen"]

        # Total cellular density (carrying capacity = 1.0)
        phi_tot = p + h + n
        space_factor = torch.clamp(1.0 - phi_tot, min=0.0)

        # Monod / Michaelis-Menten oxygen utilization
        util = c / (c + K_m)

        # Transition masks based on microenvironmental oxygenation
        mask_prolif = torch.sigmoid(20.0 * (c - c_hypox))
        mask_hypox = torch.sigmoid(20.0 * (c - c_necro)) * (1.0 - mask_prolif)
        mask_anoxia = 1.0 - torch.sigmoid(20.0 * (c - c_necro))

        # Rates:
        # Proliferation -> Hypoxia
        trans_p_to_h = 0.15 * p * (1.0 - mask_prolif)
        # Hypoxia -> Necrosis
        trans_h_to_n = 0.20 * h * mask_anoxia

        # Proliferation occurs only in viable outer zone under sufficient oxygen
        growth_p = mu_p * util * p * space_factor * mask_prolif

        # Hypoxic recovery if oxygen returns
        recovery_h = 0.05 * h * mask_prolif

        dp_dt = growth_p - trans_p_to_h + recovery_h
        dh_dt = trans_p_to_h - trans_h_to_n - recovery_h
        dn_dt = trans_h_to_n - mu_d * n

        # Oxygen consumption: rapid uptake by proliferating, basal by hypoxic, 0 by necrotic
        dc_dt = - gamma_c * (p + 0.3 * h) * util

        return {
            "oxygen": dc_dt,
            "proliferating": dp_dt,
            "hypoxic": dh_dt,
            "necrotic": dn_dt
        }

    fields = {
        "oxygen": FieldSpec(
            name="oxygen", diffusivity=0.80,
            bc_type={"x": "dirichlet", "y": "dirichlet", "z": "dirichlet"},
            bc_values={"x0": 1.0, "x1": 1.0, "y0": 1.0, "y1": 1.0, "z0": 1.0, "z1": 1.0},
            clip_min=0.0, clip_max=1.0
        ),
        "proliferating": FieldSpec(
            name="proliferating", diffusivity=0.02,
            bc_type={"x": "neumann", "y": "neumann", "z": "neumann"},
            clip_min=0.0, clip_max=1.0
        ),
        "hypoxic": FieldSpec(
            name="hypoxic", diffusivity=0.005,
            bc_type={"x": "neumann", "y": "neumann", "z": "neumann"},
            clip_min=0.0, clip_max=1.0
        ),
        "necrotic": FieldSpec(
            name="necrotic", diffusivity=0.001,
            bc_type={"x": "neumann", "y": "neumann", "z": "neumann"},
            clip_min=0.0, clip_max=1.0
        )
    }

    return PDEProblemSpec(
        name="Stage2_Avascular_Tumor_Spheroid",
        description="3D Avascular Spheroid Zonation with Greenspan (1972) Necrotic Core Benchmark",
        fields=fields,
        reaction_kinetics=kinetics
    )


# =============================================================================
# Stage 3: 3D Vascularized Glioblastoma Microenvironment
# Full Multiphysics: Angiogenesis, Darcy Interstitial Hypertension & ECM Remodeling
# =============================================================================
def build_vascular_glioblastoma_microenvironment_problem(
    k_darcy: float = 2e-4,      # Darcy hydraulic conductivity
    p_micro: float = 28.0,      # Microvascular perfusion pressure (mmHg)
    alpha_vegf: float = 0.35,   # VEGF secretion rate by hypoxic tumor cells
    chi_vessel: float = 0.40,   # Chemotactic sensitivity of capillary tips to VEGF
    mmp_secretion: float = 0.15 # Matrix metalloproteinase (MMP) ECM degradation rate
) -> PDEProblemSpec:
    """
    Stage 3: Full 3D Multiphysics Glioblastoma Tumor Microenvironment.
    Simulates:
      1. Proliferating Glioma Core (p): Migrates, consumes nutrient, degrades ECM via MMPs.
      2. Hypoxic Quiescent Halo (h): Triggers HIF-1alpha stabilization, secreting pro-angiogenic VEGF.
      3. Central Necrotic Debris (n): Anoxic pseudopalisading necrotic core.
      4. Oxygen / Glucose (c): Diffuses from host vasculature, depleted inside dense tumor tissue.
      5. VEGF Signaling Factor (v): Angiogenic cue diffusing outwards from hypoxic core.
      6. Endothelial Capillary Network (b): Sprouts via chemotaxis along grad(v) toward hypoxic zone.
      7. Extracellular Matrix / Stroma (m): Barrier matrix degraded by tumor MMP enzymes.
      8. Darcy Interstitial Fluid Pressure (P_IFP): High fluid hypertension in core, outward convective seepage.
    """
    def kinetics(state, t):
        p = state["proliferating"]
        h = state["hypoxic"]
        n = state["necrotic"]
        c = state["oxygen"]
        v = state["vegf"]
        b = state["vessels"]
        m = state["ecm"]

        phi_tot = p + h + n
        space_free = torch.clamp(1.0 - phi_tot, min=0.0)
        util = c / (c + 0.12)

        # Hypoxia / Necrosis switches
        is_prolif = torch.sigmoid(22.0 * (c - 0.42))
        is_hypox = torch.sigmoid(22.0 * (c - 0.14)) * (1.0 - is_prolif)
        is_necro = 1.0 - torch.sigmoid(22.0 * (c - 0.14))

        # Tumor Kinetics: Proliferation, Death, Transitions
        growth_p = 0.28 * util * p * space_free * (0.3 + 0.7 * m) * is_prolif
        death_p = 0.18 * p * (1.0 - is_prolif)
        death_h = 0.22 * h * is_necro
        recover_h = 0.08 * h * is_prolif

        dp = growth_p - death_p + recover_h
        dh = death_p - death_h - recover_h
        dn = death_h - 0.06 * n

        # Angiogenic VEGF Factor:
        # Heavily secreted by hypoxic cells (h), natural decay, absorbed by sprouting endothelial vessels (b)
        prod_vegf = alpha_vegf * h * (1.0 - util)
        decay_vegf = 0.10 * v + 0.25 * b * v
        dv = prod_vegf - decay_vegf

        # Capillary Vasculature (b):
        # Sprouting proliferation induced by VEGF signaling
        growth_b = 0.20 * b * (v / (v + 0.15)) * (1.0 - b)
        decay_b = 0.03 * b * (1.0 - is_prolif) # regression inside necrotic core due to solid stress collapse
        db = growth_b - decay_b

        # Extracellular Matrix (m):
        # Degraded by tumor invasive enzymes (MMP secretion proportional to proliferating tumor p)
        dm = - mmp_secretion * p * m + 0.02 * (1.0 - m) * space_free

        # Oxygen (c):
        # Supplied by microvessels (b), consumed by tumor cells
        supply_c = 0.65 * b * (1.0 - c)
        uptake_c = (0.50 * p + 0.15 * h) * util
        dc = supply_c - uptake_c

        return {
            "oxygen": dc,
            "proliferating": dp,
            "hypoxic": dh,
            "necrotic": dn,
            "vegf": dv,
            "vessels": db,
            "ecm": dm
        }

    # Chemotaxis:
    # 1. Capillary vessels (b) migrate toward VEGF (v) gradient: J = chi_b * b * grad(v)
    # 2. Proliferating glioma cells (p) migrate toward oxygen (c) gradient: J = chi_p * p * grad(c)
    fields = {
        "oxygen": FieldSpec(
            name="oxygen", diffusivity=0.75,
            bc_type={"x": "dirichlet", "y": "dirichlet", "z": "dirichlet"},
            bc_values={"x0": 0.9, "x1": 0.9, "y0": 0.9, "y1": 0.9, "z0": 0.9, "z1": 0.9},
            clip_min=0.0, clip_max=1.0
        ),
        "proliferating": FieldSpec(
            name="proliferating", diffusivity=0.03,
            bc_type={"x": "neumann", "y": "neumann", "z": "neumann"},
            chemotaxis=[ChemotaxisSpec(target_field="oxygen", chi=0.15, saturation_k=0.3)],
            advection_by_darcy=True,
            clip_min=0.0, clip_max=1.0
        ),
        "hypoxic": FieldSpec(
            name="hypoxic", diffusivity=0.006,
            bc_type={"x": "neumann", "y": "neumann", "z": "neumann"},
            clip_min=0.0, clip_max=1.0
        ),
        "necrotic": FieldSpec(
            name="necrotic", diffusivity=0.001,
            bc_type={"x": "neumann", "y": "neumann", "z": "neumann"},
            clip_min=0.0, clip_max=1.0
        ),
        "vegf": FieldSpec(
            name="vegf", diffusivity=0.35,
            bc_type={"x": "neumann", "y": "neumann", "z": "neumann"},
            clip_min=0.0, clip_max=2.0
        ),
        "vessels": FieldSpec(
            name="vessels", diffusivity=0.02,
            bc_type={"x": "neumann", "y": "neumann", "z": "neumann"},
            chemotaxis=[ChemotaxisSpec(target_field="vegf", chi=chi_vessel, saturation_k=0.2)],
            clip_min=0.0, clip_max=1.0
        ),
        "ecm": FieldSpec(
            name="ecm", diffusivity=0.005,
            bc_type={"x": "neumann", "y": "neumann", "z": "neumann"},
            clip_min=0.0, clip_max=1.0
        )
    }

    darcy = DarcySpec(
        pressure_field_name="pressure",
        hydraulic_conductivity=k_darcy,
        vascular_permeability_coeff=0.08,
        lymphatic_drainage_coeff=0.04,
        microvascular_pressure=p_micro,
        lymphatic_pressure=0.0,
        poisson_iters=25
    )

    return PDEProblemSpec(
        name="Stage3_Vascular_Glioblastoma_Microenvironment",
        description="3D Vascularized Glioblastoma with Angiogenesis, Darcy Interstitial Hypertension, & ECM Remodeling",
        fields=fields,
        reaction_kinetics=kinetics,
        darcy_spec=darcy
    )
