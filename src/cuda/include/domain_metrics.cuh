#ifndef DOMAIN_METRICS_CUH
#define DOMAIN_METRICS_CUH

#include <cuda_runtime.h>

struct DomainMetrics {
    int nx, ny, nz;
    int pad_nx, pad_ny, pad_nz;
    
    float dx, dy, dz;
    float dt;
    float nu;
    float rho;
    float ub;
    
    // Inverse metric factors
    float inv_dx, inv_dy, inv_dz;
    float inv_2dx, inv_2dy, inv_2dz;
    float inv_dx2, inv_dy2, inv_dz2;
    
    // 27-point stencil diagonal
    float diag;
    float inv_diag;
    float jacobi_omega;

    // Multigrid levels
    int nlevel;
    int mg_iterations;

    // Multiphysics Flags
    int enable_heat;
    int enable_turbulence;
    int enable_vof;

    // Thermal Parameters (Boussinesq Natural Convection)
    float thermal_diffusivity;
    float beta_thermal;
    float t_ref;
    float gx, gy, gz;

    // Turbulence Parameters (Spalart-Allmaras)
    float sa_cb1;
    float sa_cb2;
    float sa_sigma_inv;
    float sa_cv1_3;
    float sa_cw2;
    float sa_cw3;
    float sa_kappa2;

    // Multiphase VoF Parameters
    float rho1, rho2;
    float nu1, nu2;
    float sigma_tension;
    float c_alpha;
};

// Device constant memory symbol accessible across all kernels
#ifdef DEFINE_DOMAIN_METRICS
__constant__ DomainMetrics c_metrics;
#else
extern __constant__ DomainMetrics c_metrics;
#endif

#endif /* DOMAIN_METRICS_CUH */
