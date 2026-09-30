#ifndef THAPAR_CFD_ENGINE_H
#define THAPAR_CFD_ENGINE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Status / Return Codes */
typedef enum {
    THAPAR_STATUS_SUCCESS = 0,
    THAPAR_STATUS_INVALID_ARGUMENT = -1,
    THAPAR_STATUS_CUDA_ERROR = -2,
    THAPAR_STATUS_OUT_OF_MEMORY = -3,
    THAPAR_STATUS_GRAPH_ERROR = -4,
    THAPAR_STATUS_NUMERICAL_DIVERGENCE = -5,
    THAPAR_STATUS_FILE_NOT_FOUND = -6,
    THAPAR_STATUS_NOT_INITIALIZED = -7
} thapar_status_t;

/* Field Types for Extraction */
typedef enum {
    THAPAR_FIELD_U = 0,
    THAPAR_FIELD_V = 1,
    THAPAR_FIELD_W = 2,
    THAPAR_FIELD_VELOCITY_MAGNITUDE = 3,
    THAPAR_FIELD_PRESSURE = 4,
    THAPAR_FIELD_TEMPERATURE = 5,
    THAPAR_FIELD_TURBULENT_VISCOSITY = 6,
    THAPAR_FIELD_VOF_ALPHA = 7,
    THAPAR_FIELD_OBSTACLE_MASK = 8,
    THAPAR_FIELD_DIVERGENCE = 9
} thapar_field_type_t;

/* Coordinate Axes */
typedef enum {
    THAPAR_AXIS_X = 0, /* Slices in Y-Z plane */
    THAPAR_AXIS_Y = 1, /* Slices in X-Z plane */
    THAPAR_AXIS_Z = 2  /* Slices in X-Y plane */
} thapar_axis_t;

/* Boundary Types */
typedef enum {
    THAPAR_BC_NO_SLIP_WALL = 0,
    THAPAR_BC_FREE_SLIP_SYMMETRY = 1,
    THAPAR_BC_DIRICHLET_INFLOW = 2,
    THAPAR_BC_NEUMANN_OUTFLOW = 3,
    THAPAR_BC_MOVING_LID = 4,
    THAPAR_BC_PERIODIC = 5,
    THAPAR_BC_ISOTHERMAL_WALL = 6
} thapar_bc_type_t;

/* Boundary Specification */
typedef struct {
    thapar_bc_type_t type;
    float u_val;
    float v_val;
    float w_val;
    float p_val;
    float t_val;
} thapar_patch_bc_t;

/* Domain Configuration */
typedef struct {
    int nx, ny, nz;
    float dx, dy, dz;
    float dt;
    float nu;
    float rho;
    float ub;
    int nlevel;
    int mg_iterations;
    int enable_cuda_graph;
    int enable_heat;
    int enable_turbulence;
    int enable_vof;

    /* Multiphysics Physical Parameters */
    float thermal_diffusivity;
    float beta_thermal;
    float t_ref;
    float gx, gy, gz;

    float rho1, rho2;
    float nu1, nu2;
    float sigma_tension;
    float c_alpha;
    
    /* 6 Outer Boundaries: xmin, xmax, ymin, ymax, zmin, zmax */
    thapar_patch_bc_t bc_xmin;
    thapar_patch_bc_t bc_xmax;
    thapar_patch_bc_t bc_ymin;
    thapar_patch_bc_t bc_ymax;
    thapar_patch_bc_t bc_zmin;
    thapar_patch_bc_t bc_zmax;
} thapar_config_t;

/* Solver Engine Opaque Handle */
typedef struct thapar_solver_s thapar_solver_t;

/* Lifecycle & Execution Functions */
thapar_solver_t* thapar_create_solver(const thapar_config_t* config);
thapar_solver_t* thapar_create_solver_from_json(const char* sim_spec_json);
thapar_status_t  thapar_step(thapar_solver_t* solver);
thapar_status_t  thapar_step_multiple(thapar_solver_t* solver, int num_steps);
void             thapar_destroy_solver(thapar_solver_t* solver);

/* Field & Slice Access */
thapar_status_t  thapar_get_slice(
    thapar_solver_t* solver,
    thapar_field_type_t field_type,
    thapar_axis_t axis,
    int index,
    float* out_buffer,
    size_t buffer_size);

thapar_status_t  thapar_get_field_3d(
    thapar_solver_t* solver,
    thapar_field_type_t field_type,
    float* out_buffer,
    size_t buffer_size);

thapar_status_t  thapar_set_field_3d(
    thapar_solver_t* solver,
    thapar_field_type_t field_type,
    const float* host_buffer,
    size_t count);

thapar_status_t  thapar_set_obstacle_mask(
    thapar_solver_t* solver,
    const float* host_mask,
    size_t count);

/* Diagnostics & Verification */
thapar_status_t  thapar_get_diagnostics(
    thapar_solver_t* solver,
    float* out_max_divergence,
    float* out_pressure_residual,
    float* out_mass_flux_error,
    size_t* out_vram_allocated_bytes,
    double* out_last_step_ms);

const char*      thapar_get_last_error(thapar_solver_t* solver);

#ifdef __cplusplus
}
#endif

#endif /* THAPAR_CFD_ENGINE_H */
