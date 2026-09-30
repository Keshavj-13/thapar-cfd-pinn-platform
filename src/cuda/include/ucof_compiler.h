#ifndef UCOF_COMPILER_H
#define UCOF_COMPILER_H

#include <string>
#include <vector>
#include <memory>
#include <cuda.h>
#include <nvrtc.h>

struct UCOFConfig {
    bool enable_heat = false;
    bool enable_turbulence = false;
    bool enable_vof = false;
    
    int nx = 64, ny = 64, nz = 64;
    float dx = 0.01f, dy = 0.01f, dz = 0.01f;
    float dt = 0.001f;
    float nu = 0.01f;
    float rho = 1.0f;
    
    // Thermal
    float alpha_th = 0.01f;
    float beta = 0.003f;
    float t_ref = 293.15f;
    float gx = 0.0f, gy = -9.81f, gz = 0.0f;
    
    // Turbulence (Spalart-Allmaras)
    float sa_cb1 = 0.1355f;
    float sa_cb2 = 0.622f;
    float sa_sigma = 0.66667f;
    float sa_kappa = 0.41f;
    
    // VoF
    float rho1 = 1000.0f, rho2 = 1.0f;
    float nu1 = 1e-6f, nu2 = 1.5e-5f;
    float c_alpha = 1.0f;
};

class UCOFCompiler {
public:
    UCOFCompiler();
    ~UCOFCompiler();

    // Synthesizes specialized CUDA C++ macro-kernel AST string with inlined physical constants
    std::string synthesize_stage1_predictor_source(const UCOFConfig& cfg);
    std::string synthesize_stage3_scalar_source(const UCOFConfig& cfg);

    // Compiles specialized source to PTX via NVRTC targeting Ampere SM 8.0
    bool compile_to_ptx(const std::string& cuda_source, const std::string& kernel_name, std::string& ptx_out, std::string& log_out);

    // Loads compiled PTX into CUDA Driver Module
    bool load_ptx_module(const std::string& ptx, CUmodule& module, CUfunction& function, const char* func_name);

    // Computes unique cache key for configuration
    std::string compute_cache_key(const UCOFConfig& cfg, const std::string& stage_name);

private:
    bool m_driver_initialized = false;
};

#endif /* UCOF_COMPILER_H */
