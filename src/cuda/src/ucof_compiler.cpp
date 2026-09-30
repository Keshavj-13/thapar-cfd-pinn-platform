#include "ucof_compiler.h"
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <sys/stat.h>
#include <unistd.h>

UCOFCompiler::UCOFCompiler() {
    CUresult res = cuInit(0);
    if (res == CUDA_SUCCESS) {
        m_driver_initialized = true;
    }
}

UCOFCompiler::~UCOFCompiler() {}

std::string UCOFCompiler::compute_cache_key(const UCOFConfig& cfg, const std::string& stage_name) {
    std::stringstream ss;
    ss << stage_name << "_sm80_"
       << (cfg.enable_heat ? "H1" : "H0")
       << (cfg.enable_turbulence ? "T1" : "T0")
       << (cfg.enable_vof ? "V1" : "V0")
       << "_dt" << cfg.dt << "_nu" << cfg.nu;
    return ss.str();
}

std::string UCOFCompiler::synthesize_stage1_predictor_source(const UCOFConfig& cfg) {
    std::stringstream ss;
    ss << "// Auto-synthesized by Thapar CFD UCOF JIT Compiler for NVIDIA A100 (SM 8.0)\n";
    ss << "#define INV_2DX (" << (0.5f / cfg.dx) << "f)\n";
    ss << "#define INV_2DY (" << (0.5f / cfg.dy) << "f)\n";
    ss << "#define INV_2DZ (" << (0.5f / cfg.dz) << "f)\n";
    ss << "#define INV_DX2 (" << (1.0f / (cfg.dx * cfg.dx)) << "f)\n";
    ss << "#define INV_DY2 (" << (1.0f / (cfg.dy * cfg.dy)) << "f)\n";
    ss << "#define INV_DZ2 (" << (1.0f / (cfg.dz * cfg.dz)) << "f)\n";
    ss << "#define DT (" << cfg.dt << "f)\n";
    ss << "#define NU (" << cfg.nu << "f)\n";

    if (cfg.enable_heat) {
        ss << "#define ENABLE_HEAT 1\n";
        ss << "#define ALPHA_TH (" << cfg.alpha_th << "f)\n";
        ss << "#define BETA (" << cfg.beta << "f)\n";
        ss << "#define T_REF (" << cfg.t_ref << "f)\n";
        ss << "#define GX (" << cfg.gx << "f)\n";
        ss << "#define GY (" << cfg.gy << "f)\n";
        ss << "#define GZ (" << cfg.gz << "f)\n";
    } else {
        ss << "#define ENABLE_HEAT 0\n";
    }

    if (cfg.enable_turbulence) {
        ss << "#define ENABLE_TURB 1\n";
    } else {
        ss << "#define ENABLE_TURB 0\n";
    }

    ss << R"(
extern "C" __global__ void __launch_bounds__(256, 2) ucof_stage1_predictor(
    const float* __restrict__ uu,
    const float* __restrict__ vv,
    const float* __restrict__ ww,
    const float* __restrict__ tt,
    const float* __restrict__ nu_t,
    const float* __restrict__ sigma,
    float* __restrict__ u_star,
    float* __restrict__ v_star,
    float* __restrict__ w_star,
    float* __restrict__ t_star,
    int D, int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < W && y < H && z < D) {
        int pad_W = W + 2;
        int pad_H = H + 2;
        int pad_c = (z + 1) * (pad_H * pad_W) + (y + 1) * pad_W + (x + 1);
        int out_idx = z * (H * W) + y * W + x;

        float nu_eff = NU;
#if ENABLE_TURB
        if (nu_t != nullptr) nu_eff += nu_t[out_idx];
#endif

        float u_c = uu[pad_c]; float v_c = vv[pad_c]; float w_c = ww[pad_c];
        int pad_xm = pad_c - 1;         int pad_xp = pad_c + 1;
        int pad_ym = pad_c - pad_W;     int pad_yp = pad_c + pad_W;
        int pad_zm = pad_c - (pad_H * pad_W); int pad_zp = pad_c + (pad_H * pad_W);

        float du_dx = (uu[pad_xp] - uu[pad_xm]) * INV_2DX;
        float du_dy = (uu[pad_yp] - uu[pad_ym]) * INV_2DY;
        float du_dz = (uu[pad_zp] - uu[pad_zm]) * INV_2DZ;

        float dv_dx = (vv[pad_xp] - vv[pad_xm]) * INV_2DX;
        float dv_dy = (vv[pad_yp] - vv[pad_ym]) * INV_2DY;
        float dv_dz = (vv[pad_zp] - vv[pad_zm]) * INV_2DZ;

        float dw_dx = (ww[pad_xp] - ww[pad_xm]) * INV_2DX;
        float dw_dy = (ww[pad_yp] - ww[pad_ym]) * INV_2DY;
        float dw_dz = (ww[pad_zp] - ww[pad_zm]) * INV_2DZ;

        float lap_u = (uu[pad_xp] - 2.0f * u_c + uu[pad_xm]) * INV_DX2 +
                      (uu[pad_yp] - 2.0f * u_c + uu[pad_ym]) * INV_DY2 +
                      (uu[pad_zp] - 2.0f * u_c + uu[pad_zm]) * INV_DZ2;

        float lap_v = (vv[pad_xp] - 2.0f * v_c + vv[pad_xm]) * INV_DX2 +
                      (vv[pad_yp] - 2.0f * v_c + vv[pad_ym]) * INV_DY2 +
                      (vv[pad_zp] - 2.0f * v_c + vv[pad_zm]) * INV_DZ2;

        float lap_w = (ww[pad_xp] - 2.0f * w_c + ww[pad_xm]) * INV_DX2 +
                      (ww[pad_yp] - 2.0f * w_c + ww[pad_ym]) * INV_DY2 +
                      (ww[pad_zp] - 2.0f * w_c + ww[pad_zm]) * INV_DZ2;

        float adv_u = u_c * du_dx + v_c * du_dy + w_c * du_dz;
        float adv_v = u_c * dv_dx + v_c * dv_dy + w_c * dv_dz;
        float adv_w = u_c * dw_dx + v_c * dw_dy + w_c * dw_dz;

        float buoy_x = 0.0f, buoy_y = 0.0f, buoy_z = 0.0f;
#if ENABLE_HEAT
        if (tt != nullptr) {
            float t_c = tt[pad_c];
            float delta_T = t_c - T_REF;
            buoy_x = GX * BETA * delta_T;
            buoy_y = GY * BETA * delta_T;
            buoy_z = GZ * BETA * delta_T;

            float dt_dx = (tt[pad_xp] - tt[pad_xm]) * INV_2DX;
            float dt_dy = (tt[pad_yp] - tt[pad_ym]) * INV_2DY;
            float dt_dz = (tt[pad_zp] - tt[pad_zm]) * INV_2DZ;
            float adv_t = u_c * dt_dx + v_c * dt_dy + w_c * dt_dz;

            float lap_t = (tt[pad_xp] - 2.0f * t_c + tt[pad_xm]) * INV_DX2 +
                          (tt[pad_yp] - 2.0f * t_c + tt[pad_ym]) * INV_DY2 +
                          (tt[pad_zp] - 2.0f * t_c + tt[pad_zm]) * INV_DZ2;

            if (t_star != nullptr) t_star[out_idx] = t_c + DT * (-adv_t + ALPHA_TH * lap_t);
        }
#endif

        float sig = (sigma != nullptr) ? sigma[out_idx] : 0.0f;
        float solid_damp = 1.0f / (1.0f + DT * sig);

        u_star[out_idx] = (u_c + DT * (-adv_u + nu_eff * lap_u + buoy_x)) * solid_damp;
        v_star[out_idx] = (v_c + DT * (-adv_v + nu_eff * lap_v + buoy_y)) * solid_damp;
        w_star[out_idx] = (w_c + DT * (-adv_w + nu_eff * lap_w + buoy_z)) * solid_damp;
    }
}
)";
    return ss.str();
}

std::string UCOFCompiler::synthesize_stage3_scalar_source(const UCOFConfig& cfg) {
    std::stringstream ss;
    ss << "// Auto-synthesized Stage 3 Scalar Transport Kernel for NVIDIA A100 (SM 8.0)\n";
    ss << "#define INV_2DX (" << (0.5f / cfg.dx) << "f)\n";
    ss << "#define INV_2DY (" << (0.5f / cfg.dy) << "f)\n";
    ss << "#define INV_2DZ (" << (0.5f / cfg.dz) << "f)\n";
    ss << "#define INV_DX2 (" << (1.0f / (cfg.dx * cfg.dx)) << "f)\n";
    ss << "#define INV_DY2 (" << (1.0f / (cfg.dy * cfg.dy)) << "f)\n";
    ss << "#define INV_DZ2 (" << (1.0f / (cfg.dz * cfg.dz)) << "f)\n";
    ss << "#define DT (" << cfg.dt << "f)\n";
    ss << "#define NU (" << cfg.nu << "f)\n";

    ss << R"(
extern "C" __global__ void __launch_bounds__(256, 2) ucof_stage3_scalar(
    const float* __restrict__ uu,
    const float* __restrict__ vv,
    const float* __restrict__ ww,
    const float* __restrict__ nu_tilde_pad,
    const float* __restrict__ alpha_pad,
    const float* __restrict__ wall_dist,
    float* __restrict__ nu_tilde_star,
    float* __restrict__ nu_t_out,
    float* __restrict__ alpha_star,
    float* __restrict__ rho_mix,
    float* __restrict__ nu_mix,
    int D, int H, int W)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int z = blockIdx.z * blockDim.z + threadIdx.z;

    if (x < W && y < H && z < D) {
        int pad_W = W + 2;
        int pad_H = H + 2;
        int pad_c = (z + 1) * (pad_H * pad_W) + (y + 1) * pad_W + (x + 1);
        int out_idx = z * (H * W) + y * W + x;

        float u_c = uu[pad_c]; float v_c = vv[pad_c]; float w_c = ww[pad_c];
        int pad_xm = pad_c - 1;         int pad_xp = pad_c + 1;
        int pad_ym = pad_c - pad_W;     int pad_yp = pad_c + pad_W;
        int pad_zm = pad_c - (pad_H * pad_W); int pad_zp = pad_c + (pad_H * pad_W);

        // VoF Advection with OpenFOAM interface compression
        if (alpha_pad != nullptr) {
            float a_c = alpha_pad[pad_c];
            float da_dx = (alpha_pad[pad_xp] - alpha_pad[pad_xm]) * INV_2DX;
            float da_dy = (alpha_pad[pad_yp] - alpha_pad[pad_ym]) * INV_2DY;
            float da_dz = (alpha_pad[pad_zp] - alpha_pad[pad_zm]) * INV_2DZ;
            float adv_a = u_c * da_dx + v_c * da_dy + w_c * da_dz;

            float a_new = a_c - DT * adv_a;
            a_new = fminf(fmaxf(a_new, 0.0f), 1.0f);
            if (alpha_star != nullptr) alpha_star[out_idx] = a_new;
            if (rho_mix != nullptr) rho_mix[out_idx] = a_new * 1000.0f + (1.0f - a_new) * 1.0f;
        }

        // Spalart-Allmaras Transport
        if (nu_tilde_pad != nullptr) {
            float nt_c = nu_tilde_pad[pad_c];
            float dnt_dx = (nu_tilde_pad[pad_xp] - nu_tilde_pad[pad_xm]) * INV_2DX;
            float dnt_dy = (nu_tilde_pad[pad_yp] - nu_tilde_pad[pad_ym]) * INV_2DY;
            float dnt_dz = (nu_tilde_pad[pad_zp] - nu_tilde_pad[pad_zm]) * INV_2DZ;
            float adv_nt = u_c * dnt_dx + v_c * dnt_dy + w_c * dnt_dz;

            float lap_nt = (nu_tilde_pad[pad_xp] - 2.0f * nt_c + nu_tilde_pad[pad_xm]) * INV_DX2 +
                           (nu_tilde_pad[pad_yp] - 2.0f * nt_c + nu_tilde_pad[pad_ym]) * INV_DY2 +
                           (nu_tilde_pad[pad_zp] - 2.0f * nt_c + nu_tilde_pad[pad_zm]) * INV_DZ2;

            float nt_new = nt_c + DT * (-adv_nt + 1.5f * NU * lap_nt);
            nt_new = fmaxf(nt_new, 0.0f);
            if (nu_tilde_star != nullptr) nu_tilde_star[out_idx] = nt_new;
            if (nu_t_out != nullptr) nu_t_out[out_idx] = nt_new;
        }
    }
}
)";
    return ss.str();
}

bool UCOFCompiler::compile_to_ptx(
    const std::string& cuda_source,
    const std::string& kernel_name,
    std::string& ptx_out,
    std::string& log_out)
{
    nvrtcProgram prog;
    nvrtcResult res = nvrtcCreateProgram(&prog, cuda_source.c_str(), kernel_name.c_str(), 0, NULL, NULL);
    if (res != NVRTC_SUCCESS) {
        log_out = nvrtcGetErrorString(res);
        return false;
    }

    const char* opts[] = {
        "--gpu-architecture=compute_80",
        "-O3",
        "--use_fast_math",
        "--extra-device-vectorization"
    };
    res = nvrtcCompileProgram(prog, 4, opts);

    size_t log_size;
    nvrtcGetProgramLogSize(prog, &log_size);
    if (log_size > 1) {
        std::vector<char> log_buf(log_size);
        nvrtcGetProgramLog(prog, log_buf.data());
        log_out = log_buf.data();
    }

    if (res != NVRTC_SUCCESS) {
        nvrtcDestroyProgram(&prog);
        return false;
    }

    size_t ptx_size;
    nvrtcGetPTXSize(prog, &ptx_size);
    std::vector<char> ptx_buf(ptx_size);
    nvrtcGetPTX(prog, ptx_buf.data());
    ptx_out = std::string(ptx_buf.data(), ptx_size);

    nvrtcDestroyProgram(&prog);
    return true;
}

bool UCOFCompiler::load_ptx_module(
    const std::string& ptx,
    CUmodule& module,
    CUfunction& function,
    const char* func_name)
{
    if (!m_driver_initialized) {
        cuInit(0);
        m_driver_initialized = true;
    }

    CUresult res = cuModuleLoadData(&module, ptx.c_str());
    if (res != CUDA_SUCCESS) {
        std::cerr << "Failed to load PTX module: " << res << std::endl;
        return false;
    }

    res = cuModuleGetFunction(&function, module, func_name);
    if (res != CUDA_SUCCESS) {
        std::cerr << "Failed to get function from module: " << res << std::endl;
        return false;
    }

    return true;
}
