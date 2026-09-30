cmake_minimum_required(VERSION 3.20)
project(thapar_cfd_cuda LANGUAGES C CXX CUDA)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Target NVIDIA A100-SXM4 (Ampere Compute Capability 8.0)
set(CMAKE_CUDA_ARCHITECTURES 80)

# Optimization and Microarchitectural Tuning flags
add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-O3;--use_fast_math;-diag-suppress=20044>")

# Header search paths
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

find_package(CUDAToolkit REQUIRED)

# 1. Main Shared Library: libthaparcfd.so (C ABI Decoupled Core)
add_library(thaparcfd SHARED
    src/thapar_cfd_engine.cu
    src/ucof_compiler.cpp
)

target_link_libraries(thaparcfd PRIVATE
    CUDA::nvrtc
    CUDA::cuda_driver
    m
)

set_target_properties(thaparcfd PROPERTIES
    CUDA_SEPARABLE_COMPILATION OFF
    SOVERSION 1
    VERSION 1.0.0
)

# 2. Standalone Verification Executables
# Task 1: C ABI Test (Pure C executable dynamically linking libthaparcfd.so)
add_executable(test_c_abi
    tests/test_c_abi.c
)
target_link_libraries(test_c_abi PRIVATE thaparcfd m)

# Task 3: Taylor-Green Vortex Gate Test (Anisotropic metric tensor & 2nd order convergence)
add_executable(test_taylor_green
    tests/test_taylor_green.cu
)
target_link_libraries(test_taylor_green PRIVATE thaparcfd m)

# Task 4: Modular Boundary Patch & Flux Conservation Gate Test
add_executable(test_boundary_flux
    tests/test_boundary_flux.cu
)
target_link_libraries(test_boundary_flux PRIVATE thaparcfd m)

# Task 5: 3D Multigrid Poisson Convergence Gate Test (rho <= 0.15)
add_executable(test_multigrid_poisson
    tests/test_multigrid_poisson.cu
)
target_link_libraries(test_multigrid_poisson PRIVATE thaparcfd m)

# Phase 2: Natural Convection Ra=10^5 Gate Test (de Vahl Davis Benchmark)
add_executable(test_natural_convection
    tests/test_natural_convection.cu
)
target_link_libraries(test_natural_convection PRIVATE thaparcfd m)

# Phase 2: Multiphase VoF Dam Break Monotonicity & Conservation Gate Test
add_executable(test_dam_break_vof
    tests/test_dam_break_vof.cu
)
target_link_libraries(test_dam_break_vof PRIVATE thaparcfd m)

# Phase 2: Spalart-Allmaras Turbulence Positivity Gate Test
add_executable(test_spalart_allmaras
    tests/test_spalart_allmaras.cu
)
target_link_libraries(test_spalart_allmaras PRIVATE thaparcfd m)

# Performance Benchmark: GPU-Resident 3D Lid-Driven Cavity vs OpenFOAM icoFoam
add_executable(bench_lid_cavity
    tests/bench_lid_cavity.c
)
target_link_libraries(bench_lid_cavity PRIVATE thaparcfd m)

# Performance Benchmark: GPU-Resident Natural Convection (Ra=10^5) vs OpenFOAM buoyantBoussinesqPimpleFoam
add_executable(bench_natural_convection
    tests/bench_natural_convection.c
)
target_link_libraries(bench_natural_convection PRIVATE thaparcfd m)
