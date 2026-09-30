#ifndef GPU_ARRAY_3D_CUH
#define GPU_ARRAY_3D_CUH

#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <utility>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        char buf[256]; \
        snprintf(buf, sizeof(buf), "CUDA Error at %s:%d - %s (%s)", \
                 __FILE__, __LINE__, cudaGetErrorString(err), #call); \
        fprintf(stderr, "%s\n", buf); \
        throw std::runtime_error(buf); \
    } \
} while (0)

template <typename T>
class GPUArray3D {
private:
    T* d_data;
    int nx, ny, nz;
    size_t count;
    size_t bytes;

public:
    GPUArray3D() : d_data(nullptr), nx(0), ny(0), nz(0), count(0), bytes(0) {}

    GPUArray3D(int x, int y, int z) : d_data(nullptr), nx(x), ny(y), nz(z) {
        allocate(x, y, z);
    }

    ~GPUArray3D() {
        free();
    }

    // Disable copy semantics to prevent accidental device pointer aliasing
    GPUArray3D(const GPUArray3D&) = delete;
    GPUArray3D& operator=(const GPUArray3D&) = delete;

    // Move constructor
    GPUArray3D(GPUArray3D&& other) noexcept 
        : d_data(other.d_data), nx(other.nx), ny(other.ny), nz(other.nz),
          count(other.count), bytes(other.bytes) {
        other.d_data = nullptr;
        other.nx = other.ny = other.nz = 0;
        other.count = other.bytes = 0;
    }

    // Move assignment
    GPUArray3D& operator=(GPUArray3D&& other) noexcept {
        if (this != &other) {
            free();
            d_data = other.d_data;
            nx = other.nx;
            ny = other.ny;
            nz = other.nz;
            count = other.count;
            bytes = other.bytes;

            other.d_data = nullptr;
            other.nx = other.ny = other.nz = 0;
            other.count = other.bytes = 0;
        }
        return *this;
    }

    void allocate(int x, int y, int z) {
        if (d_data != nullptr) {
            free();
        }
        nx = x; ny = y; nz = z;
        count = static_cast<size_t>(nx) * static_cast<size_t>(ny) * static_cast<size_t>(nz);
        bytes = count * sizeof(T);
        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_data), bytes));
        zero();
    }

    void free() {
        if (d_data != nullptr) {
            cudaFree(d_data);
            d_data = nullptr;
        }
        nx = ny = nz = 0;
        count = bytes = 0;
    }

    void zero(cudaStream_t stream = 0) {
        if (d_data != nullptr && bytes > 0) {
            CUDA_CHECK(cudaMemsetAsync(d_data, 0, bytes, stream));
        }
    }

    void copy_from_host(const T* h_ptr, cudaStream_t stream = 0) {
        if (!h_ptr || !d_data) return;
        CUDA_CHECK(cudaMemcpyAsync(d_data, h_ptr, bytes, cudaMemcpyHostToDevice, stream));
    }

    void copy_to_host(T* h_ptr, cudaStream_t stream = 0) const {
        if (!h_ptr || !d_data) return;
        CUDA_CHECK(cudaMemcpyAsync(h_ptr, d_data, bytes, cudaMemcpyDeviceToHost, stream));
    }

    void copy_from_device(const GPUArray3D<T>& src, cudaStream_t stream = 0) {
        if (count != src.count) {
            throw std::runtime_error("GPUArray3D shape mismatch during device copy");
        }
        CUDA_CHECK(cudaMemcpyAsync(d_data, src.d_data, bytes, cudaMemcpyDeviceToDevice, stream));
    }

    __host__ __device__ T* data() { return d_data; }
    __host__ __device__ const T* data() const { return d_data; }

    __host__ __device__ int get_nx() const { return nx; }
    __host__ __device__ int get_ny() const { return ny; }
    __host__ __device__ int get_nz() const { return nz; }
    __host__ __device__ size_t get_count() const { return count; }
    __host__ __device__ size_t get_bytes() const { return bytes; }
};

#endif /* GPU_ARRAY_3D_CUH */
