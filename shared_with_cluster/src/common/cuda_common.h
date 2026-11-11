#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

#include <cuda_runtime.h>
#include <cusparse.h>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

#define CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\"\n", \
                __FILE__, __LINE__, err, cudaGetErrorString(err), #call); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

#define CUSPARSE_CHECK(call) do { \
    cusparseStatus_t status = (call); \
    if (status != CUSPARSE_STATUS_SUCCESS) { \
        fprintf(stderr, "cuSPARSE error at %s:%d code=%d \"%s\"\n", \
                __FILE__, __LINE__, (int)status, #call); \
        exit(EXIT_FAILURE); \
    } \
} while (0)

class GpuTimer {
public:
    GpuTimer() {
        CUDA_CHECK(cudaEventCreate(&start_evt));
        CUDA_CHECK(cudaEventCreate(&stop_evt));
    }
    ~GpuTimer() {
        cudaEventDestroy(start_evt);
        cudaEventDestroy(stop_evt);
    }
    void start(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(start_evt, stream));
    }
    void stop(cudaStream_t stream = 0) {
        CUDA_CHECK(cudaEventRecord(stop_evt, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_evt));
    }
    float elapsed_ms() {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start_evt, stop_evt));
        return ms;
    }
private:
    cudaEvent_t start_evt;
    cudaEvent_t stop_evt;
};

// Architecture-independent double-precision atomicAdd fallback for CC < 6.0
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 600
static __device__ __forceinline__ double atomicAdd(double* address, double val) {
    unsigned long long int* address_as_ull = (unsigned long long int*)address;
    unsigned long long int old = *address_as_ull, assumed;
    do {
        assumed = old;
        old = atomicCAS(address_as_ull, assumed,
                        __double_as_longlong(val + __longlong_as_double(assumed)));
    } while (assumed != old);
    return __longlong_as_double(old);
}
#endif

#endif // CUDA_COMMON_H
