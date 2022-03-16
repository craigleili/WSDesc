#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <vector>
#include <iostream>

#ifndef MAX_THREADS
#define MAX_THREADS 512
#endif

// https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#atomic-functions
__device__ __forceinline__ float atomicMul(float* address, float val)
{
    int* address_as_int = (int*)address;
    int old = *address_as_int, assumed;

    do {
        assumed = old;
        old = atomicCAS(address_as_int, assumed, __float_as_int(val * __int_as_float(assumed)));
    // Note: uses integer comparison to avoid hang in case of NaN (since NaN != NaN)
    } while (assumed != old);

    return __int_as_float(old);
}

// https://stackoverflow.com/a/14038590
#define GPU_ERROR_CHECK(ans) {gpu_assert((ans), __FILE__, __LINE__);}
inline void gpu_assert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr,"\nGPU Assert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

inline int64_t gpu_blocks(int64_t total_threads, int64_t threads_per_block) {
    return (total_threads + threads_per_block - 1) / threads_per_block;
}
