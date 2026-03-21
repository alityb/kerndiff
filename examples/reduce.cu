#include <cuda_runtime.h>

// v1: each thread atomically adds to a global accumulator
// Massive serialization: atomicAdd on a single address
// from thousands of threads is the worst case for atomics.
__global__ void reduce(const float* __restrict__ in,
                       float* __restrict__ out,
                       float* __restrict__ tmp,
                       int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) atomicAdd(out, in[idx]);
}

