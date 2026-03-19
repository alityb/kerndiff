#include <cuda_runtime.h>

// v2: coalesced (row-major) access
#define MATRIX_N 1024

__global__ void copy_strided(const float* __restrict__ in,
                             float* __restrict__ out,
                             float* __restrict__ tmp,
                             int n) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y;
    if (col >= MATRIX_N || row >= MATRIX_N) return;
    out[row * MATRIX_N + col] = in[row * MATRIX_N + col];
}

