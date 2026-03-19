#include <cuda_runtime.h>
#include <math.h>

// v1: two-pass layer norm
#define LN_N 2048
#define LN_ROWS 512

__global__ void layernorm(const float* __restrict__ in,
                          float* __restrict__ out,
                          float* __restrict__ tmp,
                          int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= LN_ROWS) return;

    const float* x = in + row * LN_N;
    float* y = out + row * LN_N;

    __shared__ float smem[256];
    float sum = 0.0f;
    for (int i = tid; i < LN_N; i += blockDim.x) sum += x[i];
    smem[tid] = sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float mean = smem[0] / LN_N;

    float sq = 0.0f;
    for (int i = tid; i < LN_N; i += blockDim.x) {
        float d = x[i] - mean;
        sq += d * d;
    }
    smem[tid] = sq;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_std = rsqrtf(smem[0] / LN_N + 1e-5f);

    for (int i = tid; i < LN_N; i += blockDim.x) {
        y[i] = (x[i] - mean) * inv_std;
    }
}

