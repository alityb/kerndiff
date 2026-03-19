#include <cuda_runtime.h>
#include <math.h>

// v1: two-pass softmax
// Pass 1: find row max (reads input once)
// Pass 2: exp(x - max) and sum, then normalize (reads twice)
// Total: 3 reads + 1 write of the N-element row
#define SOFTMAX_N 4096
#define SOFTMAX_ROWS 1024

__global__ void softmax(const float* __restrict__ in,
                        float* __restrict__ out,
                        float* __restrict__ tmp,
                        int n) {
    int row = blockIdx.x;
    int tid = threadIdx.x;
    if (row >= SOFTMAX_ROWS) return;

    const float* x = in + row * SOFTMAX_N;
    float* y = out + row * SOFTMAX_N;
    float* t = tmp + row * SOFTMAX_N;

    __shared__ float smax;
    float local_max = -1e38f;
    for (int i = tid; i < SOFTMAX_N; i += blockDim.x) {
        local_max = fmaxf(local_max, x[i]);
    }

    __shared__ float smem[256];
    smem[tid] = local_max;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] = fmaxf(smem[tid], smem[tid + s]);
        __syncthreads();
    }
    if (tid == 0) smax = smem[0];
    __syncthreads();

    float m = smax;
    float local_sum = 0.0f;
    for (int i = tid; i < SOFTMAX_N; i += blockDim.x) {
        float e = expf(x[i] - m);
        t[i] = e;
        local_sum += e;
    }
    smem[tid] = local_sum;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) smem[tid] += smem[tid + s];
        __syncthreads();
    }
    float inv_sum = 1.0f / smem[0];
    for (int i = tid; i < SOFTMAX_N; i += blockDim.x) {
        y[i] = t[i] * inv_sum;
    }
}

