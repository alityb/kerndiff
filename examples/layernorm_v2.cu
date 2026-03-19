#include <cuda_runtime.h>
#include <math.h>

// v2: Welford online mean+variance in one pass
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

    float mean = 0.0f;
    float M2 = 0.0f;
    int count = 0;
    for (int i = tid; i < LN_N; i += blockDim.x) {
        count++;
        float delta = x[i] - mean;
        mean += delta / count;
        float delta2 = x[i] - mean;
        M2 += delta * delta2;
    }

    __shared__ float s_mean[256];
    __shared__ float s_M2[256];
    __shared__ float s_count[256];
    s_mean[tid] = mean;
    s_M2[tid] = M2;
    s_count[tid] = (float)count;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float na = s_count[tid];
            float nb = s_count[tid + s];
            float delta = s_mean[tid + s] - s_mean[tid];
            float n_ab = na + nb;
            s_mean[tid] = (na * s_mean[tid] + nb * s_mean[tid + s]) / n_ab;
            s_M2[tid] += s_M2[tid + s] + delta * delta * na * nb / n_ab;
            s_count[tid] = n_ab;
        }
        __syncthreads();
    }
    mean = s_mean[0];
    float inv_std = rsqrtf(s_M2[0] / LN_N + 1e-5f);

    for (int i = tid; i < LN_N; i += blockDim.x) {
        y[i] = (x[i] - mean) * inv_std;
    }
}

