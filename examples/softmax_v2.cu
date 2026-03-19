#include <cuda_runtime.h>
#include <math.h>

// v2: online softmax — single pass, no intermediate storage
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

    float m = -1e38f;
    float l = 0.0f;
    for (int i = tid; i < SOFTMAX_N; i += blockDim.x) {
        float xi = x[i];
        float m_new = fmaxf(m, xi);
        l = l * expf(m - m_new) + expf(xi - m_new);
        m = m_new;
    }

    __shared__ float sm[256];
    __shared__ float sl[256];
    sm[tid] = m;
    sl[tid] = l;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            float m2 = fmaxf(sm[tid], sm[tid + s]);
            sl[tid] = sl[tid] * expf(sm[tid] - m2) + sl[tid + s] * expf(sm[tid + s] - m2);
            sm[tid] = m2;
        }
        __syncthreads();
    }
    m = sm[0];
    l = sl[0];

    for (int i = tid; i < SOFTMAX_N; i += blockDim.x) {
        y[i] = expf(x[i] - m) / l;
    }
}

