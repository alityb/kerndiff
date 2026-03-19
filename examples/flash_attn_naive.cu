#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define SEQ_LEN 512
#define HEAD_DIM 64

static __device__ float g_scores[SEQ_LEN * SEQ_LEN];

extern "C" __global__ void flash_attn(half* a, half* b, half* c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const float inv_sqrt_d = 1.0f / sqrtf((float)HEAD_DIM);

    if (n < SEQ_LEN * HEAD_DIM || i >= SEQ_LEN) {
        return;
    }

    // Phase 1: score matrix S = Q @ K^T / sqrt(d)
    for (int j = 0; j < SEQ_LEN; ++j) {
        float acc = 0.0f;
        for (int d = 0; d < HEAD_DIM; ++d) {
            float q = __half2float(a[i * HEAD_DIM + d]);
            float k = __half2float(b[j * HEAD_DIM + d]);
            acc += q * k;
        }
        g_scores[i * SEQ_LEN + j] = acc * inv_sqrt_d;
    }

    // Phase 2: row-wise softmax over S
    float row_max = -INFINITY;
    for (int j = 0; j < SEQ_LEN; ++j) {
        float v = g_scores[i * SEQ_LEN + j];
        row_max = v > row_max ? v : row_max;
    }

    float row_sum = 0.0f;
    for (int j = 0; j < SEQ_LEN; ++j) {
        float e = expf(g_scores[i * SEQ_LEN + j] - row_max);
        g_scores[i * SEQ_LEN + j] = e;
        row_sum += e;
    }

    float inv_row_sum = row_sum > 0.0f ? 1.0f / row_sum : 0.0f;
    for (int j = 0; j < SEQ_LEN; ++j) {
        g_scores[i * SEQ_LEN + j] *= inv_row_sum;
    }

    // Phase 3: O = softmax(S) @ V, with V reusing Q buffer (a)
    for (int d = 0; d < HEAD_DIM; ++d) {
        float out = 0.0f;
        for (int j = 0; j < SEQ_LEN; ++j) {
            float w = g_scores[i * SEQ_LEN + j];
            float v = __half2float(a[j * HEAD_DIM + d]);
            out += w * v;
        }
        c[i * HEAD_DIM + d] = __float2half(out);
    }
}

#ifdef FLASH_ATTN_STANDALONE_MAIN
int main() { return 0; }
#endif
