#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <math.h>

#define SEQ_LEN 512
#define HEAD_DIM 64
#define TILE_SIZE 32

extern "C" __global__ void flash_attn(half* a, half* b, half* c, int n) {
    __shared__ float smem_q[TILE_SIZE][HEAD_DIM];
    __shared__ float smem_k[TILE_SIZE][HEAD_DIM];
    __shared__ float smem_v[TILE_SIZE][HEAD_DIM];
    __shared__ float smem_scores[TILE_SIZE][TILE_SIZE];

    const int local_row = threadIdx.x;
    const int q_tile = blockIdx.x;
    const int q_row = q_tile * TILE_SIZE + local_row;
    const float inv_sqrt_d = 1.0f / sqrtf((float)HEAD_DIM);

    if (n < SEQ_LEN * HEAD_DIM || local_row >= TILE_SIZE || q_row >= SEQ_LEN) {
        return;
    }

    // Q tile stays resident for the whole K/V sweep.
    for (int d = 0; d < HEAD_DIM; ++d) {
        smem_q[local_row][d] = __half2float(a[q_row * HEAD_DIM + d]);
    }
    __syncthreads();

    float m_i = -INFINITY;
    float l_i = 0.0f;
    float o_i[HEAD_DIM];
#pragma unroll
    for (int d = 0; d < HEAD_DIM; ++d) {
        o_i[d] = 0.0f;
    }

    for (int kv_tile = 0; kv_tile < SEQ_LEN; kv_tile += TILE_SIZE) {
        const int kv_row = kv_tile + local_row;

        for (int d = 0; d < HEAD_DIM; ++d) {
            if (kv_row < SEQ_LEN) {
                smem_k[local_row][d] = __half2float(b[kv_row * HEAD_DIM + d]);
                smem_v[local_row][d] = __half2float(a[kv_row * HEAD_DIM + d]);  // V reuses Q buffer.
            } else {
                smem_k[local_row][d] = 0.0f;
                smem_v[local_row][d] = 0.0f;
            }
        }
        __syncthreads();

        float row_max = -INFINITY;
        for (int j = 0; j < TILE_SIZE; ++j) {
            float acc = 0.0f;
            for (int d = 0; d < HEAD_DIM; ++d) {
                acc += smem_q[local_row][d] * smem_k[j][d];
            }
            float score = acc * inv_sqrt_d;
            smem_scores[local_row][j] = score;
            row_max = score > row_max ? score : row_max;
        }

        float tile_sum = 0.0f;
        for (int j = 0; j < TILE_SIZE; ++j) {
            float e = expf(smem_scores[local_row][j] - row_max);
            smem_scores[local_row][j] = e;
            tile_sum += e;
        }

        float m_new = row_max > m_i ? row_max : m_i;
        float alpha = (m_i == -INFINITY) ? 0.0f : expf(m_i - m_new);
        float beta = expf(row_max - m_new);
        float l_new = alpha * l_i + beta * tile_sum;

        float scale_old = (l_new > 0.0f) ? (alpha * l_i / l_new) : 0.0f;
        float scale_new = (l_new > 0.0f) ? (beta / l_new) : 0.0f;

        for (int d = 0; d < HEAD_DIM; ++d) {
            float tile_out = 0.0f;
            for (int j = 0; j < TILE_SIZE; ++j) {
                tile_out += smem_scores[local_row][j] * smem_v[j][d];
            }
            o_i[d] = o_i[d] * scale_old + scale_new * tile_out;
        }

        m_i = m_new;
        l_i = l_new;
        __syncthreads();
    }

    for (int d = 0; d < HEAD_DIM; ++d) {
        c[q_row * HEAD_DIM + d] = __float2half(o_i[d]);
    }
}

#ifdef FLASH_ATTN_STANDALONE_MAIN
int main() { return 0; }
#endif
