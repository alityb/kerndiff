// cuda_inclusive_scan.cu — Hillis-Steele inclusive prefix scan using shared memory
// O(N log N) work per block, but all reads/writes go through fast shared memory.
// Computes an inclusive scan: output[i] = sum(input[0..i]).
// Matches the semantics of tl.cumsum / tl.associative_scan in Triton.
//
// Valid cross-language comparison:
//   kerndiff examples/cuda_inclusive_scan.cu examples/triton_scan_v2.py --fn prefix_scan
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

__global__ void prefix_scan(float* __restrict__ a,
                             float* __restrict__ b,
                             float* __restrict__ c,
                             int n) {
    __shared__ float smem[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    smem[tid] = (idx < n) ? a[idx] : 0.0f;
    __syncthreads();

    // Hillis-Steele inclusive scan in shared memory
    for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        float val = (tid >= stride) ? smem[tid - stride] : 0.0f;
        __syncthreads();
        smem[tid] += val;
        __syncthreads();
    }

    if (idx < n) c[idx] = smem[tid];
}
