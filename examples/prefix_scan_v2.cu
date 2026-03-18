// v2: Blelloch work-efficient scan using shared memory
// O(N) work per block, shared memory for intra-block — less DRAM traffic
#ifndef BLOCK_SIZE
#define BLOCK_SIZE 128
#endif

__global__ void prefix_scan(float* __restrict__ a,
                            float* __restrict__ b,
                            float* __restrict__ c,
                            int n) {
    __shared__ float sdata[BLOCK_SIZE];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;

    sdata[tid] = (idx < n) ? a[idx] : 0.0f;
    __syncthreads();

    // Up-sweep (reduce)
    for (int stride = 1; stride < BLOCK_SIZE; stride <<= 1) {
        int ai = (tid + 1) * (stride << 1) - 1;
        if (ai < BLOCK_SIZE)
            sdata[ai] += sdata[ai - stride];
        __syncthreads();
    }

    if (tid == 0)
        sdata[BLOCK_SIZE - 1] = 0.0f;
    __syncthreads();

    // Down-sweep
    for (int stride = BLOCK_SIZE >> 1; stride > 0; stride >>= 1) {
        int ai = (tid + 1) * (stride << 1) - 1;
        if (ai < BLOCK_SIZE) {
            float tmp = sdata[ai - stride];
            sdata[ai - stride] = sdata[ai];
            sdata[ai] += tmp;
        }
        __syncthreads();
    }

    if (idx < n)
        c[idx] = sdata[tid];
}
