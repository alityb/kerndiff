// v1: Hillis-Steele parallel prefix scan (inclusive), global memory only
// O(N log N) work per block, all reads/writes go through global memory
__global__ void prefix_scan(float* __restrict__ a,
                            float* __restrict__ b,
                            float* __restrict__ c,
                            int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx];
    __syncthreads();

    for (int stride = 1; stride < (int)blockDim.x; stride <<= 1) {
        float val = 0.0f;
        if (idx < n && (int)threadIdx.x >= stride)
            val = c[idx - stride];
        __syncthreads();
        if (idx < n && (int)threadIdx.x >= stride)
            c[idx] += val;
        __syncthreads();
    }
}
