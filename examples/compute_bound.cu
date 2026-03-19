// Compute-bound kernel: each thread runs a chain of FMAs on register values.
// No memory traffic after the initial load — all work is in registers.
__global__ void compute_bound(const float* __restrict__ a, float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = a[i];
    // 256 dependent FMAs — each iteration depends on the previous (no ILP exploit)
    for (int j = 0; j < 256; j++) {
        v = v * 1.0001f + 0.0001f;
    }
    c[i] = v;
}
