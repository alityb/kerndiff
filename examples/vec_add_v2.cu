// DEGRADED VERSION — committed to test git mode
__global__ void vec_add(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ c,
                        int n) {
    int base = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (base + 0 < n) c[base + 0] = a[base + 0] + b[base + 0];
    if (base + 1 < n) c[base + 1] = a[base + 1] + b[base + 1];
    if (base + 2 < n) c[base + 2] = a[base + 2] + b[base + 2];
    if (base + 3 < n) c[base + 3] = a[base + 3] + b[base + 3];
}
