// Memory-bound kernel: streams N floats from A to C.
// At 4MB with 1 float per thread, nearly all time is spent on HBM reads.
__global__ void mem_bound(const float* __restrict__ a, float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i];
}
