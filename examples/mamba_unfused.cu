// mamba_unfused.cu — Simulates 3-step unfused Mamba SSD pipeline
// Each step reads from and writes to global memory (3 HBM round-trips)
// Step 1: h[i] = alpha * a[i] + b[i]        → write h to c
// Step 2: y[i] = c[i] * b[i]                → write y to c (read back from HBM)
// Step 3: o[i] = c[i] + b[i % BLOCK_SIZE]   → write o to c (read back from HBM)

__global__ void ssd_pipeline(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        // Step 1: decay — write intermediate to c (HBM)
        float alpha = 0.99f;
        c[idx] = alpha * a[idx] + b[idx];
    }
    __syncthreads();
    __threadfence();

    if (idx < n) {
        // Step 2: project — read h back from c (HBM round-trip), write y to c
        float h = c[idx];
        c[idx] = h * b[idx];
    }
    __syncthreads();
    __threadfence();

    if (idx < n) {
        // Step 3: output — read y back from c (HBM round-trip), write final to c
        float y = c[idx];
        c[idx] = y + b[idx % 128];
    }
}
