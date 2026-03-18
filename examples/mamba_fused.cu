// mamba_fused.cu — Single fused Mamba SSD pipeline
// All 3 steps in registers — only 1 HBM round-trip instead of 3
// h = alpha * a[i] + b[i]      (decay, in register)
// y = h * b[i]                 (project, in register)
// c[i] = y + b[i % BLOCK_SIZE] (output, single write)

__global__ void ssd_pipeline(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float alpha = 0.99f;
        float h = alpha * a[idx] + b[idx];   // decay — stays in register
        float y = h * b[idx];                 // project — stays in register
        c[idx] = y + b[idx % 128];            // output — single write to HBM
    }
}
