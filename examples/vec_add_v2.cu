// v2: float4 vectorized loads — 4x fewer memory transactions
__global__ void vec_add(const float* __restrict__ a,
                        const float* __restrict__ b,
                        float* __restrict__ c,
                        int n) {
    int idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 3 < n) {
        float4 a4 = *reinterpret_cast<const float4*>(a + idx);
        float4 b4 = *reinterpret_cast<const float4*>(b + idx);
        float4 c4 = {a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w};
        *reinterpret_cast<float4*>(c + idx) = c4;
    }
}
