#include <cuda_runtime.h>

// v2: warp-level reduce with __shfl_down_sync,
// then shared memory tree across warps,
// then one atomicAdd per block.
__global__ void reduce(const float* __restrict__ in,
                       float* __restrict__ out,
                       float* __restrict__ tmp,
                       int n) {
    __shared__ float sdata[32];
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    float val = (idx < n) ? in[idx] : 0.0f;

    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }

    int lane = threadIdx.x & 31;
    int warp = threadIdx.x >> 5;
    if (lane == 0) sdata[warp] = val;
    __syncthreads();

    val = (threadIdx.x < (blockDim.x >> 5)) ? sdata[lane] : 0.0f;
    if (warp == 0) {
        for (int offset = 16; offset > 0; offset >>= 1) {
            val += __shfl_down_sync(0xffffffff, val, offset);
        }
        if (lane == 0) atomicAdd(out, val);
    }
}

