# examples

## vec_add

Two versions of vector addition that demonstrate float4 vectorization.

v1: scalar float loads (baseline)
v2: float4 vectorized loads

Usage:
  kerndiff examples/vec_add_v1.cu examples/vec_add_v2.cu --fn vec_add

Expected: v2 shows higher dram_bw, better global_load_eff, similar latency
or faster depending on memory bandwidth saturation.

### --dtype

For kernels using non-float types:
  kerndiff v1.cu v2.cu --fn my_kernel --dtype half
  kerndiff v1.cu v2.cu --fn my_kernel --dtype int
