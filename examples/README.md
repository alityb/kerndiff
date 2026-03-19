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

## reduce

Parallel sum reduction — global atomic (v1) vs warp shuffle tree (v2).
- Key metric: stall_mio, l1_bank_conflicts
- Expected: v2 significantly faster on large N due to elimination of atomic contention

## softmax

Row-wise softmax — two-pass (v1) vs online single-pass (v2).
- Key metric: dram_bw, register_count
- Expected: v2 ~1.5x faster; one fewer memory pass

## coalesced / strided access

Matrix copy — column-major strided reads (v1) vs row-major coalesced reads (v2).
- Key metric: global_load_eff (should be the dominant change)
- Expected: v2 shows near-peak global_load_eff; large latency gap

## layernorm

Layer normalization — two-pass mean+variance (v1) vs Welford online (v2).
- Key metric: dram_bw, stall_sync, register_count
- Expected: v2 ~1.3-1.5x faster; one fewer read of input
