```text
  gpu: NVIDIA H100 SXM5 80GB (mock)
warning: mock mode -- no GPU required.
  compiling...                            ok  0.0s
  warming up (32 iters)...                ok
  profiling v1 (min of 20)...             ok  247us
  profiling v2 (min of 20)...             ok  189us
  extracting ptx...                       ok
  v2 is 1.31x faster  (247.3us -> 189.1us)  [v1: 247-256us ±1%  v2: 189-196us ±1%]
  metric                              v1              v2       delta
  ------------------------------------------------------------------
  latency                    247.3us ±1%     189.1us ±1%      -23.5%  ++
  l1_bank_conflicts                 124K            297K       +173K  --
  shared_mem                        16KB            32KB       +16KB  ++
  l2_hit_rate                      41.2%           67.4%     +26.2pp  ++
  warp_stall_mio                    18.2             7.1      -61.0%  ++
  sm_throughput                    61.3%           79.4%     +18.1pp  ++
  memory_throughput                72.1%           89.3%     +17.2pp  ++
  dram_bw                          412.3           509.1      +23.5%  ++
  ptx_instructions                312847          247300      -65547  ++
  sm_occupancy                     62.4%           51.2%     -11.2pp  --
  register_count                      64              72          +8  -
  l1_hit_rate                      38.1%           41.0%      +2.9pp  +
  warp_divergence                   2.1%            2.0%      -0.1pp  +
  global_load_eff                  79.3%           82.1%      +2.8pp  +
  warp_stall_lmem                   3.1%            3.0%      -0.1pp  +
  ------------------------------------------------------------------
  roofline [compute]               12%bw           15%bw  21% headroom
  ptx diff
  ----------------------------------------------
  instruction             v1      v2       delta
  ld.shared               12      28      +133.3%
  st.shared               24      48      +100.0%
  ld.global               48      31       -35.4%
  bar.sync                 3       4       +33.3%
```

## install
pip install -e .

## usage
kerndiff v1.cu v2.cu --fn kernel_name     # explicit diff
kerndiff kernel.cu --fn kernel_name       # diff against last git commit

## options
--runs N       timing runs, take min (default 20)
--warmup N     warmup iters before timing (default 32)
--format       term | json
--output FILE  write output to file
--no-color     disable ANSI
--gpu N        GPU device index (default 0)
--arch X       nvcc SM arch (auto-detected from GPU name if not set)
--mock         fixture data, no GPU needed
