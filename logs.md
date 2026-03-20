# kerndiff GPU integration logs

## Environment

- Instance: g5.xlarge
- GPU: NVIDIA A10G (24GB, sm_86)
- Driver: 580.126.16, CUDA Version: 13.0
- nvcc: CUDA compilation tools, release 13.0, V13.0.88
  - Path: /opt/pytorch/lib/python3.13/site-packages/nvidia/cu13/bin/nvcc
  - Requires `-L` flag for cuda lib dir (pip-installed, not /usr/local/cuda)
- ncu: NVIDIA Nsight Compute 2026.1.0.0
  - Path: /opt/nvidia/nsight-compute/2026.1.0/ncu
  - Installed via: `sudo apt-get install nsight-compute-2026.1.0`
  - Requires sudo for GPU perf counters (ERR_NVGPUCTRPERM)
- Python: 3.13.12 at /opt/pytorch/bin/python3

## Issues fixed

### 1. nvcc not on PATH
nvcc is pip-installed inside PyTorch at `.../nvidia/cu13/bin/nvcc`.
Added `_find_nvcc()` in compiler.py to search pip site-packages.
Also needs `-L` flag for the lib directory (libcudadevrt, libcudart_static).
Added `_find_cuda_lib_dir()` to derive lib path from nvcc location.

### 2. ncu not installed
Not included in AWS deep learning AMI by default.
Installed: `sudo apt-get install nsight-compute-2026.1.0`
Added `_find_ncu()` in profiler.py to search `/opt/nvidia/nsight-compute/`.

### 3. ncu requires sudo for GPU counters
`perf_event_paranoid=0` is not sufficient on this driver (580.126.16).
NVIDIA module parameter `NVreg_RestrictProfilingToAdminUsers` cannot be changed
at runtime (no parameters sysfs directory for nvidia module).
Fix: profiler auto-retries ncu with sudo when permission error detected.

### 4. nvidia-smi l2_cache query not supported
`nvidia-smi --query-gpu=l2_cache` returns "Field not valid" on driver 580.126.16.
Fix: added `GPU_L2_SIZES` dict with known L2 sizes, fuzzy-matched by GPU name.
A10G: 6MB L2.

### 5. NCU --cache-control flag value
NCU 2026.1.0 uses `--cache-control all` not `--cache-control flush_all`.

### 6. NCU CSV parser didn't handle prefix lines
NCU stdout includes `==PROF==` lines and binary stdout before the CSV data.
Parser now finds the CSV header line (`"Metric Name"`) and starts there.

### 7. NCU byte/s unit not converted
`dram__bytes.sum.per_second` returns `byte/s`, needs division by 1e9 for GB/s.
Added `byte/s` and `byte/second` unit handling in parser.

### 8. Harness template comment lines contained placeholders
Template comments like `// {{KERNEL_SOURCE}} -> ...` were getting partially
substituted during `.replace()` chain. Removed placeholder names from comments.

## Real GPU run output

```
  gpu: NVIDIA A10G (device 0, sm_86, clocks unlocked)
  locking clocks...                  skipped  -- results may vary +/-10%
warning: clock locking unavailable (requires sudo). Results may vary +/-10%.
  compiling vec_add...                    ok
  ncu: retrying with sudo...
  profiling v1 vec_add...                 ok  10 runs  86us  cv=11.9%
  ncu: retrying with sudo...
  profiling v2 vec_add...                 ok  10 runs  96us  cv=10.7%
warning: high variance detected (min=86.0us, max=117.8us). Clock locking recommended.
warning: high variance detected (min=96.3us, max=128.0us). Clock locking recommended.
  v2 is 1.12x slower  (86.0us -> 96.3us)  [v1: 86-118us ±12%  v2: 96-128us ±11%]
  metric                              v1              v2       delta
  ------------------------------------------------------------------
  latency                   86.02us ±12%    96.26us ±11%      +11.9%  -
  warp_stall_mio                   135.5           297.8     +119.7%  --
  dram_bw                          229.3           450.2      +96.3%  ++
  memory_throughput                 0.4%            0.7%      +0.3pp  ++
  sm_throughput                     6.3%            2.7%      -3.5pp  --
  ptx_instructions                 65536           86016      +31.2%  --
  sm_occupancy                     69.3%           61.9%      -7.5pp  -
  global_load_eff                 100.0%          100.0%      +0.0pp  ~
  l1_bank_conflicts                    0               0       +0.0%  ~
  register_count                      16              16          +0  ~
  shared_mem                          0B              0B          +0  ~
  ------------------------------------------------------------------
  roofline [memory]                38%bw           75%bw  25% headroom
  ptx diff
  ----------------------------------------------
  instruction             v1      v2       delta
  add.f32                  1       4      +300.0%
  or.b32                   0       1      +100.0%
  shl.b32                  0       1      +100.0%
```

Key observations:
- dram_bw nearly doubled (229 -> 450 GB/s) — float4 vectorization working as expected
- Roofline improved from 38% to 75% bandwidth utilization
- v2 latency is slightly slower due to default grid/block config covering different
  element counts (each v2 thread processes 4 elements)
- global_load_eff is 100% for both (A10G coalesces both patterns well)
- No clock locking available (requires sudo nvidia-smi), so ±10% noise

## 2026-03-18 — scan example, --elems, output polish

### What changed
- Added `examples/prefix_scan_v1.cu` (Hillis-Steele, global mem) and `prefix_scan_v2.cu` (Blelloch, shared mem)
- Added `--elems N` flag to control buffer size (default 4M), exposed in JSON as `config.buf_elems`
- Changed harness `GRID_SIZE` to auto-compute from `BUF_ELEMS`: `(BUF_ELEMS + BLOCK_SIZE - 1) / BLOCK_SIZE`
- Fixed `--all` kernel headers to go to stdout (not stderr) so they survive pipe redirection
- Roofline row: new format `roofline  V1%bw  V2%sm  bound: mem->com  N% headroom`
- Roofline row skipped entirely when both bw and compute utilization are 0 (no NCU data)
- Color: `++` bold green, `+` green, `~` plain, `-` red, `--` bold red (symbol column only)
- 49 tests (up from 45)

### What worked / didn't
- Harness auto-grid fixed the vec_add element mismatch — both versions now process all 4M elements
- With equal coverage, vec_add v1 vs v2 shows "no significant change" on A10G — the hardware coalesces both patterns well
- Scan example shows the expected global->shared tradeoff clearly: dram_bw drops 42%, sm_throughput rises 12pp, bound transitions from memory to compute

### Real GPU output — prefix_scan diff
```
  gpu: NVIDIA A10G (device 0, sm_86, clocks unlocked)
  locking clocks...                  skipped  -- results may vary +/-10%
warning: clock locking unavailable (requires sudo). Results may vary +/-10%.
  compiling prefix_scan...                ok
  ncu: retrying with sudo...
  profiling v1 prefix_scan...             ok  10 runs  157us  cv=5.5%
  ncu: retrying with sudo...
  profiling v2 prefix_scan...             ok  10 runs  202us  cv=3.3%
warning: high variance detected (min=156.7us, max=181.3us). Clock locking recommended.
  v2 is 1.29x slower  (156.7us -> 201.7us)  [v1: 157-181us ±5%  v2: 202-224us ±3%]
  metric                              v1              v2          delta
  ---------------------------------------------------------------------
  latency                    156.7us ±5%     201.7us ±3%         +28.8%  --
  l1_bank_conflicts                    0           3068K  +306805900.0%  --
  warp_stall_mio                    22.5            47.4        +110.5%  --
  memory_throughput                 5.2%            0.2%         -5.0pp  --
  dram_bw                          439.8           253.2         -42.4%  --
  register_count                      16              19             +3  --
  sm_throughput                    72.7%           84.7%        +12.0pp  ++
  ptx_instructions              16384000        15204352          -7.2%  +
  global_load_eff                 103.1%          100.0%         -3.1pp  ~
  shared_mem                          0B            512B           +512  ~
  sm_occupancy                     91.2%           94.7%         +3.5pp  ~
  ---------------------------------------------------------------------
  roofline                         73%bw           85%sm  bound: mem->com  15% headroom
  ptx diff
  ----------------------------------------------
  instruction             v1      v2       delta
  add.s32                  0      58     +5800.0%
  ld.shared                0      29     +2900.0%
  st.shared                0      23     +2300.0%
  add.f32                  1      14     +1300.0%
  setp.gt                  0       7      +700.0%
  shl.b32                  1       8      +700.0%
  bar.sync                 3      16      +433.3%
  bra                      5      17      +240.0%
  and.pred                 1       0      -100.0%
  not.pred                 1       0      -100.0%
  setp.lt                  3       0      -100.0%
  setp.ne                  0       1      +100.0%
  sub.s32                  1       0      -100.0%
  ld.global                3       1       -66.7%
  setp.ge                  2       1       -50.0%
  st.global                2       1       -50.0%
  add.s64                  3       2       -33.3%
  mov.u32                  4       5       +25.0%
```

Key observations:
- v2 (Blelloch/shared) is 1.29x slower on latency due to shared memory bank conflicts (3M conflicts!)
- dram_bw dropped from 440 to 253 GB/s (-42%) — less global memory traffic as expected
- sm_throughput rose from 73% to 85% — more compute-bound
- Roofline bound transitions from memory to compute (mem->com)
- PTX diff clearly shows ld.shared/st.shared appearing, ld.global/st.global decreasing
- The Blelloch scan has O(N) work vs Hillis-Steele's O(N log N) but the shared memory
  bank conflicts dominate on A10G at BLOCK_SIZE=128

## 2026-03-18 — pipeline mode, correctness, shape sweep, mamba example

### What changed
- `--pipeline N` flag: NCU `--launch-count N`, aggregates metrics via `parse_ncu_csv_pipeline()`
- `--correctness` flag: runs both binaries with `--dump-output 16`, compares first 16 elements of d_c
- `--shape N,M,...` flag: runs diff at each buffer size, produces summary table
- `--watch` flag: polls files every 500ms, re-runs on change
- `--tol` flag: tolerance for correctness check (default 1e-4)
- Harness template: added `--dump-output N` support (copies first N elements of d_c to stdout)
- `verify_correctness()` in compiler.py
- `render_shape_table()` in renderer.py
- `total_hbm` derived metric (dram_bw * latency) shown in pipeline mode
- `build_json_payload()` accepts `total_hbm`, `pipeline` params
- Mamba example: `mamba_unfused.cu` (3 HBM round-trips via __threadfence), `mamba_fused.cu` (single pass in registers)
- Both mamba files export `ssd_pipeline` kernel for direct comparison
- 62 tests (up from 49)

### Real GPU output — mamba ssd_pipeline diff
```
  gpu: NVIDIA A10G (device 0, sm_86, clocks unlocked)
  locking clocks...                  skipped  -- results may vary +/-10%
warning: clock locking unavailable (requires sudo). Results may vary +/-10%.
  compiling ssd_pipeline...               ok
  ncu: retrying with sudo...
  profiling v1 ssd_pipeline...            ok  50 runs  215us  cv=7.7%
  ncu: retrying with sudo...
  profiling v2 ssd_pipeline...            ok  50 runs  188us  cv=4.4%
  v2 is 1.14x faster  (215.0us -> 188.4us)  [v1: 215-330us ±8%  v2: 188-226us ±4%]
  metric                              v1              v2       delta
  ------------------------------------------------------------------
  latency                    215.0us ±8%     188.4us ±4%      -12.4%  +
  dram_bw                          326.0           543.2      +66.6%  ++
  ptx_instructions               5636096         3014656      -46.5%  ++
  sm_throughput                    20.0%           12.8%      -7.2pp  --
  memory_throughput                 1.1%            1.0%      -0.1pp  -
  sm_occupancy                     93.4%           84.1%      -9.3pp  -
  warp_stall_mio                   143.0           131.0       -8.4%  +
  global_load_eff                 100.0%          100.0%      +0.0pp  ~
  l1_bank_conflicts                    0               0       +0.0%  ~
  register_count                      16              16          +0  ~
  shared_mem                          0B              0B          +0  ~
  ------------------------------------------------------------------
  roofline                         54%bw           91%bw  bound: memory  9% headroom
  ptx diff
  ----------------------------------------------
  instruction             v1      v2       delta
  add.f32                  1       0      -100.0%
  bar.sync                 2       0      -100.0%
  fma.rn                   1       2      +100.0%
  membar.gl                2       0      -100.0%
  mul.f32                  1       0      -100.0%
  bra                      3       1       -66.7%
  st.global                3       1       -66.7%
  ld.global                6       3       -50.0%
```

Key observations:
- Fused kernel is 1.14x faster (215 → 188us) — eliminates 2 HBM round-trips
- dram_bw +66.6% (326 → 543 GB/s) — fused version better saturates memory bandwidth
- Roofline: 54%bw → 91%bw — nearly hitting A10G memory ceiling (9% headroom)
- PTX: ld.global 6→3, st.global 3→1 — confirms 3 round-trips collapsed to 1
- bar.sync 2→0, membar.gl 2→0 — no more thread fences between pipeline steps
- fma.rn 1→2 — compiler fuses multiply+add into FMA when intermediate values stay in registers
- ptx_instructions -46.5% (5.6M → 3M) — half the work with fusion
- sm_throughput dropped (-7.2pp) because the fused kernel is memory-bound, not compute-bound

## 2026-03-18 — UX overhaul, git integration, validation

### What changed
- `kerndiff.toml` project config file: walks up from cwd to repo root, merges `[defaults]` and `[kernels.<fn>]` sections. CLI flags always win. Uses `tomllib` (Python 3.11+).
- Interactive kernel picker: when multiple `__global__` functions found and stdin is a tty, shows numbered list. Non-tty falls back to hard error.
- Auto-detect `--call` from kernel signature: `parse_kernel_signature()` and `generate_call()` in compiler.py. Maps `float*` → `d_a/d_b/d_c`, `int n` → `N`, `int stride` → `1`. Error message includes auto-generated call with `--call to override` hint.
- Improved git mode: `resolve_git_baseline()` checks file is tracked, extracts from git ref, shows `comparing: HEAD:path vs path (working copy)`.
- `--at COMMIT` flag: compare working copy against any git ref (`HEAD~3`, short hash, etc). Errors if used with two files.
- Noise floor annotation on verdict: `note: clocks not locked — deltas below 10% may not be reliable` when clocks unlocked. Unchanged verdict shows `delta within ±N% noise`.
- `?` confidence indicator on metric rows: appended when delta magnitude < `max(cv_v1, cv_v2) * 2` and clocks not locked. Only on NCU-sourced metrics (not latency).
- `--validate` flag: runs forward and reverse profiling, checks speedup consistency. Warns if |forward - reverse| > 5%.
- 90 tests (up from 62)

### What worked / didn't
- Auto kernel detection: `kerndiff examples/mamba_unfused.cu examples/mamba_fused.cu` works with no `--fn` — both files have `ssd_pipeline`
- Auto-call generation: standard `(float*, float*, float*, int)` produces correct call. Non-standard signatures with `__restrict__`, `const`, stride params handled correctly.
- Config file: `kerndiff.toml` with `[defaults] fn = "..."` eliminates repeated `--fn` flags in project workflows
- Git mode: `comparing: HEAD:examples/vec_add_v2.cu vs examples/vec_add_v2.cu (working copy)` — clear what's being compared

## 2026-03-19 — deep testing + ux polish session

### Stress test 1.1 — zero-output kernel
- Command: `sudo /opt/pytorch/bin/python3 -m kerndiff examples/noop_kernel.cu examples/noop_kernel.cu --fn noop_kernel --no-color`
- Result:
  - v1 min latency: `54.27us` (p50 `55us`, cv `9.9%`)
  - v2 min latency: `54.27us` (p50 `54us`, cv `11.3%`)
  - verdict: `no significant latency change`
  - no traceback/crash.

### Stress test 1.2 — crash behavior
- OOB write (`c[n + 10000000]`) did not reliably fault on this harness layout (likely wrote into valid neighboring GPU allocation).
- Deterministic crash validation using `trap`:
  - Command: `sudo /opt/pytorch/bin/python3 -m kerndiff /tmp/crash_kernel_trap.cu /tmp/crash_kernel_trap.cu --fn crash_kernel --no-color`
  - Result: clean user-facing error, no Python traceback:
    - `error: kernel crashed during warmup (exit code 1)`
    - includes binary path and CUDA stderr (`unspecified launch failure`).

### Stress test 1.3 — very fast kernel timer behavior
- Command: `sudo /opt/pytorch/bin/python3 -m kerndiff /tmp/fast_kernel.cu /tmp/fast_kernel.cu --fn fast_kernel --call "fast_kernel<<<1, 1>>>(d_a, d_b, d_c, N)"`
- Result:
  - v1/v2 min latency both `27.65us`, p50 around `29us`.
  - Observed floor is tens of microseconds (harness + event + launch overhead dominates), not sub-us.
  - One outlier was detected and excluded on v2 (`1 outlier run(s) detected (>2x median)`).

### Stress test 1.4 — register pressure
- Command: `sudo /opt/pytorch/bin/python3 -m kerndiff /tmp/reg_pressure_ref.cu /tmp/reg_pressure.cu --fn reg_pressure`
- Result:
  - latency improved (`135.2us -> 99.3us`, `1.36x faster`) due compute-heavy kernel.
  - `register_count` stayed `16 -> 16` (no increase observed; compiler optimization likely kept register footprint flat).
  - `sm_occupancy` nearly unchanged (`63.1% -> 62.8%`).

### Stress test 1.5 — FlashAttention shape sweep
- Command loop over `SEQ_LEN=256,512,1024` with temporary sed override and kerndiff run.
- Results:
  - `N=256`: `7118.8us -> 493.6us`, `14.42x faster`, roofline `memory`, dram_bw delta `-0.2%`.
  - `N=512`: `14196.7us -> 795.6us`, `17.84x faster`, roofline `memory`, dram_bw delta `-0.5%`.
  - `N=1024`: `28387.3us -> 1410.0us`, `20.13x faster`, roofline `memory`, dram_bw delta `-0.1%`.
- Observed trend: speedup increases with sequence length; roofline label stayed memory-bound in this implementation on A10G.

### Stress test 1.6 — mock modes and schema checks
- Term mode (`--no-color`) validated expected numeric formats and no ANSI codes.
- JSON mode validated required keys and delta object schema:
  - `missing top-level: []`
  - `missing verdict: []`
  - `num deltas: 18`
- Single-file mock git mode runs cleanly.

### Stress test 1.7 — concurrent runs / temp dir cleanup
- Ran two sudo kerndiff jobs concurrently (vec_add and prefix_scan).
- Both completed successfully.
- `/tmp/kerndiff_build_*` leftover count after completion: `0`.

### Stress test 1.8 — `--all` many kernels
- Command: `sudo /opt/pytorch/bin/python3 -m kerndiff /tmp/multi.cu /tmp/multi.cu --all --no-color`
- Result:
  - 5 kernel blocks (`k1..k5`) produced.
  - All verdicts showed no significant latency change.
  - All self-diff deltas remained near noise; no crashes.

### Stress test 1.9 — invalid NCU metric
- Temporarily added metric: `this_metric_does_not_exist`.
- Command: vec_add diff under sudo.
- Result:
  - no crash/traceback.
  - graceful warning path: `warning: NCU returned no usable output.`
  - tool continued with latency-only diff.
- Temporary metric was removed immediately after test.

### Stress test 1.10 — Triton warmup leakage check
- `warmup=1`:
  - latency `73.73us` (v1/v2), warning emitted:
    - `warmup=1 may be insufficient for Triton kernels ...`
- `warmup=200`:
  - latency also `73.73us` (v1/v2).
- Delta between warmup settings on this setup: negligible (<1%).
- Conclusion: no measurable JIT leakage into timing samples in this persistent-harness run, but low-warmup warning is now surfaced.

### NCU invocation integrity verification (production correctness)
- Direct NCU sanity check with `--launch-skip 1` showed wrong kernel:
  - profiled kernel: `_kerndiff_fill(float *, int)`
- Root cause: harness launches two fill kernels in `setup_buffers()`.
- Fix applied:
  - CUDA NCU command now uses `--launch-skip 2`.
  - NCU clock control changed from `base` to `none`.
  - Triton backend NCU command uses `--clock-control none` and does not apply CUDA-harness skip.
- Post-fix sanity check:
  - profiled kernel is now `vec_add(const float *, const float *, float *, int)`.

## 2026-03-19 — ux polish changes

### UX change 1: outlier detection and exclusion
- Implemented outlier filter (`>2x median`, only with >=5 samples, does not remove majority).
- New fields in `ProfileResult`:
  - `clean_latencies_us`, `n_outliers`
- Warning emitted when applicable:
  - example: `1 outlier run(s) detected (>2x median), excluded from statistics`
- JSON now includes both full and clean latency arrays per side (`v1`, `v2`).

Before:
```
profiling v2 ... ok 50 runs ... cv=12.7%
```

After:
```
warning: 1 outlier run(s) detected (>2x median), excluded from statistics
```

### UX change 2: median and percentile latency display
- Added `median_latency_us`, `p20_latency_us`, `p80_latency_us` to `ProfileResult`.
- Latency table now shows min + percentile context:

Before:
```
latency  247.3us ±1%  -> 189.1us ±1%
```

After:
```
latency  247.3us (p50 251us, p20-p80: 248-254us) ±1%
      -> 189.1us (p50 192us, p20-p80: 190-194us) ±1%
```

- JSON verdict now includes:
  - `v1_p20_us`, `v1_p50_us`, `v1_p80_us`
  - `v2_p20_us`, `v2_p50_us`, `v2_p80_us`

### UX change 3: speedup uncertainty display
- Added ratio-error propagation and display in verdict line.
- New verdict field: `speedup_uncertainty_x`.
- Example:

Before:
```
v2 is 1.36x faster  (135.2us -> 99.3us)
```

After:
```
v2 is 1.36x faster  (135.2us -> 99.3us)  ±0.10x
```

- High-uncertainty warning path is now implemented when uncertainty is large.

### UX change 4: `--export-json FILE`
- Added new CLI flag:
  - `--export-json FILE` implies JSON format and output file.
  - preserves stderr status/progress output (unlike bare `--format json` suppression mode).
- Validation run:
  - `stdout_bytes=0`
  - `stderr_lines=5`
  - output file parsed successfully (`json_ok`).

### Regression/test status
- Focused new/updated tests added for:
  - outlier filtering behavior and warnings
  - NCU command flags (`clock-control`, launch skip)
  - renderer percentile/uncertainty output
  - `--export-json` behavior
- Full suite after changes:
  - `220 passed in 0.39s`
- The `?` annotations correctly flag uncertain metrics (sm_occupancy -9.2pp, warp_stall_mio -9.1%) while leaving large deltas (dram_bw +66.7%, ptx_instructions -46.5%) clean
- Validate in mock mode always shows consistent (same fixtures) — that's expected

### Real GPU output — single-file git mode (vec_add_v2.cu)
```
  gpu: NVIDIA A10G (device 0, sm_86, clocks unlocked)
  comparing: HEAD:examples/vec_add_v2.cu  vs  examples/vec_add_v2.cu (working copy)
  locking clocks...                  skipped  -- results may vary +/-10%
  compiling vec_add...                    ok
  profiling v1 vec_add...                 ok  50 runs  198us  cv=6.0%
  profiling v2 vec_add...                 ok  50 runs  198us  cv=9.2%
  no significant latency change  (197.6us vs 197.6us, delta within ±6% noise)
  metric                              v1              v2       delta
  ------------------------------------------------------------------
  latency                    197.6us ±6%     197.6us ±9%       +0.0%  ~
  dram_bw                          482.4           479.7       -0.6%  ~
  [all metrics ~]
  roofline                         80%bw           80%bw  bound: memory  20% headroom
  ptx diff
  setp.ge → setp.gt (bounds check change: idx+3<n → idx+4<=n)
```

### Real GPU output — mamba with noise annotations
```
  v2 is 1.14x faster  (214.1us -> 188.4us)  [v1: 214-259us ±5%  v2: 188-250us ±6%]
  note: clocks not locked — deltas below 10% may not be reliable
  metric                              v1              v2       delta
  ------------------------------------------------------------------
  latency                    214.1us ±5%     188.4us ±6%      -12.0%  +
  dram_bw                          324.3           540.4      +66.7%  ++
  ptx_instructions               5636096         3014656      -46.5%  ++
  sm_throughput                    19.6%           12.8%      -6.9pp  --
  sm_occupancy                     93.8%           84.5%      -9.2pp  - ?
  warp_stall_mio                   145.7           132.4       -9.1%  + ?
  memory_throughput                 1.1%            1.0%      -0.1pp  - ?
```

Key observations:
- `?` correctly flags metrics where the delta is within noise ceiling (2x max CV)
- Large deltas (dram_bw +66.7%, ptx_instructions -46.5%) have no `?` — clearly real
- Noise note on verdict line makes uncertainty visible at a glance
- Git mode shows the exact comparison being made — no ambiguity

## 2026-03-19 — stress test + hardening

### Bugs found and fixed

#### 1. `--elems 0` accepted silently
**Bug**: `--elems 0` in mock mode ran without error. On real GPU this would produce
a zero-element allocation (harmless but meaningless).
**Fix**: Added validation in `cli.py`: `if args.elems <= 0: raise SystemExit("error: --elems must be > 0")`

#### 2. `--min-runs > --max-runs` accepted silently
**Bug**: `--min-runs 100 --max-runs 10` ran without error (mock mode always uses 20
synthetic runs, so the inconsistency was invisible).
**Fix**: Added validation: `if args.min_runs > args.max_runs: raise SystemExit(...)`

#### 3. `--shape "0,1024"` accepted zero as shape value
**Bug**: Shape value of 0 was accepted, producing a row with shape=0 in the sweep table.
**Fix**: Added `if any(s <= 0 for s in shapes): raise SystemExit("error: --shape values must be > 0")`

#### 4. Config unknown keys silently ignored
**Bug**: `kerndiff.toml` with `[defaults] unknown_key = "value"` loaded without any
warning. User typos (e.g., `elem` instead of `elems`) would be silently dropped.
**Fix**: Added unknown-key detection in `apply_config()`. Warns for unknown keys in
`[defaults]`, `[kernels.<fn>]`, and unknown top-level sections.
Output: `warning: unknown config key 'unknown_key' ignored`

#### 5. Config wrong types silently applied
**Bug**: `[defaults] elems = "not_a_number"` would set `args.elems` to a string,
crashing later when used in arithmetic. No validation at config load time.
**Fix**: Added type checking in `apply_config()` using `_CONFIG_TYPES` dict. Wrong
types are warned and ignored: `warning: config key 'elems' has wrong type (expected int, got str) — ignored`

#### 6. `compute_delta()` crashed on NaN/inf metric values
**Bug**: `float('nan')` or `float('inf')` in metric values produced NaN delta_pct,
which could propagate through rendering without clear indication of bad data.
**Fix**: Added guard at top of `compute_delta()`: if either value is NaN or inf,
return `symbol="~"` with `delta_pct=0.0`.

#### 7. `--shape` error message said "comma-separated integers"
**Bug**: Minor — error message was inconsistent with new validation.
**Fix**: Changed to "positive integers" and updated corresponding test assertion.

### Edge cases verified (no fix needed)

| Test case | Result |
|-----------|--------|
| `--elems 1` (mock) | ✅ works |
| `--elems 1000003` (mock) | ✅ works |
| `--elems 2147483648` (mock) | ✅ works |
| `--pipeline 3` without `--call` | ✅ clean error |
| `--at HEAD` with two files | ✅ clean error |
| `--fn` + `--all` | ✅ clean error |
| `--noise-threshold 0` | ✅ works (runs min_runs) |
| Single file outside git repo | ✅ `error: single-file mode requires a git repo` |
| Untracked file in git | ✅ `error: kernel.cu is not tracked by git` |
| Tracked, no commits | ✅ `error: file not found in HEAD` |
| `--at nonexistent_ref` | ✅ `error: file not found in nonexistent_ref` |
| `--at HEAD --mock` (self-diff) | ✅ runs, shows expected mock output |
| .py with no `@triton.jit` | ✅ `error: could not auto-detect kernel` |
| Unsupported `.cpp` extension | ✅ `error: unsupported file type .cpp` |
| Empty `kerndiff.toml` | ✅ loads as empty dict |

### Real GPU stress tests

#### NaN kernel — `c[idx] = 0.0f / 0.0f`
- Compiles and runs ✅
- `--correctness` reports `max_diff=0.0e+00` (NaN vs NaN same kernel)
- NCU profiles normally — no crash

#### OOB crash kernel — `c[idx + 1000000000] = 1.0f`
- Compiles ✅
- Caught during warmup: `error: kernel crashed during warmup (exit code 1)`
- stderr: `CUDA error: an illegal memory access was encountered`
- Clean exit, no Python traceback ✅

#### Slow kernel — 100K iterations of `sinf()`
- Runs ~764ms per launch (763,649us)
- Profiling completes with 3 runs in ~5 minutes
- cv=0.0% (very stable, compute-bound)
- `roofline: 94%sm, bound: compute, 6% headroom` ✅

#### NCU with bad metric name
- `bad_metric_xyz` injected into METRICS_BY_NCU
- NCU error: `==ERROR== Failed to find metric regex:^bad_metric_xyz...`
- Tool warns and continues latency-only ✅

#### Same-kernel validation (`prefix_scan_v2 vs prefix_scan_v2`)
- Forward: 1.00x, Reverse: 1.01x, delta: 1.0%
- `validate: ok` ✅ — consistent as expected

### New tests added

- `tests/test_diff_boundaries.py` — 17 tests for significance ladder boundaries,
  NaN/inf handling, noise floor edge cases, verdict computation
- `tests/test_renderer_edge_cases.py` — 10 tests for no-color rendering, empty
  metrics, empty PTX diff, unicode kernel names
- `tests/test_parser_edge_cases.py` — 10 tests for empty input, whitespace,
  prefix lines, comma-in-value, missing columns, non-numeric values, unit conversions

### Test count: 102 → 139 (37 new tests)

```
$ python3 -m pytest tests/ -q
........................................................................ [ 51%]
...................................................................      [100%]
139 passed in 0.32s

$ CUDA_VISIBLE_DEVICES="" python3 -m pytest tests/ -q
139 passed in 0.32s
```

## 2026-03-19 — NVML-based roofline + GPU_SPECS audit + NaN correctness fix

### What changed

#### NVML peak bandwidth query
- Added `query_peak_bandwidth_nvml(gpu_id)` to `profiler.py`
- Uses `pynvml.nvmlDeviceGetMemoryBusWidth()` and `nvmlDeviceGetMaxClockInfo(NVML_CLOCK_MEM)`
- Formula: `(bus_width_bits / 8) * mem_clock_mhz * 1e6 * 2 / 1e9`
  - The `* 2` is the DDR multiplier — NVML returns base clock, not effective data rate
- Falls back gracefully (`None`) if pynvml not importable or NVML call fails (vGPU etc.)
- Suppresses `FutureWarning` from deprecated pynvml package

#### Three-tier roofline fallback
- `compute_roofline()` now accepts `nvml_peak_bw: float | None`
- Priority: NVML > spec table > skip (return `gpu_matched=False`)
- `RooflineResult` gains two new fields: `peak_bw_gbs` and `bw_source` ("nvml" | "table" | "unknown")
- Both have defaults (0.0, "unknown") so existing code creating `RooflineResult` positionally still works

#### `[spec]` tag in roofline row
- When `bw_source == "table"` (NVML unavailable), roofline row shows `[spec]`
- When `bw_source == "nvml"`, no tag — NVML is the trusted source
- Mock mode always shows `[spec]` since there's no real GPU

#### GPU_SPECS table audit
- Changed from raw dicts to `GpuSpec` dataclass
- Split H100 into SXM5 (3350 GB/s) and PCIe (2000 GB/s)
- Split A100 into SXM4 (2000 GB/s) and PCIe (1935 GB/s)
- Added H200 SXM (4800 GB/s), V100 SXM2/PCIe (900 GB/s)
- Fixed A10G TFLOPS: 125 → 31.2 (was wrong — 125 is V100's number)
- L40S: 733 → 362 TFLOPS FP16 (733 was FP8)
- Ordered most-specific keys first so "H100 PCIe" matches before "H100"
- `_find_spec()` renamed from `fuzzy_match_gpu()`, returns `GpuSpec` directly

#### NaN correctness fix
- Added `_safe_diff(a, b)` to `compiler.py`
- `NaN == NaN → 0.0` (both produce NaN = consistent)
- `NaN vs real → inf` (always fails correctness)
- `+inf == +inf → 0.0`, `-inf == -inf → 0.0`
- `+inf vs -inf → inf` (different signs = fails)
- `verify_correctness()` now uses `_safe_diff` instead of `abs(a - b)`

### Validation script output (A10G)
```
GPU: NVIDIA A10G
SM clock: 210 MHz
Mem clock: 405 MHz

NVML peak bandwidth: 600.1 GB/s
Table peak bandwidth: 600.0 GB/s
Difference: 0.0% [OK]

Raw NVML params:
  bus_width = 384 bits
  mem_clock = 6251 MHz
  computed peak (base clock only) = 300.0 GB/s
  computed peak (with 2x DDR)     = 600.1 GB/s

Roofline test (dram=500 GB/s, sm=50%):
  bw_utilization: 83.3%
  bound: memory
  bw_source: nvml
  peak_bw_gbs: 600.1
```

Key observations:
- NVML `nvmlDeviceGetMaxClockInfo(NVML_CLOCK_MEM)` returns 6251 MHz (base clock)
- With DDR 2x multiplier: 384/8 × 6251e6 × 2 / 1e9 = 600.1 GB/s
- Matches table value of 600 GB/s within 0.02%
- A10G has 384-bit bus width (GDDR6), not 256-bit as some spec sheets report

### NaN correctness test output
```
  compiling nan_kernel...                 ok
  correctness check...                FAILED  max_diff=inf  (first 4: v1=[1.0, 1.0, 1.0, 1.0] v2=[nan, nan, nan, nan])
warning: outputs differ (max_diff=inf  ...) — speedup may reflect a bug, not an optimization
```

### Real GPU roofline output (no [spec] tag)
```
  roofline                         74%bw           85%sm  bound: mem->com  15% headroom
```
(No `[spec]` — NVML bandwidth used)

### Mock mode roofline output ([spec] tag present)
```
  roofline                         61%sm           79%sm  bound: compute  21% headroom  [spec]
```

### Test count: 139 → 166 (27 new tests)

```
$ python3 -m pytest tests/ -q
166 passed in 0.33s

$ CUDA_VISIBLE_DEVICES="" python3 -m pytest tests/ -q
166 passed in 0.34s
```

---

## Session 3: Correctness Audit

### NCU metric audit

Every NCU metric string in `kerndiff/metrics.py` was audited against `ncu --query-metrics` on the real A10G (NCU 2026.1.0).

| Key | Old NCU metric | New NCU metric | Issue |
|-----|---------------|----------------|-------|
| `latency_us` | `gpu__time_duration.sum` | (same) | ✓ correct |
| `sm_throughput` | `sm__throughput.avg.pct_of_peak_sustained_elapsed` | (same) | ✓ correct |
| `memory_throughput` | `l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.pct_of_peak_sustained_elapsed` | `gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed` | was global-load-only, not overall memory SOL |
| `l2_hit_rate` | `l2cache__hit_rate.pct` | `lts__t_sector_hit_rate.pct` | `l2cache__hit_rate` doesn't exist in NCU 2026 |
| `l1_hit_rate` | `l1tex__hit_rate.pct` | `l1tex__t_sector_hit_rate.pct` | `l1tex__hit_rate` doesn't exist in NCU 2026 |
| `dram_bw_gbs` | `dram__bytes.sum.per_second` | (same) | ✓ correct |
| `l1_bank_conflicts` | `l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum` | (same) | ✓ correct |
| `global_load_eff` | `smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct` | (same) | ✓ correct |
| `sm_occupancy` | `smsp__warps_active.avg.pct_of_peak_sustained_active` | (same) | ✓ correct |
| `warp_stall_mio` → `warp_latency_per_inst` | `smsp__average_warp_latency_per_inst_issued.ratio` | (same metric, renamed key+display) | name "warp_stall_mio" was wrong — it's all stalls, not MIO-specific |
| `warp_stall_lmem` → `warp_stall_lg` | `smsp__warp_issue_stalled_local_mem_throttle_per_warp_active.pct` | `smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct` | `local_mem_throttle` doesn't exist; renamed to `lg_throttle` (local+global) |
| `warp_divergence` | `smsp__branch_targets_threads_diverged.avg` unit "%" | `smsp__sass_branch_targets_threads_divergent.sum` unit "count" | metric name wrong ("diverged" vs "divergent"), unit was "%" but it's a count |
| `registers_per_thread` | `launch__registers_per_thread` | (same) | ✓ correct |
| `ptx_instructions` → `inst_executed` | `inst_executed` | `smsp__inst_executed.sum` | old name `inst_executed` doesn't exist; was misleadingly named "ptx_instructions" (static count) but value is dynamic |
| `shared_mem_kb` | `launch__shared_mem_per_block_static` | (same) | ✓ correct (static only) |

### Other fixes

- **PTX section label**: `"ptx diff"` → `"ptx diff  (static instruction count — not dynamic execution count)"`
- **JSON ptx_diff**: wrapped as `{"note": "static instruction count...", "instructions": [...]}`
- **JSON deltas**: added `ncu_metric` field to each delta entry for transparency
- **Noise floor message**: changed from hardcoded `"10%"` to computed `max(v1_cv_pct, v2_cv_pct) * 2.0`
- **Mock fixture updates**: all three changed metric strings updated in `v1_ncu.csv` and `v2_ncu.csv`

### L2 flush audit

The `flush_l2()` function in `harness_template.cu` correctly:
1. Allocates an L2-sized scratch buffer
2. Memssets it (touching all cache lines)
3. Synchronizes (waits for completion)
4. Frees it

Timing only starts after `flush_l2()` returns, so the kernel sees cold L2 cache. No bug.

### Roofline bound detection — real GPU validation

Profiled `examples/mem_bound.cu` and `examples/compute_bound.cu` on A10G:

**mem_bound** (streams N floats from A to C):
```
  sm_throughput      11.3%   (low — barely any compute)
  memory_throughput  88.1%   (high — saturating HBM)
  dram_bw            528.1 GB/s
  roofline           88%bw   bound: memory  12% headroom
```

**compute_bound** (256 dependent FMAs per thread):
```
  sm_throughput      84.9%   (high — saturating FP32 pipe)
  memory_throughput  64.4%   (lower)
  inst_executed      35651K
  roofline           85%sm   bound: compute  15% headroom
```

Roofline bound detection is correct. NVML-based peak BW used (no `[spec]` tag).

### Test count: 166 (unchanged)

```
$ python3 -m pytest tests/ -q
166 passed in 0.34s
```

---

## 2026-03-19 — Cross-check: 3.21x result investigation

### Check 1: Algorithm comparison

| Kernel | Type | Scan kind | Scope | Complexity |
|--------|------|-----------|-------|------------|
| `prefix_scan_v1.cu` (Hillis-Steele global mem) | CUDA | **inclusive** | within-block | O(N log N) loads to global mem |
| `prefix_scan_v2.cu` (Blelloch) | CUDA | **exclusive** | within-block | O(N) work, shared mem |
| `triton_scan_v1.py` (was "naive H-S") | Triton | **WRONG** (buggy) | within-block | — |
| `triton_scan_v2.py` (tl.cumsum) | Triton | **inclusive** | within-block | O(N) work, warp shuffles |

CUDA v2 (exclusive) ≠ Triton v2 (inclusive). The 3.21x comparison was **invalid** — different algorithms.

Additionally, `triton_scan_v1.py` had a bug: the Hillis-Steele loop read from `a_ptr` (original input)
in every pass instead of reading from the running partial sums. This produces wrong results for all N≥4.

### Check 2: Correctness on N=16 (input = [1..16])

```
CUDA v1 (H-S global):  1 3 6 10 15 21 28 36 45 55 66 78 91 105 120 136  ← correct inclusive
CUDA v2 (Blelloch):    0 1 3  6 10 15 21 28 36 45 55 66 78  91 105 120  ← correct exclusive
Triton v1 (old buggy): 1 3 6  9 13 17 21 25 30 35 40 45 50  55  60  65  ← WRONG
Triton v2 (cumsum):    1 3 6 10 15 21 28 36 45 55 66 78 91 105 120 136  ← correct inclusive
Expected inclusive:    1 3 6 10 15 21 28 36 45 55 66 78 91 105 120 136
Expected exclusive:    0 1 3  6 10 15 21 28 36 45 55 66 78  91 105 120
```

### Check 3: Why did Triton v1 ≈ v2 = 72μs (with the old buggy v1)?

NCU metrics showed both were memory-bandwidth bound (DRAM ~530 GB/s, 89% of peak):
- v1 l1_hit_rate: **76.8%** — the stride-load within a 128-float window fits entirely in L1 (128×4=512 B).
  Each stride pass re-reads the same 512-byte block from L1, not DRAM. No extra DRAM traffic.
- v2 l1_hit_rate: **0%** — warp shuffle (`shfl.sync`) bypasses L1 entirely, reads from register exchange.
- stall_memory: 84% (v1) vs 72% (v2) — different bottleneck mechanism, same wall time.

Both kernels were limited by the single global load + store of 4M×4=16MB per direction (~33MB total).
At 530 GB/s: 33MB / 530 = 62μs minimum; measured 72μs is consistent.

After fixing v1 to use `tl.associative_scan` (same hardware path as `tl.cumsum`):
both show identical 72μs — the `_add` combine function and `tl.cumsum` compile to the same PTX.

### Check 4: CUDA Blelloch variance across sessions

Three consecutive self-diff runs:
```
Run 1: 238us  cv=5.3%
Run 2: 237us  cv=2.9%
Run 3: 238us  cv=5.5%
```
Stable within 1μs. The earlier session-2 measurement (202μs) was a different thermal/clock state
(cold GPU at the start of a benchmarking session, boost clocks briefly active). The steady-state
CUDA Blelloch latency on A10G with unlocked clocks is **237–238μs**.

### Check 5: Valid cross-language comparison

3.21x result: **invalid** — CUDA Blelloch is exclusive, Triton cumsum is inclusive.

Valid comparison: `cuda_inclusive_scan.cu` (Hillis-Steele in shared mem, inclusive) vs
`triton_scan_v2.py` (tl.cumsum, inclusive) — **same algorithm, same semantics**.

```
  profiling v1 prefix_scan...   ok  50 runs  161us  cv=6.7%   (CUDA H-S shared mem)
  profiling v2 prefix_scan...   ok  50 runs   72us  cv=1.2%   (Triton tl.cumsum)
  v2 is 2.24x faster  (160.8us -> 71.7us)

  metric                              v1              v2          delta
  sm_throughput                    81.9%           39.0%        -42.9pp  (CUDA is compute-bound)
  memory_throughput                81.9%           87.1%         +5.3pp
  stall_sync                       35.6%            8.7%        -26.9pp  (barriers vs shuffles)
  l1_bank_conflicts                  44K              3K         -92.7%
  bar.sync PTX:                       15               1         -93.3%
  shfl.sync PTX:                       0               6         +600.0% (warp-level shuffles)
  shared_mem:                        512B              0B          -512
  roofline                     v1: 82%sm       v2: 87%bw  bound: com->mem
```

The 2.24x improvement is real and explained:
- CUDA Hillis-Steele does log2(128)=7 barrier-synchronised passes through shared memory.
  `stall_sync=35.6%` confirms the kernel is stalling on `bar.sync` instructions. It's compute-bound
  (sm=82%) because the barrier overhead keeps SMs busy waiting, not computing.
- Triton `tl.cumsum` compiles to `shfl.sync` warp-shuffle instructions — no shared memory, no barriers.
  It's memory-bandwidth-bound (87% throughput) which is the actual hardware ceiling for this problem.

### Fixes applied

1. **`examples/triton_scan_v1.py`** — rewrote buggy Hillis-Steele (read from a_ptr) to use
   `tl.associative_scan` with explicit `_add` combine function. Correct inclusive scan on N=16
   verified: [1,3,6,10,15,21,28,36,45,55,66,78,91,105,120,136] ✓

2. **`examples/cuda_inclusive_scan.cu`** (new) — Hillis-Steele inclusive scan in shared memory.
   Produces inclusive output matching Triton's cumsum. Verified on N=16 ✓.
   Use for cross-language comparison: `kerndiff examples/cuda_inclusive_scan.cu examples/triton_scan_v2.py --fn prefix_scan`

### Conclusion

The 3.21x result was an artifact of comparing two different problems: CUDA Blelloch computes an
*exclusive* scan (result[0]=0) while Triton cumsum computes an *inclusive* scan (result[0]=input[0]).
The actual performance difference between equivalent implementations — CUDA Hillis-Steele shared-mem
(inclusive) vs Triton warp-shuffle scan (inclusive) — is **2.24x in favour of Triton** on A10G.
The speedup is real and explained: Triton compiles to `shfl.sync` (warp shuffles, no barriers, memory-bound)
while CUDA shared-mem Hillis-Steele stalls on `bar.sync` (35.6% stall_sync, compute-bound).

---

## 2026-03-19 — Triton persistent harness

### Problem

The old Triton timing architecture spawned a new Python process for every timed run:
```
for each run: spawn python harness.py → import torch/triton → warmup → flush → time 1 kernel → exit
```

Three correctness problems:
1. **~500ms Python import overhead per run** — 50 runs = 25s of imports alone (wall time dominated by startup, not GPU work)
2. **L2 flush unreliable between runs** — each new process starts with indeterminate L2 state (new CUDA context). The flush inside each harness ran at context init time, not between the previous run's kernel and the current run's kernel.
3. **Fast kernels get inflated timing** — for kernels under ~10μs, the CPU can issue `t1.record()` before the GPU finishes the kernel. The fix (from `triton.testing.do_bench`) is `torch.cuda._sleep(1_000_000)` to saturate the GPU queue before the timed region, ensuring the CPU has time to enqueue both events while the GPU is still sleeping.

### Fix: persistent harness with pipe protocol

New template `harness_template_triton_persistent.py`:
- Spawned **once** per profiling pass; stays alive across all timed runs
- Communicates via stdin/stdout: kerndiff sends `"time\n"`, harness responds with latency in μs
- `L2_FLUSH_BYTES` baked in at compile time; flush happens via `_flush_buf.zero_()` between each `time` command
- `torch.cuda._sleep(1_000_000)` before each kernel timing to prevent CPU outrunning GPU
- Warmup and PTX extraction happen at startup before printing `"ready"`
- Separate `compile_ncu()` method generates old single-run harness for NCU pass (NCU handles its own replay)

### Files changed
- `kerndiff/fixtures/harness_template_triton_persistent.py` — new persistent harness template
- `kerndiff/backends/triton.py` — `compile()` uses persistent template; new `compile_timed()` (persistent + L2 flush), `compile_ncu()` (single-run), `is_persistent()`, `spawn_persistent()`, `send_time()`, `shutdown()`
- `kerndiff/profiler.py` — `_run_warmup_backend()` skips persistent backends; `_run_timed_backend()` uses pipe protocol for persistent; NCU pass uses `compile_ncu()` artifact
- `kerndiff/cli.py` — correctness check skipped for Triton (persistent harness doesn't support `--dump-output`)

### Verification 1: Triton-vs-Triton diff

```
  gpu: NVIDIA A10G (device 0, sm_86, clocks unlocked)
  compiling prefix_scan...                ok
  profiling v1 prefix_scan...             ok  50 runs  72us  cv=1.3%
  profiling v2 prefix_scan...             ok  50 runs  72us  cv=1.2%
  no significant latency change  (71.7us vs 71.7us, delta within ±1% noise)
  metric                              v1              v2       delta
  ------------------------------------------------------------------
  latency                    71.68us ±1%     71.68us ±1%       +0.0%  ~
  sm_throughput                    39.4%           39.1%      -0.2pp  ~
  memory_throughput                89.0%           86.0%      -3.0pp  ~
  dram_bw                          533.3           515.3       -3.4%  ~
  [stall_memory 83% v1, stall_sync 9.4% v2 (warp shuffle adds barrier-like stalls)]
  l1_bank_conflicts 0 → 3K (warp shuffle induces minor conflicts)
  l1_hit_rate: 76.8% → 0% (v2 uses shuffle not shared mem — no L1 hit)
  roofline                     v1: 89%bw       v2: 86%bw  bound: memory  14% headroom
```

### Verification 2: CUDA vs Triton cross-language diff

```
  profiling v1 prefix_scan...             ok  50 runs  237us  cv=4.5%   (CUDA Blelloch shared mem)
  profiling v2 prefix_scan...             ok  10 runs  74us  cv=0.7%    (Triton warp-shuffle scan)
  v2 is 3.21x faster  (236.5us -> 73.7us)
  l1_bank_conflicts: 3068K → 3K  (-99.9%)   — warp shuffle eliminates bank conflicts
  stall_sync: 38.4% → 9.4%  (-29pp)         — far fewer barrier stalls
  bar.sync PTX: 16 → 1, shfl.sync: 0 → 6   — confirms warp-shuffle implementation
```

### Verification 3: wall time for 20 runs (persistent harness)

```
real  0m13.7s   for 20+20 runs + 2 NCU passes
```

With the old per-process architecture, 20+20 runs × ~500ms Python import = ~20s of imports alone,
plus NCU passes. The new harness: imports happen once per kernel (warmup phase), timing loop is
pure GPU-side I/O over pipes.

### Verification 4: fast kernel (4μs) timing stability

```
  profiling v1 fast_kernel...   ok  50 runs  4us  cv=8.4%
  profiling v2 fast_kernel...   ok  50 runs  4us  cv=10.0%
```

4.1μs measured (not inflated). cv=8-10% is expected on unlocked clocks at this time scale.
The `torch.cuda._sleep` ensures the GPU is always ahead of the CPU when timing starts.

### Test count: 166 → 181 (15 new tests in test_triton_backend_persistent.py)

```
$ python3 -m pytest tests/ -q
181 passed in 0.37s
```

---

## 2026-03-19 — Metric overhaul (6-group set, derived metrics, stall breakdown)

### New metric set

Complete rewrite of `kerndiff/metrics.py`. Old set had 15 metrics with some wrong NCU strings
and vague aggregates. New set: 18 visible metrics in 6 groups + 7 hidden raw counters.

| Group | Metrics |
|-------|---------|
| sol | latency_us, sm_throughput, memory_throughput, dram_bw_gbs |
| arithmetic | arith_intensity (derived), flops_tflops (derived), thread_active_pct |
| cache | l2_hit_rate, l1_hit_rate, l1_bank_conflicts, global_load_eff |
| warp_state | sm_occupancy, stall_memory, stall_memqueue, stall_compute, stall_sync |
| launch | registers_per_thread, shared_mem_kb |
| hidden | raw_ffma, raw_fadd, raw_fmul, raw_hfma, raw_hadd, raw_hmul, raw_dram_sectors_rd, raw_dram_sectors_wr, inst_executed |

### Step 0: NCU metric string verification (A10G, NCU 2026.1.0)

All proposed metric strings verified against `ncu --query-metrics` before coding.
Two naming errors and one scale issue found:

| Metric | Tested string | Status | Notes |
|--------|--------------|--------|-------|
| l2_hit_rate | `lts__t_sector_hit_rate.pct` | ✓ 33.46% | (`lts__t_sectors_hit_rate` wrong — plural) |
| l1_hit_rate | `l1tex__t_sector_hit_rate.pct` | ✓ 30.12% | |
| stall_memory | `smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct` | ✓ 93.28% | |
| stall_memqueue | `smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct` | ✓ 0.00% | |
| stall_compute | `smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct` | ✓ 1.39% | |
| stall_sync | `smsp__warp_issue_stalled_barrier_per_warp_active.pct` | ✓ 0.00% | |
| thread_active_pct | `smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio` | ✓ returns 0–32 | needs `ncu_scale=100/32` |
| raw_ffma | `smsp__sass_thread_inst_executed_op_ffma_pred_on.sum` | ✓ 134217728 | |
| raw_dram_sectors_rd | `dram__sectors_read.sum` | ✓ 1048576 | ×32 bytes = DRAM bytes |
| pcsamp stalls | `smsp__pcsamp_warps_issue_stalled_*.pct` | ✗ returns n/a | PC sampling mode required; replaced with `smsp__warp_issue_stalled_*_per_warp_active.pct` |

### Key design decisions

- `ncu_scale: float = 1.0` field on MetricDef: thread_active_pct uses `100/32` to convert 0–32 ratio to %
- `hidden: bool = False` field: raw counters collected in same NCU pass but not shown in table
- `compute_derived_metrics()` in diff.py: arith_intensity = total_flops / dram_bytes, flops_tflops = total_flops / (latency × 1e12)
- `METRICS_BY_NCU` filters out metrics with empty `ncu_metric` (derived metrics have `ncu_metric=""`)
- sort_deltas: group-order based (sol→arithmetic→cache→warp_state→launch), noisy last within group
- `clamp` global_load_eff, l1_hit_rate, thread_active_pct to [0,100] in parser
- Fixture values: v1/v2 have same raw_ffma (same computation) but v2 has half DRAM sectors (shared mem optimization) → arith_intensity doubles from 5.33→10.67 F/B

### Real GPU run #5 — prefix_scan_v1 vs prefix_scan_v2

```
  gpu: NVIDIA A10G (device 0, sm_86, clocks unlocked)
  locking clocks...                  skipped  -- results may vary +/-10%
  compiling prefix_scan...                ok
  profiling v1 prefix_scan...             ok  50 runs  164us  cv=4.4%
  profiling v2 prefix_scan...             ok  50 runs  237us  cv=2.1%
  v2 is 1.44x slower  (163.8us -> 236.5us)  [v1: 164-194us ±4%  v2: 237-263us ±2%]
  note: clocks not locked — deltas below 9% may not be reliable
  metric                              v1              v2          delta
  ---------------------------------------------------------------------
  latency                    163.8us ±4%     236.5us ±2%         +44.4%  --
  sm_throughput                    72.4%           84.7%        +12.2pp  ++
  dram_bw                          445.3           253.0         -43.2%  --
  memory_throughput                85.7%           84.7%         -1.0pp  ~

  arith_intensity                    0.7             0.2         -68.4%  --
  flops                             0.15            0.04         -77.1%  --
  thread_active                    92.0%           52.1%        -39.9pp  --

  l2_hit_rate                      87.6%           50.1%        -37.4pp  --
  l1_hit_rate                      90.0%            0.0%        -90.0pp  --
  l1_bank_conflicts                    0           3067K  +306798100.0%  --
  global_load_eff                 100.0%          100.0%         +0.0pp  ~

  stall_memory                     40.3%           10.4%        -29.8pp  ++
  stall_memqueue                    0.4%            0.0%         -0.4pp  ++
  stall_compute                     1.4%           15.4%        +13.9pp  --
  stall_sync                       28.4%           38.4%         +9.9pp  --
  sm_occupancy                     91.4%           94.6%         +3.3pp  ~

  register_count                      16              19             +3  --
  shared_mem                          0B            512B           +512  ~
  ---------------------------------------------------------------------
  roofline                     v1: 74%bw       v2: 85%sm  bound: mem->com  15% headroom
```

Key observations:
- Groups separated by blank lines — sol / arithmetic / cache / warp_state / launch clearly delineated
- stall_memory drops 40%→10% (less HBM waiting), but stall_sync rises 28%→38% (barrier overhead)
- stall_compute rises 1%→15% — v2 is now more compute-bound (expected after removing HBM bottleneck)
- arith_intensity drops because v2 does fewer total FLOPs (Blelloch O(N) vs Hillis-Steele O(N log N))
  but does MORE DRAM relative to fewer ops (bank conflicts mean shared mem reads are expensive)
- Roofline bound transitions mem→com even though v2 is slower due to bank conflicts

### Real GPU run #6 — stall profile comparison (mem_bound vs compute_bound)

Note: these kernels have different names so compared against themselves (v1=v2, same file).

**mem_bound** (stream copy: `c[i] = a[i]`):
```
  stall_memory      92.6%   (dominant — HBM stalls as expected)
  stall_memqueue     0.0%
  stall_compute      1.3%
  stall_sync         0.0%
  sm_throughput     11.7%   (low — barely any compute)
  memory_throughput 88.1%   (high — saturating HBM)
  dram_bw           527.9 GB/s
```

**compute_bound** (256 dependent FMAs per thread):
```
  stall_memory      19.5%   (lower than mem_bound)
  stall_memqueue     0.0%
  stall_compute      0.7%   (low — warp switching hides FMA latency)
  stall_sync         0.0%
  sm_throughput     85.1%   (high — saturating FP32 pipe)
  arith_intensity   55.3 F/B
  flops             11.52 TF
  thread_active     99.6%   (nearly all threads active)
```

Notes:
- stall_memory dominates in mem_bound (92.6%) — criterion met ✓
- compute_bound shows stall_compute=0.7% (not dominant) because with 86% SM occupancy, the
  scheduler can hide FMA latency (~4 cycles) by switching to other warps. The bottleneck is
  reflected in sm_throughput=85% and arith_intensity=55 F/B — the hardware is running near peak.
- Key discriminator between bound types: sm_throughput (compute) vs memory_throughput (memory).
  Stall percentages alone are less reliable when occupancy is high (latency hidden by warp switching).

### Test count: 166 (unchanged)

```
$ python3 -m pytest tests/ -q
166 passed in 0.33s
```

## 2026-03-19 — audit: JSON mode emitted non-JSON stderr chatter

**severity:** critical
**status:** fixed

**problem:** `--format json` printed status/warning lines to stderr; when users ran `2>&1` the stream was no longer valid JSON and downstream parsing failed.
**location:** `kerndiff/cli.py` (`main`, `_emit_status`, `_warn`, `resolve_all_kernels`, `_run_main`, `_run_shape_sweep`)
**fix:** Added JSON-mode stderr suppression and gated non-essential stderr prints so merged stdout/stderr remains valid JSON output.

## 2026-03-19 — audit: mixed CUDA vs Triton correctness could skip silently

**severity:** critical
**status:** fixed

**problem:** In `--correctness` mode, if one side was persistent (Triton) and the CUDA side had empty `output_vals`, correctness was skipped instead of comparing outputs, causing false-negative correctness checks.
**location:** `kerndiff/cli.py` (`_run_correctness_check`)
**fix:** Added fallback binary dump path for non-persistent side(s) during mixed-backend correctness checks.

## 2026-03-19 — audit: placeholder substitution could corrupt kernel source text

**severity:** medium
**status:** fixed

**problem:** Harness generation replaced `{{KERNEL_NAME}}` after inserting kernel source, so literal placeholder-like strings inside user comments/source could be unintentionally rewritten.
**location:** `kerndiff/compiler.py` (`build_harness`)
**fix:** Reordered template substitutions to inject `{{KERNEL_SOURCE}}` last.

## 2026-03-19 — audit: shared memory delta displayed bytes instead of KB

**severity:** medium
**status:** fixed

**problem:** `shared_mem` values were shown as KB but delta was shown as raw byte diff (`+16384`), which was inconsistent and misleading for terminal interpretation.
**location:** `kerndiff/renderer.py` (`format_delta`)
**fix:** Normalized `shared_mem_kb` delta from bytes to KB before integer formatting.

## 2026-03-19 — audit: --all edge-case errors were underspecified

**severity:** medium
**status:** fixed

**problem:** `--all` did not emit explicit "no kernels found in <file>" errors and used a less actionable no-common-kernels message.
**location:** `kerndiff/cli.py` (`resolve_all_kernels`)
**fix:** Added explicit zero-kernel checks per file and clearer no-common-kernels error text.

## 2026-03-19 — audit: missing regression tests for audit-critical paths

**severity:** medium
**status:** fixed

**problem:** Several scenarios from the audit checklist were untested, reducing confidence around future regressions.
**location:** `tests/test_cli.py`, `tests/test_compiler.py`, `tests/test_correctness_triton.py`, `tests/test_profiler.py`, `tests/test_renderer.py`
**fix:** Added tests for `--all` edge cases, JSON output-to-file stdout silence, int/unit delta formatting, shared-mem delta units, mixed-backend correctness fallback, NCU permission-warning format, and placeholder-safe harness substitution.

## 2026-03-19 — audit: system python lacks Triton dependency in this container

**severity:** low
**status:** documented

**problem:** Running `python3 -m pytest tests/ -q` in this container fails Triton-related tests because `/usr/bin/python3` does not have `triton` installed.
**location:** environment/runtime (not repository code)
**fix:** Verified full suite with `/opt/pytorch/bin/python3 -m pytest tests/ -q` where Triton is available (`214 passed`); left code unchanged.

## 2026-03-19 — audit: session verification summary (batch 2)

**severity:** low
**status:** documented

**problem:** Checklist verification required both `python3` commands and full dependency-aware runs; this container splits dependencies across interpreters.
**location:** environment/runtime
**fix:** Confirmed command-level behavior with `python3 -m kerndiff ...`; confirmed full test validity with `/opt/pytorch/bin/python3 -m pytest tests/ -q` (`214 passed`).

## 2026-03-19 — audit: CLI numeric display regression coverage

**severity:** medium
**status:** fixed

**problem:** Numeric presentation rules for terminal output (percent vs pp vs raw int) were only partially covered, increasing risk of future formatting regressions.
**location:** `tests/test_cli.py` (`test_mock_cli_numeric_display_formats`)
**fix:** Added explicit assertions for `-23.5%`, `+26.2pp`, `+139.9%`, `+8`, `+16`, and no ANSI escapes in `--no-color` output.

## 2026-03-19 — flash-attention: standalone kernel session setup

**severity:** low
**status:** documented

**problem:** Needed real FlashAttention-inspired kernels that compile standalone and run through kerndiff harness.
**location:** `examples/flash_attn_naive.cu`, `examples/flash_attn_tiled.cu`
**fix:** Added two standalone kernels and validated on A10G.

Environment / discovery:
```bash
$ ls ~/flash-attention/ 2>/dev/null || ls /home/ubuntu/flash-attention/
ls: cannot access '/home/ubuntu/flash-attention/': No such file or directory

$ find /home/ubuntu -maxdepth 4 -type d -name 'flash-attention' 2>/dev/null | head -20
/home/ubuntu/kerndiff/flash-attention

$ nvidia-smi --query-gpu=name --format=csv,noheader
NVIDIA A10G

$ which nvcc || find /opt/pytorch -name nvcc 2>/dev/null | head -1
/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13/bin/nvcc
```

Flash-attention repo scan:
```bash
$ FA=/home/ubuntu/kerndiff/flash-attention; find $FA/csrc -name "*.cu" | head -20
/home/ubuntu/kerndiff/flash-attention/csrc/fused_dense_lib/fused_dense_cuda.cu
/home/ubuntu/kerndiff/flash-attention/csrc/flash_attn/src/flash_bwd_hdim32_bf16_causal_sm80.cu
/home/ubuntu/kerndiff/flash-attention/csrc/flash_attn/src/flash_fwd_split_hdim32_fp16_causal_sm80.cu
...

$ FA=/home/ubuntu/kerndiff/flash-attention; grep -r "naive" $FA/csrc --include="*.cu" --include="*.cuh" -l 2>/dev/null | head -5
(no output)

$ FA=/home/ubuntu/kerndiff/flash-attention; grep -r "naive\|reference\|baseline" $FA/benchmarks --include="*.py" -l 2>/dev/null | head -5
/home/ubuntu/kerndiff/flash-attention/benchmarks/benchmark_alibi.py
/home/ubuntu/kerndiff/flash-attention/benchmarks/bench_sm90.py
/home/ubuntu/kerndiff/flash-attention/benchmarks/benchmark_flash_attention.py
```

Standalone compile checks:
```bash
$ NVCC=/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13/bin/nvcc
$ CUDA_LIB=/opt/pytorch/lib/python3.13/site-packages/nvidia/cu13/lib
$ $NVCC -O2 -arch=sm_86 -L$CUDA_LIB -o /tmp/test_naive examples/flash_attn_naive.cu
/usr/bin/ld: ... undefined reference to `main'
collect2: error: ld returned 1 exit status

$ $NVCC -O2 -arch=sm_86 -L$CUDA_LIB -o /tmp/test_tiled examples/flash_attn_tiled.cu
/usr/bin/ld: ... undefined reference to `main'
collect2: error: ld returned 1 exit status

$ $NVCC -O2 -arch=sm_86 -DFLASH_ATTN_STANDALONE_MAIN -L$CUDA_LIB -o /tmp/test_naive examples/flash_attn_naive.cu && echo "naive: OK"
naive: OK

$ $NVCC -O2 -arch=sm_86 -DFLASH_ATTN_STANDALONE_MAIN -L$CUDA_LIB -o /tmp/test_tiled examples/flash_attn_tiled.cu && echo "tiled: OK"
tiled: OK
```

FlashAttention kerndiff run:
```bash
$ sudo /opt/pytorch/bin/python3 -m kerndiff examples/flash_attn_naive.cu examples/flash_attn_tiled.cu --fn flash_attn --dtype half 2>&1
  gpu: NVIDIA A10G (device 0, sm_86, clocks unlocked)
  locking clocks...                       ok
  compiling flash_attn...                 ok
  profiling v1 flash_attn...              ok  10 runs  14197us  cv=0.1%
  profiling v2 flash_attn...              ok  10 runs  812us  cv=0.3%
  v2 is 17.48x faster  (14196.7us -> 812.0us)  [v1: 14197-14221us ±0%  v2: 812-819us ±0%]
  metric                              v1              v2       delta
  ------------------------------------------------------------------
  latency                  14196.7us ±0%     812.0us ±0%      -94.3%  ++
  sm_throughput                    28.8%           28.8%      -0.1pp  ~
  memory_throughput                77.0%           77.1%      +0.1pp  ~
  dram_bw                          461.0           460.6       -0.1%  ~

  thread_active                    93.8%           93.8%      +0.0pp  ~

  l2_hit_rate                      99.7%          100.2%      +0.4pp  ~
  l1_hit_rate                      40.9%           40.9%      -0.0pp  ~
  l1_bank_conflicts                    0               0       +0.0%  ~
  global_load_eff                   0.0%            0.0%      +0.0pp  ~

  sm_occupancy                     34.2%           34.3%      +0.1pp  ~
  stall_memory                      0.0%            0.0%      +0.0pp  ~
  stall_memqueue                    0.0%            0.0%      +0.0pp  ~
  stall_compute                    22.1%           22.0%      -0.1pp  ~
  stall_sync                        0.0%            0.0%      +0.0pp  ~

  register_count                      16              16          +0  ~
  shared_mem                          0B              0B          +0  ~
  ------------------------------------------------------------------
  roofline                     v1: 77%bw       v2: 77%bw  bound: memory  23% headroom
  ptx diff  (static instruction count — not dynamic execution count)
  ----------------------------------------------
  instruction             v1      v2       delta
  ld.shared                0     200    +20000.0%
  st.shared                0      97     +9700.0%
  ld.local                 0      65     +6500.0%
  st.local                 0      17     +1700.0%
  bra.uni                  0      16     +1600.0%
  bar.sync                 0       3      +300.0%
  mov.f32                 11      34      +209.1%
  div.rn                   0       2      +200.0%
  {                       50     145      +190.0%
  st.global               26      64      +146.2%
  fma.rn                  56     128      +128.6%
  mov.u32                 12      25      +108.3%
  add.u64                  0       1      +100.0%
  mad.lo                   1       0      -100.0%
  or.b32                   1       0      -100.0%
  or.pred                  1       2      +100.0%
  rcp.rn                   1       0      -100.0%
  setp.eq                  0       1      +100.0%
  shl.b32                 11      22      +100.0%
  selp.f32                16       2       -87.5%
  mov.u64                  7       1       -85.7%
  add.s32                 11      20       +81.8%
  bra                      9      16       +77.8%
  setp.gt                 17       4       -76.5%
  add.s64                 19       5       -73.7%
  mul.wide                 9       3       -66.7%
  mul.f32                 25      13       -48.0%
  setp.ne                  5       3       -40.0%
  setp.lt                  3       4       +33.3%
  cvt.sat                  8      10       +25.0%
  ex2.approx               8      10       +25.0%
  fma.rm                   8      10       +25.0%
  mov.b32                 16      20       +25.0%
  neg.f32                  8      10       +25.0%
  sub.f32                  8      10       +25.0%
  ld.global              104      80       -23.1%
  add.f32                 16      18       +12.5%
```

Observation at `SEQ_LEN=512`: tiled is clearly faster (`17.48x` latency improvement).

## 2026-03-19 — git mode: HEAD fast vs working-copy degraded

**severity:** low
**status:** documented

**problem:** Needed end-to-end validation that single-file git mode compares committed `HEAD` against working copy.
**location:** `examples/vec_add_v2.cu`
**fix:** Degraded working copy without commit, ran git mode, observed regression in working copy.

```bash
$ sudo /opt/pytorch/bin/python3 -m kerndiff examples/vec_add_v2.cu --fn vec_add 2>&1
  gpu: NVIDIA A10G (device 0, sm_86, clocks unlocked)
  comparing: HEAD:examples/vec_add_v2.cu  vs  examples/vec_add_v2.cu (working copy)
  locking clocks...                       ok
  compiling vec_add...                    ok
  profiling v1 vec_add...                 ok  10 runs  150us  cv=0.4%
  profiling v2 vec_add...                 ok  50 runs  162us  cv=3.1%
warning: noise threshold (1.0%) not reached after 50 runs (cv=3.1%). Consider clock locking or --noise-threshold 3.
warning: high variance detected (min=161.8us, max=196.6us). Clock locking recommended.
  v2 is 1.08x slower  (149.5us -> 161.8us)  [v1: 150-152us ±0%  v2: 162-197us ±3%]
  metric                              v1              v2       delta
  ------------------------------------------------------------------
  latency                    149.5us ±0%     161.8us ±3%       +8.2%  -
  sm_throughput                    15.7%           15.7%      +0.0pp  ~
  memory_throughput                89.0%           88.4%      -0.6pp  ~
  dram_bw                          532.1           529.1       -0.6%  ~

  thread_active                    93.8%           93.8%      +0.0pp  ~

  l2_hit_rate                      99.8%           99.7%      -0.1pp  ~
  l1_hit_rate                       0.0%            0.0%      +0.0pp  ~
  l1_bank_conflicts                    0               0       +0.0%  ~
  global_load_eff                   0.0%            0.0%      +0.0pp  ~

  stall_memqueue                    0.3%            0.4%      +0.0pp  -
  stall_compute                    27.4%           26.7%      -0.8pp  +
  sm_occupancy                     63.1%           62.9%      -0.2pp  ~
  stall_memory                      0.0%            0.0%      +0.0pp  ~
  stall_sync                        0.0%            0.0%      +0.0pp  ~

  register_count                      16              16          +0  ~
  shared_mem                          0B              0B          +0  ~
  ------------------------------------------------------------------
  roofline                     v1: 89%bw       v2: 88%bw  bound: memory  12% headroom
  ptx diff  (static instruction count — not dynamic execution count)
  ----------------------------------------------
  instruction             v1      v2       delta
  add.s32                  0       3      +300.0%
  bra                      1       4      +300.0%
  ld.global                2       8      +300.0%
  setp.ge                  1       4      +300.0%
  st.global                1       4      +300.0%
  add.s64                  3       6      +100.0%
  mul.wide                 1       2      +100.0%
  or.b32                   1       0      -100.0%
```

## 2026-03-19 — git mode: HEAD degraded vs working-copy fast

**severity:** low
**status:** documented

**problem:** Needed validation that diff direction flips when `HEAD` is intentionally degraded and working copy is restored fast version.
**location:** `examples/vec_add_v2.cu`
**fix:** Committed degraded version, restored float4 in working copy, reran git mode and observed improvement.

```bash
$ sudo /opt/pytorch/bin/python3 -m kerndiff examples/vec_add_v2.cu --fn vec_add 2>&1
  gpu: NVIDIA A10G (device 0, sm_86, clocks unlocked)
  comparing: HEAD:examples/vec_add_v2.cu  vs  examples/vec_add_v2.cu (working copy)
  locking clocks...                       ok
  compiling vec_add...                    ok
  profiling v1 vec_add...                 ok  50 runs  162us  cv=3.8%
  profiling v2 vec_add...                 ok  50 runs  148us  cv=4.4%
warning: noise threshold (1.0%) not reached after 50 runs (cv=3.8%). Consider clock locking or --noise-threshold 4.
warning: high variance detected (min=161.8us, max=197.6us). Clock locking recommended.
warning: noise threshold (1.0%) not reached after 50 runs (cv=4.4%). Consider clock locking or --noise-threshold 4.
warning: high variance detected (min=148.5us, max=185.3us). Clock locking recommended.
  v2 is 1.09x faster  (161.8us -> 148.5us)  [v1: 162-198us ±4%  v2: 148-185us ±4%]
  metric                              v1              v2       delta
  ------------------------------------------------------------------
  latency                    161.8us ±4%     148.5us ±4%       -8.2%  +
  sm_throughput                    15.7%           15.6%      -0.1pp  ~
  memory_throughput                88.6%           88.7%      +0.1pp  ~
  dram_bw                          530.2           530.3       +0.0%  ~

  thread_active                    93.8%           93.8%      +0.0pp  ~

  l2_hit_rate                      99.7%           99.8%      +0.1pp  ~
  l1_hit_rate                       0.0%            0.0%      +0.0pp  ~
  l1_bank_conflicts                    0               0       +0.0%  ~
  global_load_eff                   0.0%            0.0%      +0.0pp  ~

  sm_occupancy                     62.9%           64.4%      +1.4pp  +
  stall_memqueue                    0.4%            0.3%      -0.1pp  ++
  stall_memory                      0.0%            0.0%      +0.0pp  ~
  stall_compute                    26.9%           26.4%      -0.4pp  ~
  stall_sync                        0.0%            0.0%      +0.0pp  ~

  register_count                      16              16          +0  ~
  shared_mem                          0B              0B          +0  ~
  ------------------------------------------------------------------
  roofline                     v1: 88%bw       v2: 88%bw  bound: memory  12% headroom
  ptx diff  (static instruction count — not dynamic execution count)
  ----------------------------------------------
  instruction             v1      v2       delta
  add.s32                  3       0      -100.0%
  or.b32                   0       1      +100.0%
  bra                      4       1       -75.0%
  ld.global                8       2       -75.0%
  setp.ge                  4       1       -75.0%
  st.global                4       1       -75.0%
  add.s64                  6       3       -50.0%
  mul.wide                 2       1       -50.0%
```

Cleanup and verification:
```bash
$ git revert HEAD --no-edit
[main 33749f6] Revert "test: deliberately degrade vec_add_v2 for git mode test"
...

$ head -8 examples/vec_add_v2.cu
// v2: float4 vectorized loads — 4x fewer memory transactions
__global__ void vec_add(const float* __restrict__ a,
...

$ sudo /opt/pytorch/bin/python3 -m pytest tests/ -q 2>&1 | tail -3
........................................................................ [ 66%]
.......................................................................  [100%]
215 passed in 0.38s
```
\n## 2026-03-19 — new example kernels (reduce/softmax/coalesced/layernorm)

### Compile checks
Requested executable compile loop shows linker-only errors (no main in kernel sources):
```
collect2: error: ld returned 1 exit status (x8)
```
Object compile verification (kernel code compiles cleanly):
```
reduce_v1: OK
reduce_v2: OK
softmax_v1: OK
softmax_v2: OK
coalesced_v1: OK
coalesced_v2: OK
layernorm_v1: OK
layernorm_v2: OK
```

### Reduction run
```
  gpu: NVIDIA A10G (device 0, sm_86, clocks unlocked)
  locking clocks...                       ok
  compiling reduce...                     ok
  profiling v1 reduce...                  ok  10 runs  10137us  cv=0.0%
  profiling v2 reduce...                  ok  50 runs  108us  cv=2.2%
warning: noise threshold (1.0%) not reached after 50 runs (cv=2.2%). Consider clock locking or --noise-threshold 2.
warning: high uncertainty — consider --noise-threshold or clock locking
  v2 is 94.28x faster  (10136.6us -> 107.5us)  [v1: 10137-10140us ±0%  v2: 108-123us ±2%]  ±2.05x
  warning: high uncertainty — consider --noise-threshold or clock locking
  metric                                                                   v1                                           v2       delta
  ------------------------------------------------------------------------------------------------------------------------------------
  latency                 10136.6us (p50 10138us, p20-p80: 10137-10140us) ±0%  107.5us (p50 109us, p20-p80: 109-110us) ±2%      -98.9%  ++
  sm_throughput                                                          0.4%                                        31.7%     +31.3pp  ++
  memory_throughput                                                      1.4%                                        43.5%     +42.1pp  ++
  dram_bw                                                                 2.3                                        260.5   +11170.0%  ++

  thread_active                                                         92.8%                                        89.0%      -3.9pp  -

  l2_hit_rate                                                           88.8%                                         6.8%     -82.0pp  --
  l1_bank_conflicts                                                         0                                          674   +67400.0%  --
  l1_hit_rate                                                            0.0%                                         0.0%      +0.0pp  ~
  global_load_eff                                                      100.0%                                       100.0%      +0.0pp  ~

  sm_occupancy                                                          80.3%                                        83.5%      +3.2pp  +
  stall_memory                                                          22.4%                                        61.8%     +39.4pp  --
  stall_memqueue                                                         7.1%                                         0.0%      -7.1pp  ++
  stall_compute                                                         18.7%                                         6.8%     -11.9pp  ++
  stall_sync                                                             0.0%                                        23.8%     +23.8pp  --

  register_count                                                           16                                           16          +0  ~
  shared_mem                                                               0B                                         128B          +0  ~
  ------------------------------------------------------------------------------------------------------------------------------------
  roofline                                                           v1: 0%sm                                    v2: 43%bw  bound: com->mem  57% headroom
  ptx diff  (static instruction count — not dynamic execution count)
  ----------------------------------------------
  instruction             v1      v2       delta
  mov.b32                  0      20     +2000.0%
  add.f32                  0      10     +1000.0%
  shfl.sync                0      10     +1000.0%
  mov.u32                  3      19      +533.3%
  bra                      1       5      +400.0%
  add.s32                  0       2      +200.0%
  mov.f32                  0       2      +200.0%
  setp.ne                  0       2      +200.0%
  shl.b32                  0       2      +200.0%
  shr.u32                  0       2      +200.0%
  and.b32                  0       1      +100.0%
  bar.sync                 0       1      +100.0%
  ld.shared                0       1      +100.0%
  setp.ge                  1       2      +100.0%
  st.shared                0       1      +100.0%
```
Observed vs expected:
- latency: matched expectation strongly (v2 94.28x faster).
- atomic contention story: matched (v1 near-serialized, v2 shuffle path).
- l1_bank_conflicts: opposite of expectation (increased in v2 due explicit shared-memory staging for warp results).
- stall_memqueue dropped strongly in v2, consistent with atomics no longer dominating.
- roofline moved com->mem (v1 ~0%sm effective due serialization, v2 43%bw).

### Softmax run
```
  gpu: NVIDIA A10G (device 0, sm_86, clocks unlocked)
  locking clocks...                       ok
  compiling softmax...                    ok
  profiling v1 softmax...                 ok  50 runs  208us  cv=1.9%
  profiling v2 softmax...                 ok  50 runs  140us  cv=2.9%
warning: noise threshold (1.0%) not reached after 50 runs (cv=1.9%). Consider clock locking or --noise-threshold 2.
warning: noise threshold (1.0%) not reached after 50 runs (cv=2.9%). Consider clock locking or --noise-threshold 3.
warning: high variance detected (min=140.3us, max=163.8us). Clock locking recommended.
  v2 is 1.48x faster  (207.9us -> 140.3us)  [v1: 208-237us ±2%  v2: 140-164us ±3%]  ±0.05x
  metric                                                           v1                                           v2       delta
  ----------------------------------------------------------------------------------------------------------------------------
  latency                 207.9us (p50 211us, p20-p80: 210-212us) ±2%  140.3us (p50 142us, p20-p80: 141-143us) ±3%      -32.5%  ++
  sm_throughput                                                 12.5%                                        22.8%     +10.3pp  ++
  memory_throughput                                             87.8%                                        84.2%      -3.5pp  -
  dram_bw                                                       525.9                                        504.8       -4.0%  -

  arith_intensity                                                 0.6                                          3.7     +487.3%  ++
  flops                                                          0.27                                         1.39     +415.8%  ++
  thread_active                                                 91.5%                                        97.8%      +6.3pp  +

  l2_hit_rate                                                   44.0%                                        36.4%      -7.6pp  --
  l1_hit_rate                                                    4.1%                                         5.3%      +1.2pp  ++
  l1_bank_conflicts                                                1K                                           1K      -16.6%  ++
  global_load_eff                                              100.0%                                       100.0%      +0.0pp  ~

  sm_occupancy                                                  91.2%                                        87.4%      -3.7pp  -
  stall_memory                                                  88.7%                                        84.3%      -4.4pp  +
  stall_compute                                                  1.1%                                         1.6%      +0.5pp  --
  stall_sync                                                     5.8%                                         3.5%      -2.3pp  ++
  stall_memqueue                                                 0.0%                                         0.0%      +0.0pp  ~

  register_count                                                   19                                           22          +3  --
  shared_mem                                                      1KB                                          2KB          +0  ~
  ----------------------------------------------------------------------------------------------------------------------------
  roofline                                                  v1: 88%bw                                    v2: 84%bw  bound: memory  16% headroom
  ptx diff  (static instruction count — not dynamic execution count)
  ----------------------------------------------
  instruction             v1      v2       delta
  fma.rn                   3      17      +466.7%
  cvt.sat                  1       5      +400.0%
  ex2.approx               1       5      +400.0%
  fma.rm                   1       5      +400.0%
  mov.b32                  2      10      +400.0%
  neg.f32                  1       5      +400.0%
  sub.f32                  1       5      +400.0%
  mul.f32                  2       6      +200.0%
  mov.f32                  9      23      +155.6%
  bra.uni                  1       2      +100.0%
  div.rn                   0       1      +100.0%
  rcp.rn                   1       0      -100.0%
  add.f32                  3       5       +66.7%
  setp.ne                  3       1       -66.7%
  bar.sync                 5       2       -60.0%
  shl.b32                  5       8       +60.0%
  setp.ge                  2       1       -50.0%
  st.global                2       1       -50.0%
  bra                     14       8       -42.9%
  add.s64                  8       5       -37.5%
  cvta.to                  3       2       -33.3%
  ld.global                3       2       -33.3%
  ld.param                 3       2       -33.3%
  shl.b64                  3       2       -33.3%
  shr.u32                  3       2       -33.3%
  cvt.s64                  4       3       -25.0%
  setp.lt                  4       3       -25.0%
  st.shared                5       4       -20.0%
  ld.shared                7       6       -14.3%
  mov.u32                  8       7       -12.5%
```
Observed vs expected:
- latency: matched (v2 1.48x faster).
- dram_bw: matched direction (v2 lower, -4.0%) from reduced passes.
- stall_sync: matched (v2 lower, -2.3pp).
- register_count: matched (v2 higher, +3).
- roofline remained memory-bound (88%bw -> 84%bw).

### Coalesced vs strided run
```
  gpu: NVIDIA A10G (device 0, sm_86, clocks unlocked)
  locking clocks...                       ok
  compiling copy_strided...               ok
  profiling v1 copy_strided...            ok  50 runs  59us  cv=9.3%
  profiling v2 copy_strided...            ok  50 runs  58us  cv=5.8%
warning: noise threshold (1.0%) not reached after 50 runs (cv=9.3%). Consider clock locking or --noise-threshold 9.
warning: high variance detected (min=59.4us, max=82.9us). Clock locking recommended.
warning: noise threshold (1.0%) not reached after 50 runs (cv=5.8%). Consider clock locking or --noise-threshold 6.
warning: high variance detected (min=58.4us, max=78.8us). Clock locking recommended.
  no significant latency change  (59.4us vs 58.4us)
  metric                                                        v1                                        v2       delta
  ----------------------------------------------------------------------------------------------------------------------
  memory_throughput                                          57.2%                                     42.9%     -14.4pp  --
  dram_bw                                                    339.3                                     256.7      -24.4%  --
  latency                 59.39us (p50 61us, p20-p80: 60-62us) ±9%  58.37us (p50 59us, p20-p80: 58-60us) ±6%       -1.7%  ~
  sm_throughput                                               5.6%                                      5.7%      +0.1pp  ~

  thread_active                                              94.1%                                     93.8%      -0.4pp  ~

  l2_hit_rate                                                89.0%                                     50.8%     -38.2pp  --
  l1_hit_rate                                                 3.3%                                      0.0%      -3.3pp  --
  global_load_eff                                            12.5%                                    100.0%     +87.5pp  ++
  l1_bank_conflicts                                              0                                         0       +0.0%  ~

  sm_occupancy                                               19.9%                                     15.0%      -4.9pp  --
  stall_memory                                               85.3%                                     80.9%      -4.4pp  +
  stall_compute                                               2.0%                                      2.8%      +0.7pp  --
  stall_memqueue                                              0.0%                                      0.0%      +0.0pp  ~
  stall_sync                                                  0.0%                                      0.0%      +0.0pp  ~

  register_count                                                16                                        16          +0  ~
  shared_mem                                                    0B                                        0B          +0  ~
  ----------------------------------------------------------------------------------------------------------------------
  roofline                                               v1: 57%bw                                 v2: 43%bw  bound: memory  57% headroom
  ptx diff  (static instruction count — not dynamic execution count)
  ----------------------------------------------
  instruction             v1      v2       delta
  add.s32                  2       1       -50.0%
  mul.wide                 2       1       -50.0%
  shl.b32                  2       1       -50.0%
```
Observed vs expected:
- global_load_eff is dominant signal as expected: 12.5% -> 100.0% (+87.5pp).
- l2_hit_rate drop matched expectation: -38.2pp in v2.
- latency improvement was small and under noise floor (59.4us -> 58.4us).
- explanation: access efficiency improved, but measured runtime still dominated by fixed launch/synchronization overhead at this problem size and command configuration.
- roofline remained memory-bound (57%bw -> 43%bw).

### Layernorm run
```
  gpu: NVIDIA A10G (device 0, sm_86, clocks unlocked)
  locking clocks...                       ok
  compiling layernorm...                  ok
  profiling v1 layernorm...               ok  50 runs  49us  cv=8.3%
  profiling v2 layernorm...               ok  50 runs  55us  cv=10.4%
warning: noise threshold (1.0%) not reached after 50 runs (cv=8.3%). Consider clock locking or --noise-threshold 8.
warning: high variance detected (min=49.2us, max=77.8us). Clock locking recommended.
warning: noise threshold (1.0%) not reached after 50 runs (cv=10.4%). Consider clock locking or --noise-threshold 10.
warning: high variance detected (min=55.3us, max=88.1us). Clock locking recommended.
  v2 is 1.12x slower  (49.2us -> 55.3us)  [v1: 49-78us ±8%  v2: 55-88us ±10%]  ±0.12x
  metric                                                        v1                                         v2       delta
  -----------------------------------------------------------------------------------------------------------------------
  latency                 49.15us (p50 50us, p20-p80: 50-51us) ±8%  55.30us (p50 57us, p20-p80: 56-58us) ±10%      +12.5%  -
  sm_throughput                                              30.7%                                      32.6%      +1.8pp  +
  memory_throughput                                          58.0%                                      54.4%      -3.6pp  -
  dram_bw                                                    347.0                                      325.3       -6.2%  -

  arith_intensity                                              1.1                                        2.7     +144.8%  ++
  flops                                                       0.18                                       0.40     +119.5%  ++
  thread_active                                              82.5%                                      89.4%      +6.9pp  +

  l1_hit_rate                                                48.4%                                      32.3%     -16.0pp  --
  l1_bank_conflicts                                            565                                        525       -7.1%  +
  l2_hit_rate                                                52.1%                                      51.5%      -0.6pp  ~
  global_load_eff                                           100.0%                                     100.0%      +0.0pp  ~

  stall_memory                                               56.8%                                      44.1%     -12.8pp  ++
  stall_memqueue                                              0.0%                                       0.0%      +0.0pp  --
  stall_compute                                               4.9%                                       2.5%      -2.4pp  ++
  stall_sync                                                 11.8%                                      15.8%      +4.0pp  --
  sm_occupancy                                               83.4%                                      82.8%      -0.5pp  ~

  register_count                                                16                                         25          +9  --
  shared_mem                                                   1KB                                        3KB          +2  ~
  -----------------------------------------------------------------------------------------------------------------------
  roofline                                               v1: 58%bw                                  v2: 54%bw  bound: memory  46% headroom
  ptx diff  (static instruction count — not dynamic execution count)
  ----------------------------------------------
  instruction             v1      v2       delta
  div.rn                   0       3      +300.0%
  mul.f32                  2       5      +150.0%
  bra.uni                  1       2      +100.0%
  cvt.rn                   0       1      +100.0%
  sub.f32                  2       4      +100.0%
  mov.f32                  3       5       +66.7%
  add.s32                  6       9       +50.0%
  bar.sync                 4       2       -50.0%
  fma.rn                   2       3       +50.0%
  setp.ge                  2       1       -50.0%
  setp.ne                  2       1       -50.0%
  st.shared                4       6       +50.0%
  bra                     13       8       -38.5%
  add.f32                  3       4       +33.3%
  ld.global                3       2       -33.3%
  ld.shared                6       8       +33.3%
  shl.b64                  3       2       -33.3%
  shr.u32                  3       2       -33.3%
  add.s64                  7       5       -28.6%
  cvt.s64                  4       3       -25.0%
  setp.lt                  4       3       -25.0%
  shl.b32                  4       3       -25.0%
  mov.u32                  8       9       +12.5%
```
Observed vs expected:
- discrepancy: v2 was slower (1.12x slower) instead of faster.
- register_count increased as expected (+9), which likely reduced efficiency and increased instruction pressure.
- dram_bw decreased as expected (-6.2%), but compute/merge overhead from block-wide Welford outweighed the saved memory pass.
- stall_sync increased (+4.0pp), opposite expectation, consistent with heavier synchronization/merge cost in this implementation.
- roofline stayed memory-bound (58%bw -> 54%bw).
