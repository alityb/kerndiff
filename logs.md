# kerndiff — Development Log

---

## Session: 2026-03-30

### Context
Research session comparing kerndiff's profiling methodology against the [intra-kernel-profiler (IKP)](https://github.com/yao-jz/intra-kernel-profiler) by yao-jz.

---

### Research: IKP Methodology

**Question asked:** Is IKP's approach more robust? How do they ensure accuracy without CV?

**IKP's accuracy mechanism (key insight):**
IKP doesn't rely on repeated runs to get accuracy. Instead, a single kernel launch produces _thousands_ of per-warp observations via lock-free ring buffers. Each warp writes `(timestamp, region_id, type)` events using `globaltimer` reads on lane 0 only. Cross-validation across three independent backends (globaltimer trace + NVBit instruction counts + CUPTI hardware counters) provides confidence.

**Conclusion reached:** IKP and kerndiff are solving different problems:
- IKP: "What's happening _inside_ this kernel?" (region-level, per-warp)
- kerndiff: "Is v2 faster than v1 and why?" (whole-kernel diff)

IKP is not a direct replacement, but its methodology exposed a real weakness in kerndiff's approach: **v1 and v2 are profiled sequentially, potentially minutes apart, so thermal drift can bias the comparison.**

CV tells you a measurement is _stable_ — it doesn't tell you it's correct. A "stable" v1 measurement taken at cool GPU state and a "stable" v2 measurement taken at hot GPU state are not comparable.

---

### Design Decision: Interleaved Timing

**Decision:** Implement interleaved timing (IKP-inspired, Option B) for CUDA kernels.

**Approach chosen:** Alternate `(v1, v2)` kernel launches in random-order pairs:
- For each pair, randomly choose whether to run v1 first or v2 first (eliminates position bias)
- Both kernels see the same GPU thermal/clock state within each pair
- L2 flush between every individual run (unchanged from before)
- Stop when BOTH latency lists satisfy CV threshold OR hit `max_runs`

**What we did NOT do:**
- Did not adopt IKP's `globaltimer` per-warp ring buffer approach (requires source modification or harness injection, more complexity)
- Did not implement region-level profiling (requires kernel annotation, changes the "zero annotation" philosophy)

**Architecture:**
- Added `interleave_timing(binary_a, binary_b, ...)` → `(latencies_a, latencies_b, warnings)` to `profiler.py`
- Added `pre_collected_latencies: list[float] | None = None` to `profile()` — when set, skips warmup and timing loop, goes directly to NCU
- CLI detects when both runtimes are `CUDABackend` (not Triton, not mock) and invokes interleaved path automatically — zero user-facing change required
- Triton falls back to sequential profiling (persistent processes can't be easily interleaved)

---

### Changes Made

#### `src/kerndiff/profiler.py`
- Added `pre_collected_latencies: list[float] | None = None` parameter to `profile()`
- When `pre_collected_latencies` is set: skip warmup + timing loop, emit `pre_collected_timing` trace event, go to NCU + stats
- Added `interleave_timing()` function at end of file

#### `src/kerndiff/metrics.py`
Added 5 new NCU metrics to the `warp_state` group:

| key | NCU metric | what it measures |
|-----|-----------|-----------------|
| `warp_exec_eff` | `smsp__thread_inst_executed_per_inst_executed.ratio` (scaled ×100/32) | avg active threads per warp; low = divergence or predication waste |
| `branch_divergence` | `smsp__sass_branch_targets_threads_diverged.avg.pct_of_peak_sustained_active` | % of branch targets where threads diverged |
| `stall_not_selected` | `smsp__warp_issue_stalled_no_instruction_per_warp_active.pct` | warp eligible to issue but scheduler picked another (occupancy headroom) |
| `stall_pipe_busy` | `smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct` | math pipe saturated (compute-bound signal) |
| `stall_tex_throttle` | `smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct` | texture/L1 pipe saturated |

**Why these metrics matter for diffing:**
- `warp_exec_eff` surfaces divergence improvements that don't show up in SM throughput %
- `branch_divergence` directly quantifies control-flow restructuring changes
- The three new stalls complete the picture of _why_ warps are stalled — existing stalls (`stall_memory`, `stall_compute`, `stall_sync`, `stall_memqueue`) cover long-scoreboard, short-scoreboard, barriers, and LG throttle but miss math pipe saturation, texture throttle, and scheduler "not selected" cycles

#### `src/kerndiff/cli.py`
- Added `interleave_timing` to profiler imports
- Added `pre_collected_latencies: list[float] | None = None` to `_profile_variant`
- Added interleaved timing block in `_run_single_kernel` between compilation and profiling
- Gating condition: `not args.mock and runtime_a.__class__.__name__ == "CUDABackend" and runtime_b.__class__.__name__ == "CUDABackend"`

#### `tests/test_profiler.py`
- Added `test_profile_uses_pre_collected_latencies`: verifies warmup/timing loop are skipped and trace event is emitted
- Added `test_interleave_timing_alternates_kernels`: verifies both binaries called once per pair, lists equal length ≥ min_runs

**Test result:** 228/228 passed.

---

### Open Questions / Future Work
See `IDEAS.md` for the full list. Top priorities:
1. NCU metric name validation (silent zero problem)
2. Cross-validate NCU latency vs measured latency
3. `globaltimer`-based sub-µs timing in harness
4. Correct L2 flush using cache-streaming loads
5. Register spill and tensor core utilization metrics
6. Actionable optimization suggestions based on metric patterns

---

## Session continued: 2026-03-30 (30 min extension)

After implementing the interleaved timing and new metrics, continued with the following:

### Performance Optimizations

#### Batch timing (`--multi-time N` harness flag)
**Problem:** Each timed run spawns a new subprocess (~10-20ms overhead each). For 10-50 runs = 100ms-1s of pure process overhead — dominating for short kernels.

**Fix:** Added `--multi-time N` mode to `harness_template.cu`. When used, the harness runs N timed iterations in a single process, flushing L2 between each, and prints N latency values. `_run_timed_legacy` in `profiler.py` now uses batched calls instead of one-subprocess-per-sample.

Also refactored into `_run_batch()` helper function for testability.

#### Correct L2 flush
**Problem:** Old flush used `cudaMalloc + cudaMemset + cudaFree` every run. `cudaMemset` writes through L2 (doesn't evict it). Also, malloc/free of large scratch buffers (50MB on H100) was non-trivial overhead.

**Fix:**
1. Pre-allocate scratch buffer once in a static global, lazy-reallocate only if size grows.
2. Added `_kerndiff_l2_flush` kernel using `__ldcs` (cache-streaming load instructions) which bypasses the L1/L2 cache fill path, actually evicting lines rather than just writing new ones.

### Trust / Correctness Features

#### NCU metric validation
**Problem:** If an NCU metric name doesn't exist on a given GPU/driver, NCU silently returns nothing — shows as 0.0 in kerndiff, indistinguishable from a kernel that genuinely measures 0.

**Fix:** `_check_missing_metrics()` called after NCU CSV parsing. Checks which visible (non-hidden) metrics were requested but not returned. Emits a warning listing unsupported metrics.

#### NCU latency cross-validation
**Problem:** NCU runs the kernel in a separate invocation from our timing loop. If `--launch-skip` is wrong or the kernel name doesn't match, NCU may profile a different launch — and we'd never know.

**Fix:** After NCU parse, compare `gpu__time_duration.sum` (NCU-reported kernel ns) against `min_latency_us` (measured). If ratio is outside 0.8–1.25 range, warn user. Threshold is 20%/25% to account for CUDA event vs NCU timing overhead differences.

#### Register spill detection
**Added** automatic warning when `local_load_sectors` or `local_store_sectors` is non-zero. Local memory reads/writes are register spills to L1/L2, severely impacting performance.

### New Metrics Added (second batch)

| key | what it measures |
|-----|-----------------|
| `stall_wait` | warp stalled waiting on instruction dependency (arithmetic chains) |
| `tensor_core_util` | % of peak tensor core throughput used |
| `local_load_sectors` | register spill reads (local memory load sectors) |
| `local_store_sectors` | register spill writes (local memory store sectors) |
| `sm_imbalance` | derived: `(sm_throughput / sm_occupancy) × 100%` — proxy for SM load imbalance |

### Fixture Updates
Updated both `v1_ncu.csv` and `v2_ncu.csv` with all 9 new metric entries so mock-mode tests exercise the full metric pipeline.

### Test Count
Started session: 228 tests. End of session: 233 tests (+5 new tests for new warnings and derived metrics).

---

## Session continued: 2026-03-30 (stress testing + correctness fixes)

### Stress Testing

Added two new test files:
- `tests/test_stress.py` — 92 adversarial/edge-case tests (crashes, empty inputs, malformed CSV, NaN/inf, boundary conditions, renderer edge cases)
- `tests/test_numerical_correctness.py` — 108 numerical precision tests (every formula verified with hand-computed expected values)

**Two real bugs found by stress testing:**

1. **`render_metric_table([])` crash** — `max(18, *[])` degenerates to `max(18)` which Python interprets as iterating over the integer `18`. Fixed with an early `return ""` guard in `renderer.py`.

2. **Wrong median computation assumption** — Test assumed `median([100,101,102,103,104, outlier])` equalled median of the 5 clean values. The outlier itself participates in the median calculation, shifting the threshold. Fixed the test to use correct math.

### Fix: Pairwise Speedup Point Estimate (`diff.py`)

**Problem:** `speedup = min(v1) / min(v2)` — the two minimums come from different pairs (different thermal states). The headline speedup was not derived from the same distribution as the uncertainty.

**Fix:** Replaced `_pairwise_speedup_uncertainty()` with `_pairwise_stats(latencies_a, latencies_b) → (median_speedup, stdev_of_ratios) | None`.
- Headline speedup = `median(v1_i / v2_i)` — consistent with uncertainty
- Uncertainty = `stdev(v1_i / v2_i)` — the actual spread of the ratio distribution
- Direction threshold = `max(stdev, noise_floor)` — data-driven but floored to prevent float-noise false positives
- `latency_delta_pct` = `(1/speedup - 1) * 100` when paired — consistent with the median speedup

When paired data is unavailable (Triton, mock), falls back to `min/min` and `√(cv1²+cv2²)×speedup`.

### Fix: Roofline Classification (`roofline.py`)

**Problem:** `bound = "memory" if bw_util > compute_util else "compute"` — `sm_throughput` includes memory ops and is high even for memory-bound kernels. A streaming memcopy kernel shows 90% SM throughput and gets wrongly classified as compute-bound.

**Fix 1 — Ridge point criterion:**
```
ridge_point = peak_tflops_fp32 * 1e12 / (peak_bw * 1e9)   # FLOPs/byte
bound = "compute" if arith_intensity > ridge_point else "memory"
```
Added `peak_tflops_fp32` to every `GpuSpec`, added `arith_intensity: float = 0.0` param to `compute_roofline`. Falls back to old criterion when `arith_intensity == 0` (no FP counters available).

**Fix 2 — Tensor core ridge point:**
A100 FP32 ridge = 9.75 FLOPs/byte. A100 FP16 (tensor core) ridge = 156 FLOPs/byte. A matmul with AI=50 was classified as compute-bound (50 > 9.75) but should be memory-bound (50 < 156). Added `tensor_core_util: float = 0.0` param; when `tensor_core_util >= 10%`, switches to `peak_tflops_fp16` for the ridge point.

Added `RooflineResult.ridge_point` and `RooflineResult.used_tensor_core_peak` fields. Terminal shows `[fp16]` annotation when tensor core peak was used.

**Headroom** is now bound-consistent: memory-bound headroom = distance from peak BW; compute-bound headroom = distance from peak SM throughput.

### Minor fixes

- `renderer.py`: JSON roofline payload now includes `ridge_point` and `used_tensor_core_peak`
- `renderer.py`: Shows `[fp16]` tag instead of `[spec]` when tensor core peak was used for classification

### Test Count
486 passing at end of session.
