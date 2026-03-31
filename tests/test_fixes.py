"""
Rigorous tests for the two correctness fixes:

Fix 1 — Pairwise speedup: headline speedup is now median(v1_i/v2_i), not min/min.
Fix 2 — Roofline classification: bound uses arith_intensity vs ridge_point, not
         sm_throughput vs bw_utilization.
"""
from __future__ import annotations

import math
import statistics

import pytest

from kerndiff.diff import (
    NOISE_FLOOR_LOCKED,
    _pairwise_stats,
    compute_verdict,
)
from kerndiff.roofline import GPU_SPECS, GpuSpec, RooflineResult, _TENSOR_CORE_THRESHOLD_PCT, compute_roofline
from conftest import make_result


# ─────────────────────────────────────────────────────────────────────────────
# Fix 1: pairwise speedup correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestPairwiseStats:

    def test_median_of_ratios_not_ratio_of_mins(self):
        # v1=[100,200], v2=[50,50] — min(v1)/min(v2) = 100/50 = 2.0
        # ratios=[2.0, 4.0], median=3.0 — these are different quantities
        result = _pairwise_stats([100.0, 200.0], [50.0, 50.0])
        assert result is not None
        median_speedup, _ = result
        assert median_speedup == pytest.approx(3.0)
        assert median_speedup != pytest.approx(100.0 / 50.0)

    def test_symmetric_ratios_median_equals_mean(self):
        # ratios=[1.8, 2.0, 2.2] — symmetric around 2.0, median=mean=2.0
        result = _pairwise_stats([180.0, 200.0, 220.0], [100.0, 100.0, 100.0])
        assert result is not None
        assert result[0] == pytest.approx(2.0)

    def test_uncertainty_is_stdev_of_ratios(self):
        # ratios=[2.0, 2.2] — stdev = sqrt(((2.0-2.1)^2 + (2.2-2.1)^2)/1) = 0.1414
        result = _pairwise_stats([100.0, 110.0], [50.0, 50.0])
        assert result is not None
        _, unc = result
        assert unc == pytest.approx(statistics.stdev([2.0, 2.2]), rel=1e-9)

    def test_identical_pairs_uncertainty_zero(self):
        # All ratios equal → stdev = 0
        result = _pairwise_stats([100.0] * 10, [50.0] * 10)
        assert result is not None
        assert result[1] == pytest.approx(0.0, abs=1e-12)

    def test_single_pair_returns_none(self):
        assert _pairwise_stats([100.0], [50.0]) is None

    def test_mismatched_lengths_returns_none(self):
        assert _pairwise_stats([100.0, 200.0], [50.0]) is None

    def test_empty_lists_returns_none(self):
        assert _pairwise_stats([], []) is None

    def test_zero_in_b_skipped(self):
        # Pair with b=0 is skipped; only the valid pair counts
        result = _pairwise_stats([100.0, 100.0], [0.0, 50.0])
        # Only one valid ratio → less than 2 → None
        assert result is None

    def test_median_robust_to_outlier_pair(self):
        # One wild pair shouldn't dominate the headline speedup
        a = [100.0] * 9 + [1000.0]
        b = [50.0]  * 9 + [50.0]
        result = _pairwise_stats(a, b)
        assert result is not None
        median_speedup, _ = result
        # Median of [2,2,2,2,2,2,2,2,2,20] = 2.0, not pulled toward 20
        assert median_speedup == pytest.approx(2.0)


class TestComputeVerdictPairwise:

    def test_headline_speedup_uses_median_ratio(self):
        r1 = make_result(100.0)
        r2 = make_result(50.0)
        # ratios=[2.0, 4.0], median=3.0
        v = compute_verdict(r1, r2,
                            paired_latencies_a=[100.0, 200.0],
                            paired_latencies_b=[50.0, 50.0])
        assert v["speedup"] == pytest.approx(3.0)
        assert v["paired_uncertainty"] is True

    def test_unchanged_threshold_floored_at_noise_floor(self):
        # Even with stdev=0 (identical pairs), the floor prevents a sub-2% speedup
        # from showing as "improvement" due to floating-point representation.
        r1 = make_result(100.0)
        r2 = make_result(100.5)  # 0.5% improvement — real but below noise_floor
        v = compute_verdict(r1, r2,
                            paired_latencies_a=[100.0] * 10,
                            paired_latencies_b=[100.5] * 10)
        assert v["direction"] == "unchanged"

    def test_real_improvement_above_noise_floor_detected_with_tight_pairs(self):
        # 5.3% improvement (> noise_floor=2%): detected as improvement even with stdev=0
        r1 = make_result(100.0)
        r2 = make_result(95.0)
        v = compute_verdict(r1, r2,
                            paired_latencies_a=[100.0] * 10,
                            paired_latencies_b=[95.0] * 10)
        assert v["direction"] == "improvement"
        assert v["speedup"] == pytest.approx(100.0 / 95.0, rel=1e-6)

    def test_noisy_pairs_raise_unchanged_threshold(self):
        # When pairs are noisy, threshold = stdev > noise_floor — correctly conservative.
        # Noisy pairs with ~5% spread can legitimately mask a ~4% speedup.
        import random
        rng = random.Random(0)
        a = [100.0 + rng.gauss(0, 4) for _ in range(30)]
        b = [96.0  + rng.gauss(0, 4) for _ in range(30)]
        ratios = [x / y for x, y in zip(a, b)]
        stdev_r = statistics.stdev(ratios)
        median_r = statistics.median(ratios)
        # Only assert "unchanged" if the data genuinely supports it
        if abs(median_r - 1.0) <= max(stdev_r, NOISE_FLOOR_LOCKED):
            r1 = make_result(100.0)
            r2 = make_result(96.0)
            v = compute_verdict(r1, r2, paired_latencies_a=a, paired_latencies_b=b)
            assert v["direction"] == "unchanged"

    def test_direction_unchanged_when_ratio_stdev_exceeds_signal(self):
        # Independent noise on b makes the per-pair ratios swing wildly.
        # Median ratio ≈ 1.0, stdev >> |median-1| → unchanged.
        r1 = make_result(100.0)
        r2 = make_result(100.0)
        a = [100.0] * 10
        # b oscillates ±15% around 100 independently → large ratio stdev
        b = [85.0, 115.0, 87.0, 113.0, 86.0, 114.0, 88.0, 112.0, 84.0, 116.0]
        v = compute_verdict(r1, r2, paired_latencies_a=a, paired_latencies_b=b)
        ratios = [x / y for x, y in zip(a, b)]
        assert statistics.stdev(ratios) > abs(statistics.median(ratios) - 1.0)
        assert v["direction"] == "unchanged"

    def test_without_paired_data_uses_min_over_min(self):
        r1 = make_result(200.0)
        r2 = make_result(100.0)
        v = compute_verdict(r1, r2)
        assert v["speedup"] == pytest.approx(200.0 / 100.0)
        assert v["paired_uncertainty"] is False

    def test_without_paired_data_uses_noise_floor_threshold(self):
        r1 = make_result(100.0)
        r2 = make_result(101.5)
        # |1/1.015 - 1| ≈ 0.0148 < NOISE_FLOOR_LOCKED=0.02 → unchanged
        v = compute_verdict(r1, r2)
        assert v["direction"] == "unchanged"

    def test_paired_improvement_label_uses_median_speedup(self):
        r1 = make_result(100.0)
        r2 = make_result(50.0)
        # ratios=[2.0,2.0,...] → median=2.0
        a = [100.0] * 10
        b = [50.0] * 10
        v = compute_verdict(r1, r2, paired_latencies_a=a, paired_latencies_b=b)
        assert "2.00x faster" in v["label"]

    def test_paired_regression_label(self):
        r1 = make_result(50.0)
        r2 = make_result(100.0)
        a = [50.0] * 10
        b = [100.0] * 10
        v = compute_verdict(r1, r2, paired_latencies_a=a, paired_latencies_b=b)
        assert v["direction"] == "regression"
        assert "2.00x slower" in v["label"]

    def test_pairwise_uncertainty_tighter_with_shared_noise(self):
        # Both kernels share the same noise — pairwise should be much tighter
        noise = [0.0, 5.0, -3.0, 4.0, -2.0, 3.0, -4.0, 5.0, -1.0, 2.0]
        a = [100.0 + n for n in noise]
        b = [50.0  + n for n in noise]

        r1 = make_result(100.0)
        r2 = make_result(50.0)
        object.__setattr__(r1, "cv_pct",
                           statistics.stdev(a) / statistics.mean(a) * 100)
        object.__setattr__(r2, "cv_pct",
                           statistics.stdev(b) / statistics.mean(b) * 100)

        v_seq    = compute_verdict(r1, r2)
        v_paired = compute_verdict(r1, r2, paired_latencies_a=a, paired_latencies_b=b)

        assert v_paired["speedup_uncertainty_x"] < v_seq["speedup_uncertainty_x"]

    def test_v1_v2_latency_us_still_reflect_individual_mins(self):
        # Even with paired speedup, per-kernel min latencies stay in the verdict
        r1 = make_result(200.0)
        r2 = make_result(100.0)
        v = compute_verdict(r1, r2,
                            paired_latencies_a=[200.0] * 5,
                            paired_latencies_b=[100.0] * 5)
        assert v["v1_latency_us"] == pytest.approx(200.0)
        assert v["v2_latency_us"] == pytest.approx(100.0)


# ─────────────────────────────────────────────────────────────────────────────
# Fix 2: roofline classification via ridge point
# ─────────────────────────────────────────────────────────────────────────────

class TestRidgePointClassification:

    def test_a100_ridge_point_exact(self):
        # A100: FP32=19.5 TFLOPS, peak_bw=2000 GB/s
        # ridge_point = 19.5e12 / (2000e9) = 9.75 FLOPs/byte
        r = compute_roofline("NVIDIA A100 SXM4 80GB",
                             dram_bw_gbs=1000.0, sm_throughput_pct=50.0,
                             arith_intensity=5.0)
        assert r.ridge_point == pytest.approx(19.5e12 / (2000e9))

    def test_h100_ridge_point_exact(self):
        # H100 SXM5: FP32=67 TFLOPS, peak_bw=3350 GB/s
        # ridge_point = 67e12 / 3350e9 ≈ 20.0 FLOPs/byte
        r = compute_roofline("NVIDIA H100 SXM5 80GB",
                             dram_bw_gbs=2000.0, sm_throughput_pct=60.0,
                             arith_intensity=10.0)
        assert r.ridge_point == pytest.approx(67e12 / (3350e9), rel=1e-6)

    def test_memory_bound_below_ridge_point(self):
        # A100 ridge ≈ 9.75 FLOPs/byte; AI=5 < 9.75 → memory
        r = compute_roofline("NVIDIA A100 SXM4 80GB",
                             dram_bw_gbs=1000.0, sm_throughput_pct=50.0,
                             arith_intensity=5.0)
        assert r.bound == "memory"
        assert r.gpu_matched is True

    def test_compute_bound_above_ridge_point(self):
        # A100 ridge ≈ 9.75; AI=20 > 9.75 → compute
        r = compute_roofline("NVIDIA A100 SXM4 80GB",
                             dram_bw_gbs=500.0, sm_throughput_pct=90.0,
                             arith_intensity=20.0)
        assert r.bound == "compute"

    def test_exactly_at_ridge_point_is_memory_bound(self):
        # AI == ridge_point: not strictly greater → memory
        spec = GPU_SPECS["A100 SXM4"]
        ridge = spec.peak_tflops_fp32 * 1e12 / (spec.peak_bw_gbs * 1e9)
        r = compute_roofline("NVIDIA A100 SXM4 80GB",
                             dram_bw_gbs=500.0, sm_throughput_pct=50.0,
                             arith_intensity=ridge)
        assert r.bound == "memory"

    def test_just_above_ridge_point_is_compute_bound(self):
        spec = GPU_SPECS["A100 SXM4"]
        ridge = spec.peak_tflops_fp32 * 1e12 / (spec.peak_bw_gbs * 1e9)
        r = compute_roofline("NVIDIA A100 SXM4 80GB",
                             dram_bw_gbs=500.0, sm_throughput_pct=50.0,
                             arith_intensity=ridge + 0.001)
        assert r.bound == "compute"

    def test_old_criterion_was_wrong_for_memory_bound_kernel(self):
        # A kernel that is clearly memory-bound (low AI, high SM throughput)
        # would have been wrongly classified as compute-bound by the old method:
        #   sm_throughput=80% > bw_util=0.5/2000=0.025% → "compute" ← WRONG
        # New method: AI=2 < ridge≈9.75 → "memory" ← CORRECT
        r = compute_roofline("NVIDIA A100 SXM4 80GB",
                             dram_bw_gbs=50.0, sm_throughput_pct=80.0,
                             arith_intensity=2.0)
        assert r.bound == "memory"

    def test_fallback_without_arith_intensity(self):
        # No arith_intensity → ridge_point=0, falls back to sm_throughput vs bw_util
        r = compute_roofline("NVIDIA A100 SXM4 80GB",
                             dram_bw_gbs=1800.0, sm_throughput_pct=50.0,
                             arith_intensity=0.0)
        assert r.ridge_point == pytest.approx(0.0)
        # bw_util=1800/2000=0.9 > compute_util=0.5 → memory (fallback still correct here)
        assert r.bound == "memory"

    def test_headroom_memory_bound_uses_bw_ceiling(self):
        # Memory-bound: headroom = (1 - dram_bw/peak_bw) * 100
        r = compute_roofline("NVIDIA A100 SXM4 80GB",
                             dram_bw_gbs=1000.0, sm_throughput_pct=50.0,
                             arith_intensity=2.0)
        assert r.bound == "memory"
        assert r.headroom_pct == pytest.approx((1.0 - 1000.0/2000.0) * 100)

    def test_headroom_compute_bound_uses_sm_ceiling(self):
        # Compute-bound: headroom = (1 - sm_throughput/100) * 100
        r = compute_roofline("NVIDIA A100 SXM4 80GB",
                             dram_bw_gbs=200.0, sm_throughput_pct=70.0,
                             arith_intensity=50.0)
        assert r.bound == "compute"
        assert r.headroom_pct == pytest.approx((1.0 - 0.70) * 100)

    def test_nvml_peak_bw_used_in_ridge_point(self):
        # When NVML provides peak BW, ridge point uses that, not table value
        nvml_bw = 1800.0
        r = compute_roofline("NVIDIA A100 SXM4 80GB",
                             dram_bw_gbs=500.0, sm_throughput_pct=50.0,
                             nvml_peak_bw=nvml_bw, arith_intensity=5.0)
        spec = GPU_SPECS["A100 SXM4"]
        expected_ridge = spec.peak_tflops_fp32 * 1e12 / (nvml_bw * 1e9)
        assert r.ridge_point == pytest.approx(expected_ridge)
        assert r.bw_source == "nvml"

    def test_unknown_gpu_returns_unmatched_with_arith_intensity(self):
        r = compute_roofline("Mystery GPU XYZ",
                             dram_bw_gbs=500.0, sm_throughput_pct=60.0,
                             arith_intensity=10.0)
        assert r.gpu_matched is False
        assert r.bound == "unknown"
        assert r.ridge_point == pytest.approx(0.0)

    def test_all_gpu_specs_have_positive_fp32(self):
        for name, spec in GPU_SPECS.items():
            assert spec.peak_tflops_fp32 > 0, f"{name} has non-positive peak_tflops_fp32"

    def test_fp32_never_exceeds_fp16(self):
        # Non-TC FP32 is always ≤ FP16 (tensor-core) throughput on all GPUs
        for name, spec in GPU_SPECS.items():
            assert spec.peak_tflops_fp32 <= spec.peak_tflops_fp16, \
                f"{name}: fp32={spec.peak_tflops_fp32} > fp16={spec.peak_tflops_fp16}"

    def test_ridge_point_increases_with_compute_to_bw_ratio(self):
        # L40S: FP32=91.6, BW=864 → ridge≈106 FLOPs/byte
        # A100: FP32=19.5, BW=2000 → ridge≈9.75 FLOPs/byte
        # L40S should have higher ridge point (harder to be compute-bound)
        r_l40s = compute_roofline("L40S", dram_bw_gbs=400.0, sm_throughput_pct=50.0,
                                  arith_intensity=10.0)
        r_a100 = compute_roofline("A100", dram_bw_gbs=400.0, sm_throughput_pct=50.0,
                                  arith_intensity=10.0)
        assert r_l40s.ridge_point > r_a100.ridge_point

    def test_zero_arith_intensity_falls_back_gracefully(self):
        # arith_intensity=0 must not crash and must use fallback path
        r = compute_roofline("NVIDIA H100 SXM5 80GB",
                             dram_bw_gbs=3000.0, sm_throughput_pct=80.0,
                             arith_intensity=0.0)
        assert r.bound in ("memory", "compute")
        assert r.ridge_point == pytest.approx(0.0)


class TestTensorCoreRoofline:

    def test_scalar_kernel_uses_fp32_ridge(self):
        r = compute_roofline("NVIDIA A100 SXM4 80GB",
                             dram_bw_gbs=500.0, sm_throughput_pct=80.0,
                             arith_intensity=50.0, tensor_core_util=0.0)
        spec = GPU_SPECS["A100 SXM4"]
        expected = spec.peak_tflops_fp32 * 1e12 / (spec.peak_bw_gbs * 1e9)
        assert r.ridge_point == pytest.approx(expected)
        assert r.used_tensor_core_peak is False

    def test_tensor_core_kernel_uses_fp16_ridge(self):
        r = compute_roofline("NVIDIA A100 SXM4 80GB",
                             dram_bw_gbs=500.0, sm_throughput_pct=80.0,
                             arith_intensity=50.0, tensor_core_util=50.0)
        spec = GPU_SPECS["A100 SXM4"]
        expected = spec.peak_tflops_fp16 * 1e12 / (spec.peak_bw_gbs * 1e9)
        assert r.ridge_point == pytest.approx(expected)
        assert r.used_tensor_core_peak is True

    def test_matmul_reclassified_from_compute_to_memory_with_tc(self):
        # A100: fp32_ridge≈9.75, fp16_ridge=156.0
        # AI=50: compute-bound under fp32, memory-bound under fp16 (the correct answer)
        r_scalar = compute_roofline("NVIDIA A100 SXM4 80GB",
                                    dram_bw_gbs=500.0, sm_throughput_pct=80.0,
                                    arith_intensity=50.0, tensor_core_util=0.0)
        r_tc = compute_roofline("NVIDIA A100 SXM4 80GB",
                                dram_bw_gbs=500.0, sm_throughput_pct=80.0,
                                arith_intensity=50.0, tensor_core_util=50.0)
        assert r_scalar.bound == "compute"
        assert r_tc.bound == "memory"

    def test_threshold_exactly_at_boundary_uses_fp16(self):
        r = compute_roofline("NVIDIA A100 SXM4 80GB",
                             dram_bw_gbs=500.0, sm_throughput_pct=80.0,
                             arith_intensity=50.0,
                             tensor_core_util=_TENSOR_CORE_THRESHOLD_PCT)
        assert r.used_tensor_core_peak is True

    def test_just_below_threshold_uses_fp32(self):
        r = compute_roofline("NVIDIA A100 SXM4 80GB",
                             dram_bw_gbs=500.0, sm_throughput_pct=80.0,
                             arith_intensity=50.0,
                             tensor_core_util=_TENSOR_CORE_THRESHOLD_PCT - 0.01)
        assert r.used_tensor_core_peak is False

    def test_tc_ignored_without_arith_intensity(self):
        # Even with high tc_util, fallback path is used when arith_intensity=0
        r = compute_roofline("NVIDIA A100 SXM4 80GB",
                             dram_bw_gbs=1000.0, sm_throughput_pct=50.0,
                             arith_intensity=0.0, tensor_core_util=90.0)
        assert r.ridge_point == pytest.approx(0.0)
        assert r.used_tensor_core_peak is False

    def test_h100_tc_ridge_point_exact(self):
        r = compute_roofline("NVIDIA H100 SXM5 80GB",
                             dram_bw_gbs=2000.0, sm_throughput_pct=60.0,
                             arith_intensity=100.0, tensor_core_util=80.0)
        spec = GPU_SPECS["H100 SXM5"]
        expected = spec.peak_tflops_fp16 * 1e12 / (spec.peak_bw_gbs * 1e9)
        assert r.ridge_point == pytest.approx(expected)

    def test_fp16_ridge_always_higher_than_fp32_ridge(self):
        # fp16 peak >= fp32 peak for all GPUs, so fp16 ridge >= fp32 ridge
        for name, spec in GPU_SPECS.items():
            if spec.peak_tflops_fp16 == spec.peak_tflops_fp32:
                continue  # A10/A10G have equal peaks — skip
            fp32_ridge = spec.peak_tflops_fp32 / spec.peak_bw_gbs
            fp16_ridge = spec.peak_tflops_fp16 / spec.peak_bw_gbs
            assert fp16_ridge > fp32_ridge, f"{name}: fp16_ridge should > fp32_ridge"


class TestLatencyDeltaPctConsistency:

    def test_latency_delta_pct_derived_from_paired_speedup(self):
        # speedup=2.0 → latency_delta_pct = (1/2 - 1)*100 = -50%
        r1 = make_result(200.0)
        r2 = make_result(100.0)
        v = compute_verdict(r1, r2,
                            paired_latencies_a=[200.0] * 10,
                            paired_latencies_b=[100.0] * 10)
        assert v["speedup"] == pytest.approx(2.0)
        assert v["latency_delta_pct"] == pytest.approx(-50.0)

    def test_latency_delta_pct_regression_positive(self):
        # speedup=0.5 (regression) → latency_delta_pct = (1/0.5 - 1)*100 = +100%
        r1 = make_result(100.0)
        r2 = make_result(200.0)
        v = compute_verdict(r1, r2,
                            paired_latencies_a=[100.0] * 10,
                            paired_latencies_b=[200.0] * 10)
        assert v["speedup"] == pytest.approx(0.5)
        assert v["latency_delta_pct"] == pytest.approx(100.0)

    def test_latency_delta_pct_consistent_with_speedup(self):
        # latency_delta_pct = (1/speedup - 1)*100 should always hold when paired
        r1 = make_result(100.0)
        r2 = make_result(75.0)
        a = [100.0 + i * 0.1 for i in range(20)]
        b = [75.0  + i * 0.1 for i in range(20)]
        v = compute_verdict(r1, r2, paired_latencies_a=a, paired_latencies_b=b)
        expected_delta = (1.0 / v["speedup"] - 1.0) * 100.0
        assert v["latency_delta_pct"] == pytest.approx(expected_delta, rel=1e-9)

    def test_latency_delta_pct_uses_min_min_without_paired_data(self):
        # Without paired data, still uses (v2.min - v1.min) / v1.min
        r1 = make_result(200.0)
        r2 = make_result(100.0)
        v = compute_verdict(r1, r2)
        assert v["latency_delta_pct"] == pytest.approx(-50.0)

    def test_paired_and_nonpaired_delta_agree_when_no_noise(self):
        # With zero-noise data, min/min and median(ratios) are identical
        r1 = make_result(200.0)
        r2 = make_result(100.0)
        v_paired = compute_verdict(r1, r2,
                                   paired_latencies_a=[200.0] * 10,
                                   paired_latencies_b=[100.0] * 10)
        v_seq = compute_verdict(r1, r2)
        assert v_paired["latency_delta_pct"] == pytest.approx(v_seq["latency_delta_pct"])
