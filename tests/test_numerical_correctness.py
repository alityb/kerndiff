"""
Numerical correctness tests for kerndiff.

Every test here verifies an EXACT NUMBER, not just that code doesn't crash.
If a test breaks, it means we are potentially misleading the user about
kernel performance.  All expected values are hand-computed from first principles.
"""
from __future__ import annotations

import math
import subprocess
import statistics

import pytest

from kerndiff.diff import (
    NOISE_FLOOR_LOCKED,
    NOISE_FLOOR_UNLOCKED,
    compute_all_deltas,
    compute_delta,
    compute_derived_metrics,
    compute_verdict,
)
from kerndiff.metrics import METRICS_BY_KEY
from kerndiff.parser import parse_ncu_csv, parse_ncu_csv_pipeline
from kerndiff.profiler import (
    MOCK_HARDWARE,
    _compute_cv,
    _percentile,
    _remove_outliers,
    interleave_timing,
    profile,
)
from kerndiff.roofline import compute_roofline
from conftest import make_result


# ─────────────────────────────────────────────────────────────────────────────
# 1. Unit conversions in parser
# ─────────────────────────────────────────────────────────────────────────────

class TestUnitConversions:
    """Every number that leaves the parser must be in the right unit."""

    def test_nanoseconds_to_microseconds(self):
        # 247300 ns ÷ 1000 = 247.300 µs exactly
        csv = '"Metric Name","Metric Unit","Metric Value"\n"gpu__time_duration.sum","nsecond","247300"\n'
        assert parse_ncu_csv(csv)["latency_us"] == pytest.approx(247.3)

    def test_ns_abbreviation_to_microseconds(self):
        csv = '"Metric Name","Metric Unit","Metric Value"\n"gpu__time_duration.sum","ns","1000000"\n'
        assert parse_ncu_csv(csv)["latency_us"] == pytest.approx(1000.0)

    def test_byte_per_second_to_gbs(self):
        # 900,000,000,000 byte/s ÷ 1e9 = 900.0 GB/s
        csv = '"Metric Name","Metric Unit","Metric Value"\n"dram__bytes.sum.per_second","byte/second","900000000000"\n'
        assert parse_ncu_csv(csv)["dram_bw_gbs"] == pytest.approx(900.0)

    def test_byte_s_variant_to_gbs(self):
        csv = '"Metric Name","Metric Unit","Metric Value"\n"dram__bytes.sum.per_second","byte/s","500000000000"\n'
        assert parse_ncu_csv(csv)["dram_bw_gbs"] == pytest.approx(500.0)

    def test_percent_unit_no_conversion(self):
        # percent values are stored as-is
        csv = '"Metric Name","Metric Unit","Metric Value"\n"lts__t_sector_hit_rate.pct","percent","41.2"\n'
        assert parse_ncu_csv(csv)["l2_hit_rate"] == pytest.approx(41.2)

    def test_unitless_count_no_conversion(self):
        csv = '"Metric Name","Metric Unit","Metric Value"\n"smsp__inst_executed.sum","inst","5000000"\n'
        assert parse_ncu_csv(csv)["inst_executed"] == pytest.approx(5_000_000.0)

    def test_thread_active_ncu_scale_full(self):
        # NCU returns 0–32 ratio; scale = 100/32; full active = 32 → 100%
        csv = '"Metric Name","Metric Unit","Metric Value"\n"smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio","","32.0"\n'
        assert parse_ncu_csv(csv)["thread_active_pct"] == pytest.approx(100.0)

    def test_thread_active_ncu_scale_half(self):
        # 16.0 → 16 * 100/32 = 50.0%
        csv = '"Metric Name","Metric Unit","Metric Value"\n"smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio","","16.0"\n'
        assert parse_ncu_csv(csv)["thread_active_pct"] == pytest.approx(50.0)

    def test_thread_active_ncu_scale_quarter(self):
        # 8.0 → 8 * 100/32 = 25.0%
        csv = '"Metric Name","Metric Unit","Metric Value"\n"smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio","","8.0"\n'
        assert parse_ncu_csv(csv)["thread_active_pct"] == pytest.approx(25.0)

    def test_thread_active_clamped_at_100(self):
        # NCU replay artifacts can push ratio above 32; clamp to 100%
        csv = '"Metric Name","Metric Unit","Metric Value"\n"smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio","","33.6"\n'
        assert parse_ncu_csv(csv)["thread_active_pct"] == pytest.approx(100.0)

    def test_warp_exec_eff_ncu_scale_full(self):
        # 32.0 * 100/32 = 100.0%
        csv = '"Metric Name","Metric Unit","Metric Value"\n"smsp__thread_inst_executed_per_inst_executed.ratio","","32.0"\n'
        assert parse_ncu_csv(csv)["warp_exec_eff"] == pytest.approx(100.0)

    def test_warp_exec_eff_ncu_scale_three_quarters(self):
        # 24.0 * 100/32 = 75.0%
        csv = '"Metric Name","Metric Unit","Metric Value"\n"smsp__thread_inst_executed_per_inst_executed.ratio","","24.0"\n'
        assert parse_ncu_csv(csv)["warp_exec_eff"] == pytest.approx(75.0)

    def test_warp_exec_eff_ncu_scale_zero(self):
        # 0.0 → 0.0%
        csv = '"Metric Name","Metric Unit","Metric Value"\n"smsp__thread_inst_executed_per_inst_executed.ratio","","0.0"\n'
        assert parse_ncu_csv(csv)["warp_exec_eff"] == pytest.approx(0.0)

    def test_global_load_eff_clamped_above_100(self):
        # NCU replay can push coalescing % above 100; clamp
        csv = '"Metric Name","Metric Unit","Metric Value"\n"smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct","percent","112.4"\n'
        assert parse_ncu_csv(csv)["global_load_eff"] == pytest.approx(100.0)

    def test_global_load_eff_not_clamped_below_100(self):
        csv = '"Metric Name","Metric Unit","Metric Value"\n"smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct","percent","79.3"\n'
        assert parse_ncu_csv(csv)["global_load_eff"] == pytest.approx(79.3)

    def test_l1_hit_rate_clamped_above_100(self):
        csv = '"Metric Name","Metric Unit","Metric Value"\n"l1tex__t_sector_hit_rate.pct","percent","105.0"\n'
        assert parse_ncu_csv(csv)["l1_hit_rate"] == pytest.approx(100.0)

    def test_large_number_with_commas(self):
        # 134,217,728 = 2^27
        csv = '"Metric Name","Metric Unit","Metric Value"\n"smsp__sass_thread_inst_executed_op_ffma_pred_on.sum","inst","134,217,728"\n'
        assert parse_ncu_csv(csv)["raw_ffma"] == pytest.approx(134_217_728.0)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Delta formula and symbol assignment — exact boundaries
# ─────────────────────────────────────────────────────────────────────────────

class TestDeltaFormulaPrecision:
    """
    delta_pct = ((v2 - v1) / |v1|) * 100
    Noise floor (default NOISE_FLOOR_LOCKED=0.02) → threshold = 2.0%
    Symbols: |Δ| < 2% → "~"; ≥15% favorable → "++"; 2–14% favorable → "+"
             ≥15% unfavorable → "--"; 2–14% unfavorable → "-"
    """

    def test_delta_pct_formula_exact(self):
        d = compute_delta(METRICS_BY_KEY["latency_us"], 200.0, 150.0)
        # ((150-200)/200)*100 = -25.0%
        assert d.delta_pct == pytest.approx(-25.0)

    def test_delta_pct_increase_exact(self):
        d = compute_delta(METRICS_BY_KEY["l2_hit_rate"], 40.0, 60.0)
        # ((60-40)/40)*100 = 50.0%
        assert d.delta_pct == pytest.approx(50.0)

    def test_delta_pct_v1_zero_uses_denom_one(self):
        # v1=0 → denominator fallback=1.0 → delta_pct = v2*100
        d = compute_delta(METRICS_BY_KEY["l2_hit_rate"], 0.0, 5.0)
        assert d.delta_pct == pytest.approx(500.0)

    # Symbol boundaries — noise_floor=NOISE_FLOOR_LOCKED=0.02 → threshold=2.0%

    def test_just_below_noise_floor_is_tilde(self):
        # |Δ| = 1.99% < 2.0% → "~"
        d = compute_delta(METRICS_BY_KEY["latency_us"], 100.0, 98.01)
        assert abs(d.delta_pct) == pytest.approx(1.99)
        assert d.symbol == "~"

    def test_exactly_at_noise_floor_is_plus(self):
        # |Δ| = 2.0% exactly → NOT < 2.0 → passes noise check → "+"
        d = compute_delta(METRICS_BY_KEY["latency_us"], 100.0, 98.0)
        assert abs(d.delta_pct) == pytest.approx(2.0)
        assert d.symbol == "+"

    def test_just_below_double_plus_threshold(self):
        # |Δ| = 14.99% → "+"
        d = compute_delta(METRICS_BY_KEY["latency_us"], 100.0, 85.01)
        assert abs(d.delta_pct) == pytest.approx(14.99)
        assert d.symbol == "+"

    def test_exactly_at_double_plus_threshold(self):
        # |Δ| = 15.0% exactly → "++"
        d = compute_delta(METRICS_BY_KEY["latency_us"], 100.0, 85.0)
        assert abs(d.delta_pct) == pytest.approx(15.0)
        assert d.symbol == "++"

    def test_unfavorable_just_below_noise_floor(self):
        # latency increase 1.99% → "~"
        d = compute_delta(METRICS_BY_KEY["latency_us"], 100.0, 101.99)
        assert d.symbol == "~"

    def test_unfavorable_exactly_at_noise_floor(self):
        # latency increase 2.0% exactly → "-"
        d = compute_delta(METRICS_BY_KEY["latency_us"], 100.0, 102.0)
        assert d.symbol == "-"

    def test_unfavorable_exactly_at_double_minus_threshold(self):
        # latency increase 15.0% → "--"
        d = compute_delta(METRICS_BY_KEY["latency_us"], 100.0, 115.0)
        assert d.symbol == "--"

    def test_noise_floor_unlocked_threshold_5pct(self):
        # With NOISE_FLOOR_UNLOCKED=0.05, threshold is 5%, not 2%
        d = compute_delta(METRICS_BY_KEY["latency_us"], 100.0, 97.0,
                          noise_floor=NOISE_FLOOR_UNLOCKED)
        # |Δ|=3% < 5% → "~"
        assert d.delta_pct == pytest.approx(-3.0)
        assert d.symbol == "~"

    def test_noise_floor_unlocked_just_above_threshold(self):
        d = compute_delta(METRICS_BY_KEY["latency_us"], 100.0, 94.0,
                          noise_floor=NOISE_FLOOR_UNLOCKED)
        # |Δ|=6% > 5% and < 15% → "+"
        assert d.symbol == "+"

    def test_none_lower_is_better_always_tilde(self):
        # sm_imbalance has lower_is_better=None → symbol always "~"
        d = compute_delta(METRICS_BY_KEY["sm_imbalance"], 100.0, 50.0)
        assert d.symbol == "~"
        # But delta_pct is still computed
        assert d.delta_pct == pytest.approx(-50.0)

    def test_higher_is_better_increase_favorable(self):
        # warp_exec_eff: lower_is_better=False; 60→80 = +33% → "++"
        d = compute_delta(METRICS_BY_KEY["warp_exec_eff"], 60.0, 80.0)
        assert d.favorable is True
        assert d.delta_pct == pytest.approx(100.0 * (80 - 60) / 60)
        assert d.symbol == "++"

    def test_higher_is_better_decrease_unfavorable(self):
        # warp_exec_eff: 80→60 = -25% → "--" (unfavorable)
        d = compute_delta(METRICS_BY_KEY["warp_exec_eff"], 80.0, 60.0)
        assert d.favorable is False
        assert d.symbol == "--"

    def test_lower_is_better_decrease_favorable(self):
        # branch_divergence: lower_is_better=True; 20→5 = -75% → "++"
        d = compute_delta(METRICS_BY_KEY["branch_divergence"], 20.0, 5.0)
        assert d.favorable is True
        assert d.symbol == "++"

    def test_delta_pct_stored_correctly(self):
        # Verify delta_pct stores the signed value (v2-v1)/|v1| * 100
        d = compute_delta(METRICS_BY_KEY["sm_occupancy"], 50.0, 75.0)
        assert d.delta_pct == pytest.approx(50.0)   # +50%
        d2 = compute_delta(METRICS_BY_KEY["sm_occupancy"], 75.0, 50.0)
        assert d2.delta_pct == pytest.approx(-100.0 * 25 / 75)


# ─────────────────────────────────────────────────────────────────────────────
# 3. Speedup and verdict math
# ─────────────────────────────────────────────────────────────────────────────

class TestVerdictMath:
    """Speedup, uncertainty, direction, latency_delta_pct — all hand-verified."""

    def test_speedup_2x(self):
        v = compute_verdict(make_result(200.0), make_result(100.0))
        assert v["speedup"] == pytest.approx(2.0)

    def test_speedup_half(self):
        v = compute_verdict(make_result(100.0), make_result(200.0))
        assert v["speedup"] == pytest.approx(0.5)

    def test_speedup_exactly_1(self):
        v = compute_verdict(make_result(100.0), make_result(100.0))
        assert v["speedup"] == pytest.approx(1.0)

    def test_direction_improvement(self):
        v = compute_verdict(make_result(200.0), make_result(100.0))
        assert v["direction"] == "improvement"
        assert "2.00x faster" in v["label"]

    def test_direction_regression(self):
        v = compute_verdict(make_result(100.0), make_result(200.0))
        assert v["direction"] == "regression"
        assert "2.00x slower" in v["label"]

    def test_direction_unchanged_inside_noise_floor(self):
        # speedup = 200/201 ≈ 0.995 → |speedup-1| = 0.005 < 0.02 → unchanged
        v = compute_verdict(make_result(200.0), make_result(201.0))
        assert abs(v["speedup"] - 1.0) < NOISE_FLOOR_LOCKED
        assert v["direction"] == "unchanged"

    def test_direction_improvement_just_outside_noise_floor(self):
        # speedup must satisfy |speedup-1| >= noise_floor=0.02
        # v1=100, v2=98 → speedup≈1.0204 → |1.0204-1|=0.0204 > 0.02 → improvement
        v = compute_verdict(make_result(100.0), make_result(98.0))
        assert v["direction"] == "improvement"

    def test_speedup_uncertainty_formula(self):
        # rel_err = sqrt((cv1/100)² + (cv2/100)²); unc = speedup * rel_err
        # cv1=3, cv2=4 → rel_err=sqrt(0.0009+0.0016)=sqrt(0.0025)=0.05
        # speedup=2.0 → unc=0.10
        r1 = make_result(200.0)
        r2 = make_result(100.0)
        # Manually patch cv_pct
        object.__setattr__(r1, "cv_pct", 3.0)
        object.__setattr__(r2, "cv_pct", 4.0)
        v = compute_verdict(r1, r2)
        expected_unc = 2.0 * math.sqrt(0.03**2 + 0.04**2)
        assert v["speedup_uncertainty_x"] == pytest.approx(expected_unc, rel=1e-6)

    def test_latency_delta_pct_formula(self):
        # ((v2-v1)/v1) * 100; v1=200, v2=100 → -50%
        v = compute_verdict(make_result(200.0), make_result(100.0))
        assert v["latency_delta_pct"] == pytest.approx(-50.0)

    def test_latency_delta_pct_regression(self):
        # v1=100, v2=150 → +50%
        v = compute_verdict(make_result(100.0), make_result(150.0))
        assert v["latency_delta_pct"] == pytest.approx(50.0)

    def test_noise_floor_pct_in_verdict(self):
        v = compute_verdict(make_result(100.0), make_result(90.0))
        assert v["noise_floor_pct"] == pytest.approx(NOISE_FLOOR_LOCKED * 100)

    def test_verdict_uses_min_latency_not_median(self):
        # verdict["v1_latency_us"] must be min_latency_us, not median
        r = make_result(100.0)
        # make_result sets min_latency_us=100.0, but median is higher
        v = compute_verdict(r, r)
        assert v["v1_latency_us"] == r.min_latency_us
        assert v["v1_latency_us"] != r.median_latency_us  # they differ

    def test_paired_uncertainty_is_tighter_than_independent(self):
        """
        With paired latencies where both kernels share the same noise,
        pairwise uncertainty must be strictly smaller than √(cv1²+cv2²)×speedup.
        This verifies the fundamental purpose of paired measurement.
        """
        import math as _math
        # Both kernels have a shared noise component — thermal spike affects both equally
        shared_noise = [0.0, 2.0, -1.0, 3.0, -2.0, 1.5, -0.5, 2.5, -1.5, 1.0]
        v1_lats = [100.0 + n for n in shared_noise]   # ~100µs ± noise
        v2_lats = [50.0  + n for n in shared_noise]   # ~50µs  ± same noise

        r1 = make_result(100.0)
        r2 = make_result(50.0)
        # Patch cv_pct to reflect the actual series CV
        import statistics as _stats
        cv1 = (_stats.stdev(v1_lats) / _stats.mean(v1_lats)) * 100
        cv2 = (_stats.stdev(v2_lats) / _stats.mean(v2_lats)) * 100
        object.__setattr__(r1, "cv_pct", cv1)
        object.__setattr__(r2, "cv_pct", cv2)

        v_independent = compute_verdict(r1, r2)
        v_paired = compute_verdict(r1, r2, paired_latencies_a=v1_lats, paired_latencies_b=v2_lats)

        assert v_paired["paired_uncertainty"] is True
        assert v_independent["paired_uncertainty"] is False
        # With shared noise, paired uncertainty must be much tighter
        assert v_paired["speedup_uncertainty_x"] < v_independent["speedup_uncertainty_x"]

    def test_paired_uncertainty_formula_exact(self):
        """
        Per-pair speedup_i = v1_i / v2_i; uncertainty = CV(speedup_i) × mean(speedup_i).
        Verify with a case where the exact answer is known.
        """
        import math as _math, statistics as _stats
        # v1 always 100, v2 always 50 → all ratios exactly 2.0 → CV=0 → uncertainty=0
        v1_lats = [100.0] * 10
        v2_lats = [50.0] * 10
        r1 = make_result(100.0)
        r2 = make_result(50.0)
        v = compute_verdict(r1, r2, paired_latencies_a=v1_lats, paired_latencies_b=v2_lats)
        assert v["speedup"] == pytest.approx(2.0)
        assert v["speedup_uncertainty_x"] == pytest.approx(0.0, abs=1e-10)

    def test_paired_uncertainty_with_known_ratio_variance(self):
        """
        v1=[100,110], v2=[50,50] → ratios=[2.0, 2.2]
        mean_ratio=2.1, stdev_ratio=√((0.1²+0.1²)/1)=0.1414, cv=0.1414/2.1
        uncertainty = mean_ratio × cv = 2.1 × (0.1414/2.1) = 0.1414
        """
        import math as _math
        v1_lats = [100.0, 110.0]
        v2_lats = [50.0, 50.0]
        ratios = [2.0, 2.2]
        mean_r = 2.1
        stdev_r = _math.sqrt(((2.0-2.1)**2 + (2.2-2.1)**2) / 1)  # sample stdev
        expected_unc = mean_r * (stdev_r / mean_r)   # = stdev_r
        r1 = make_result(100.0)
        r2 = make_result(50.0)
        v = compute_verdict(r1, r2, paired_latencies_a=v1_lats, paired_latencies_b=v2_lats)
        assert v["speedup_uncertainty_x"] == pytest.approx(expected_unc, rel=1e-9)

    def test_falls_back_to_independent_without_paired_data(self):
        """Without paired latencies, must use √(cv1²+cv2²)×speedup."""
        import math as _math
        r1 = make_result(200.0)
        r2 = make_result(100.0)
        object.__setattr__(r1, "cv_pct", 3.0)
        object.__setattr__(r2, "cv_pct", 4.0)
        v = compute_verdict(r1, r2)
        expected = 2.0 * _math.sqrt(0.03**2 + 0.04**2)
        assert v["speedup_uncertainty_x"] == pytest.approx(expected, rel=1e-6)
        assert v["paired_uncertainty"] is False

    def test_verdict_v1_min_max_from_all_latencies(self):
        r1 = make_result(100.0)
        r2 = make_result(50.0)
        v = compute_verdict(r1, r2)
        assert v["v1_min_us"] == min(r1.all_latencies_us)
        assert v["v1_max_us"] == max(r1.all_latencies_us)


# ─────────────────────────────────────────────────────────────────────────────
# 4. Derived metrics — FLOP counting and arithmetic intensity
# ─────────────────────────────────────────────────────────────────────────────

class TestDerivedMetricsMath:
    """
    FLOP counting rules (per thread, predicated-on):
      ffma = 2 FLOPs (fused multiply-add counts as 2)
      fadd, fmul = 1 FLOP each
      hfma = 2 FLOPs (FP16)
      hadd, hmul = 1 FLOP each

    DRAM bytes = (rd_sectors + wr_sectors) * 32
    arith_intensity = total_flops / dram_bytes
    flops_tflops = total_flops / (latency_us * 1e-6) / 1e12
    sm_imbalance = (sm_throughput / sm_occupancy) * 100
    """

    def test_ffma_counts_as_two_flops(self):
        d = compute_derived_metrics({
            "raw_ffma": 1_000_000,
            "raw_dram_sectors_rd": 1_000, "raw_dram_sectors_wr": 0,
            "latency_us": 1.0,
        })
        # total_flops = 2*1e6, dram_bytes = 1000*32 = 32000
        assert d["arith_intensity"] == pytest.approx(2_000_000 / 32_000)  # 62.5

    def test_fadd_counts_as_one_flop(self):
        d = compute_derived_metrics({
            "raw_fadd": 500_000,
            "raw_dram_sectors_rd": 500, "raw_dram_sectors_wr": 0,
            "latency_us": 1.0,
        })
        # total_flops = 500_000, dram_bytes = 500*32 = 16000
        assert d["arith_intensity"] == pytest.approx(500_000 / 16_000)  # 31.25

    def test_fmul_counts_as_one_flop(self):
        d = compute_derived_metrics({
            "raw_fmul": 1_000_000,
            "raw_dram_sectors_rd": 1_000, "raw_dram_sectors_wr": 0,
            "latency_us": 1.0,
        })
        assert d["arith_intensity"] == pytest.approx(1_000_000 / 32_000)  # 31.25

    def test_hfma_counts_as_two_flops(self):
        d = compute_derived_metrics({
            "raw_hfma": 2_000_000,
            "raw_dram_sectors_rd": 2_000, "raw_dram_sectors_wr": 0,
            "latency_us": 1.0,
        })
        assert d["arith_intensity"] == pytest.approx(4_000_000 / 64_000)  # 62.5

    def test_mixed_fp32_fp16_flops_sum(self):
        # ffma=1M (2M FLOPs), hfma=1M (2M FLOPs), fadd=1M (1M FLOPs) → 5M total
        d = compute_derived_metrics({
            "raw_ffma": 1_000_000,
            "raw_hfma": 1_000_000,
            "raw_fadd": 1_000_000,
            "raw_dram_sectors_rd": 1_000, "raw_dram_sectors_wr": 0,
            "latency_us": 1.0,
        })
        assert d["arith_intensity"] == pytest.approx(5_000_000 / 32_000)  # 156.25

    def test_dram_bytes_includes_both_read_and_write(self):
        # rd=1000 sectors + wr=500 sectors = 1500 * 32 = 48000 bytes
        d = compute_derived_metrics({
            "raw_ffma": 1_000_000,
            "raw_dram_sectors_rd": 1_000,
            "raw_dram_sectors_wr": 500,
            "latency_us": 1.0,
        })
        assert d["arith_intensity"] == pytest.approx(2_000_000 / 48_000)

    def test_flops_tflops_at_one_second(self):
        # 2e12 FLOPs, latency=1s=1e6 µs → 2 TFLOPS
        d = compute_derived_metrics({
            "raw_ffma": 1_000_000_000_000,
            "raw_dram_sectors_rd": 1_000, "raw_dram_sectors_wr": 0,
            "latency_us": 1_000_000.0,
        })
        assert d["flops_tflops"] == pytest.approx(2.0, rel=1e-9)

    def test_flops_tflops_at_one_microsecond(self):
        # 2e6 FLOPs, latency=1µs → 2e6/1e-6/1e12 = 2.0 TFLOPS
        d = compute_derived_metrics({
            "raw_ffma": 1_000_000,
            "raw_dram_sectors_rd": 1, "raw_dram_sectors_wr": 0,
            "latency_us": 1.0,
        })
        assert d["flops_tflops"] == pytest.approx(2.0, rel=1e-6)

    def test_sm_imbalance_exact(self):
        d = compute_derived_metrics({"sm_throughput": 60.0, "sm_occupancy": 80.0})
        # (60/80)*100 = 75.0
        assert d["sm_imbalance"] == pytest.approx(75.0)

    def test_sm_imbalance_balanced_is_100(self):
        d = compute_derived_metrics({"sm_throughput": 80.0, "sm_occupancy": 80.0})
        assert d["sm_imbalance"] == pytest.approx(100.0)

    def test_arith_intensity_not_computed_without_flops(self):
        d = compute_derived_metrics({
            "raw_ffma": 0,
            "raw_dram_sectors_rd": 1_000, "raw_dram_sectors_wr": 0,
            "latency_us": 100.0,
        })
        assert "arith_intensity" not in d

    def test_arith_intensity_not_computed_without_dram(self):
        d = compute_derived_metrics({
            "raw_ffma": 1_000_000,
            "raw_dram_sectors_rd": 0, "raw_dram_sectors_wr": 0,
            "latency_us": 100.0,
        })
        assert "arith_intensity" not in d

    def test_flops_tflops_not_computed_without_latency(self):
        d = compute_derived_metrics({
            "raw_ffma": 1_000_000,
            "raw_dram_sectors_rd": 100, "raw_dram_sectors_wr": 0,
            "latency_us": 0.0,
        })
        assert "flops_tflops" not in d


# ─────────────────────────────────────────────────────────────────────────────
# 5. Percentile interpolation — exact values from first principles
# ─────────────────────────────────────────────────────────────────────────────

class TestPercentilePrecision:
    """
    Formula: pos=(n-1)*q, lo=floor(pos), hi=min(lo+1,n-1), frac=pos-lo
    result = s[lo]*(1-frac) + s[hi]*frac
    """

    def test_p20_five_elements(self):
        # s=[10,20,30,40,50], n=5, p=0.2
        # pos=4*0.2=0.8, lo=0, hi=1, frac=0.8
        # result=10*0.2 + 20*0.8 = 2+16 = 18.0
        assert _percentile([10, 20, 30, 40, 50], 0.2) == pytest.approx(18.0)

    def test_p80_five_elements(self):
        # pos=4*0.8=3.2, lo=3, hi=4, frac=0.2
        # result=40*0.8 + 50*0.2 = 32+10 = 42.0
        assert _percentile([10, 20, 30, 40, 50], 0.8) == pytest.approx(42.0)

    def test_p50_five_elements(self):
        # pos=4*0.5=2.0, lo=2, hi=3, frac=0.0
        # result=30*1.0 + 40*0.0 = 30.0
        assert _percentile([10, 20, 30, 40, 50], 0.5) == pytest.approx(30.0)

    def test_p0_is_minimum(self):
        # pos=0, lo=0, hi=1, frac=0 → s[0]
        assert _percentile([10, 20, 30], 0.0) == pytest.approx(10.0)

    def test_p1_is_maximum(self):
        # pos=n-1, lo=n-1, hi=n-1, frac=0 → s[n-1]
        assert _percentile([10, 20, 30], 1.0) == pytest.approx(30.0)

    def test_p50_two_elements(self):
        # pos=1*0.5=0.5, lo=0, hi=1, frac=0.5
        # result = 10*0.5 + 20*0.5 = 15.0
        assert _percentile([10, 20], 0.5) == pytest.approx(15.0)

    def test_single_element_returns_itself(self):
        assert _percentile([42.0], 0.0) == pytest.approx(42.0)
        assert _percentile([42.0], 0.5) == pytest.approx(42.0)
        assert _percentile([42.0], 1.0) == pytest.approx(42.0)

    def test_empty_returns_zero(self):
        assert _percentile([], 0.5) == pytest.approx(0.0)

    def test_unsorted_input_is_sorted_first(self):
        # [50,10,30] should be treated as [10,30,50]
        assert _percentile([50, 10, 30], 0.0) == pytest.approx(10.0)
        assert _percentile([50, 10, 30], 1.0) == pytest.approx(50.0)

    def test_profile_p20_p80_match_percentile_formula(self, monkeypatch):
        """The p20/p80 stored in ProfileResult must equal _percentile() applied to clean_latencies."""
        monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda *a, **kw: 1024)
        monkeypatch.setattr("kerndiff.profiler.query_clock_telemetry", lambda *a, **kw: {
            "current_sm_clock_mhz": 1800, "current_mem_clock_mhz": 2000,
            "max_sm_clock_mhz": 1800, "max_mem_clock_mhz": 2000,
            "throttle_reasons": [],
        })
        monkeypatch.setattr("kerndiff.profiler._find_ncu", lambda: None)
        lats = [100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0, 170.0, 180.0, 190.0]
        result = profile(binary="/fake", kernel_name="k",
                         max_runs=50, min_runs=10, noise_threshold=1.0,
                         warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
                         mock=False, pre_collected_latencies=lats)
        clean = result.clean_latencies_us
        assert result.p20_latency_us == pytest.approx(_percentile(sorted(clean), 0.2))
        assert result.p80_latency_us == pytest.approx(_percentile(sorted(clean), 0.8))
        assert result.median_latency_us == pytest.approx(statistics.median(clean))


# ─────────────────────────────────────────────────────────────────────────────
# 6. CV formula precision
# ─────────────────────────────────────────────────────────────────────────────

class TestCVPrecision:
    """CV = (stdev / mean) * 100 — must match Python statistics exactly."""

    def test_cv_known_values(self):
        vals = [90.0, 100.0, 110.0]
        expected = (statistics.stdev(vals) / statistics.mean(vals)) * 100.0
        assert _compute_cv(vals) == pytest.approx(expected)

    def test_cv_tight_distribution(self):
        vals = [100.0, 100.1, 99.9, 100.05, 99.95]
        expected = (statistics.stdev(vals) / statistics.mean(vals)) * 100.0
        assert _compute_cv(vals) == pytest.approx(expected, rel=1e-9)

    def test_cv_high_variance(self):
        vals = [10.0, 100.0, 1000.0]
        expected = (statistics.stdev(vals) / statistics.mean(vals)) * 100.0
        assert _compute_cv(vals) == pytest.approx(expected)

    def test_cv_two_equal_values_is_zero(self):
        assert _compute_cv([50.0, 50.0]) == pytest.approx(0.0)

    def test_cv_in_profile_result_matches_clean_latencies(self, monkeypatch):
        """The cv_pct stored in ProfileResult must equal _compute_cv(clean_latencies)."""
        monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda *a, **kw: 1024)
        monkeypatch.setattr("kerndiff.profiler.query_clock_telemetry", lambda *a, **kw: {
            "current_sm_clock_mhz": 1800, "current_mem_clock_mhz": 2000,
            "max_sm_clock_mhz": 1800, "max_mem_clock_mhz": 2000,
            "throttle_reasons": [],
        })
        monkeypatch.setattr("kerndiff.profiler._find_ncu", lambda: None)
        lats = [100.0 + i * 0.5 for i in range(20)]
        result = profile(binary="/fake", kernel_name="k",
                         max_runs=50, min_runs=20, noise_threshold=0.0,
                         warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
                         mock=False, pre_collected_latencies=lats)
        clean = result.clean_latencies_us
        expected_cv = (statistics.stdev(clean) / statistics.mean(clean)) * 100.0
        assert result.cv_pct == pytest.approx(expected_cv, rel=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# 7. Outlier removal — exact threshold
# ─────────────────────────────────────────────────────────────────────────────

class TestOutlierRemovalThreshold:
    """Threshold is EXACTLY 2×median. Values AT 2×median are kept."""

    def test_value_exactly_2x_median_is_kept(self):
        # median([100,101,102,103,104]) = 102; 2×102 = 204
        samples = [100.0, 101.0, 102.0, 103.0, 104.0, 204.0]
        clean, n = _remove_outliers(samples)
        assert 204.0 in clean
        assert n == 0

    def test_value_just_above_2x_median_is_removed(self):
        # The median is computed on the FULL list including the outlier.
        # [100,100,100,100,100,201]: median of 6 values = (100+100)/2 = 100
        # threshold = 2×100 = 200; 201 > 200 → removed
        # Safety check: removing 1 of 6 is safe (1 < 3)
        samples = [100.0, 100.0, 100.0, 100.0, 100.0, 201.0]
        clean, n = _remove_outliers(samples)
        assert 201.0 not in clean
        assert n == 1

    def test_min_latency_from_clean_not_all(self, monkeypatch):
        """min_latency_us in ProfileResult must come from clean_latencies, not all_latencies."""
        monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda *a, **kw: 1024)
        monkeypatch.setattr("kerndiff.profiler.query_clock_telemetry", lambda *a, **kw: {
            "current_sm_clock_mhz": 1800, "current_mem_clock_mhz": 2000,
            "max_sm_clock_mhz": 1800, "max_mem_clock_mhz": 2000,
            "throttle_reasons": [],
        })
        monkeypatch.setattr("kerndiff.profiler._find_ncu", lambda: None)
        # One outlier at 1.0 µs (suspiciously fast — below median by far)
        # Actually outlier is ABOVE 2×median, not below
        # median([100]*9) = 100, so threshold = 200; 999.0 is removed
        lats = [100.0] * 9 + [999.0]
        result = profile(binary="/fake", kernel_name="k",
                         max_runs=50, min_runs=5, noise_threshold=0.0,
                         warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
                         mock=False, pre_collected_latencies=lats)
        # min of clean latencies = 100.0, not 999.0
        assert result.min_latency_us == pytest.approx(100.0)
        # all_latencies includes the outlier
        assert max(result.all_latencies_us) == pytest.approx(999.0)


# ─────────────────────────────────────────────────────────────────────────────
# 8. Roofline model math
# ─────────────────────────────────────────────────────────────────────────────

class TestRooflineMath:
    """
    bw_util = dram_bw_gbs / peak_bw
    compute_util = sm_throughput_pct / 100
    bound = "memory" if bw_util > compute_util else "compute"
    headroom_pct = 100 * (1 - max(bw_util, compute_util))
    """

    def test_a100_bw_utilization_exact(self):
        # A100 peak=2000 GB/s; dram_bw=1000 → bw_util=0.5
        r = compute_roofline("NVIDIA A100 SXM4 80GB", dram_bw_gbs=1000.0, sm_throughput_pct=60.0)
        assert r.bw_utilization == pytest.approx(0.5)

    def test_a100_compute_utilization_exact(self):
        # sm_throughput=60% → compute_util=0.60
        r = compute_roofline("NVIDIA A100 SXM4 80GB", dram_bw_gbs=1000.0, sm_throughput_pct=60.0)
        assert r.compute_utilization == pytest.approx(0.60)

    def test_memory_bound_when_bw_util_higher(self):
        # bw_util=0.7, compute_util=0.6 → memory bound
        r = compute_roofline("NVIDIA A100 SXM4 80GB", dram_bw_gbs=1400.0, sm_throughput_pct=60.0)
        assert r.bound == "memory"

    def test_compute_bound_when_sm_higher(self):
        # bw_util=0.5, compute_util=0.8 → compute bound
        r = compute_roofline("NVIDIA A100 SXM4 80GB", dram_bw_gbs=1000.0, sm_throughput_pct=80.0)
        assert r.bound == "compute"

    def test_headroom_pct_memory_bound(self):
        # bw_util=0.7, compute_util=0.6 → headroom = (1-0.7)*100 = 30%
        r = compute_roofline("NVIDIA A100 SXM4 80GB", dram_bw_gbs=1400.0, sm_throughput_pct=60.0)
        assert r.headroom_pct == pytest.approx(30.0)

    def test_headroom_pct_compute_bound(self):
        # bw_util=0.5, compute_util=0.8 → headroom = (1-0.8)*100 = 20%
        r = compute_roofline("NVIDIA A100 SXM4 80GB", dram_bw_gbs=1000.0, sm_throughput_pct=80.0)
        assert r.headroom_pct == pytest.approx(20.0)

    def test_h100_peak_bw_from_table(self):
        r = compute_roofline("NVIDIA H100 SXM5 80GB", dram_bw_gbs=3000.0, sm_throughput_pct=50.0)
        # H100 SXM5 peak = 3350 GB/s
        assert r.peak_bw_gbs == pytest.approx(3350.0)
        assert r.bw_utilization == pytest.approx(3000.0 / 3350.0)

    def test_nvml_overrides_table(self):
        # NVML reports 2500 GB/s, table says 3350 → use NVML
        r = compute_roofline("NVIDIA H100 SXM5 80GB", dram_bw_gbs=1000.0,
                             sm_throughput_pct=50.0, nvml_peak_bw=2500.0)
        assert r.peak_bw_gbs == pytest.approx(2500.0)
        assert r.bw_utilization == pytest.approx(1000.0 / 2500.0)
        assert r.bw_source == "nvml"

    def test_unknown_gpu_returns_unmatched(self):
        r = compute_roofline("Mystery GPU XYZ 123", dram_bw_gbs=500.0, sm_throughput_pct=70.0)
        assert r.gpu_matched is False
        assert r.bound == "unknown"

    def test_zero_dram_bw_is_compute_bound(self):
        # bw_util=0, compute_util=0.5 → compute bound
        r = compute_roofline("NVIDIA A100 SXM4 80GB", dram_bw_gbs=0.0, sm_throughput_pct=50.0)
        assert r.bound == "compute"

    def test_headroom_cannot_exceed_100(self):
        # All perfectly efficient → headroom=0
        r = compute_roofline("NVIDIA A100 SXM4 80GB", dram_bw_gbs=2000.0, sm_throughput_pct=100.0)
        assert r.headroom_pct == pytest.approx(0.0)


# ─────────────────────────────────────────────────────────────────────────────
# 9. Pipeline CSV parsing — averaging vs summing
# ─────────────────────────────────────────────────────────────────────────────

class TestPipelineParsing:
    """
    Pipeline mode: rate metrics (%, GB/s, int) → averaged across launches.
    Count/timing metrics (us, count) → summed across launches.
    """

    def _two_launch_csv(self, metric_name, unit, val1, val2):
        return (
            '"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",'
            '"Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"\n'
            f'"0","1","./b","h","k1","t","1","7","S","{metric_name}","{unit}","{val1}"\n'
            f'"0","1","./b","h","k2","t","1","7","S","{metric_name}","{unit}","{val2}"\n'
        )

    def test_latency_us_is_summed(self):
        # Two launches: 100µs and 200µs → sum=300µs
        csv = self._two_launch_csv("gpu__time_duration.sum", "nsecond", "100000", "200000")
        m = parse_ncu_csv_pipeline(csv, 2)
        assert m["latency_us"] == pytest.approx(300.0)

    def test_sm_throughput_pct_is_averaged(self):
        # 60% and 80% → average=70%
        csv = self._two_launch_csv(
            "sm__throughput.avg.pct_of_peak_sustained_elapsed", "percent", "60.0", "80.0"
        )
        m = parse_ncu_csv_pipeline(csv, 2)
        assert m["sm_throughput"] == pytest.approx(70.0)

    def test_dram_bw_is_averaged(self):
        # 400 and 600 GB/s → average=500 GB/s
        val1_bytes = str(400 * 10**9)
        val2_bytes = str(600 * 10**9)
        csv = self._two_launch_csv("dram__bytes.sum.per_second", "byte/second", val1_bytes, val2_bytes)
        m = parse_ncu_csv_pipeline(csv, 2)
        assert m["dram_bw_gbs"] == pytest.approx(500.0)

    def test_bank_conflicts_count_is_summed(self):
        # count unit → summed
        csv = self._two_launch_csv(
            "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum", "", "1000", "2000"
        )
        m = parse_ncu_csv_pipeline(csv, 2)
        assert m["l1_bank_conflicts"] == pytest.approx(3000.0)

    def test_registers_per_thread_int_unit_is_averaged(self):
        # "int" unit → averaged
        csv = self._two_launch_csv("launch__registers_per_thread", "register/thread", "64", "96")
        m = parse_ncu_csv_pipeline(csv, 2)
        assert m["registers_per_thread"] == pytest.approx(80.0)


# ─────────────────────────────────────────────────────────────────────────────
# 10. Interleaved timing — binary assignment correctness
# ─────────────────────────────────────────────────────────────────────────────

class TestInterleavedAssignment:
    """
    Critical invariant: latencies_a must contain values from binary_a ONLY,
    and latencies_b must contain values from binary_b ONLY.
    A bug here would silently swap v1/v2 latencies in some pairs,
    making every comparison wrong.
    """

    def test_binary_a_values_go_to_latencies_a(self, monkeypatch):
        # binary_a always returns 111.0, binary_b always returns 222.0
        monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda *a, **kw: 1024)
        monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *a, **kw: None)

        def _fake_run(cmd, *a, **kw):
            if cmd[0] == "/bin/a":
                return subprocess.CompletedProcess(cmd, 0, stdout="111.0\n", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="222.0\n", stderr="")

        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _fake_run)
        la, lb, _ = interleave_timing(
            "/bin/a", "/bin/b", "k",
            min_runs=20, max_runs=20, noise_threshold=0.0,
            warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
        )
        # All a-latencies must be 111.0
        assert all(v == pytest.approx(111.0) for v in la), \
            f"binary_a results leaked into wrong list: {la}"
        # All b-latencies must be 222.0
        assert all(v == pytest.approx(222.0) for v in lb), \
            f"binary_b results leaked into wrong list: {lb}"

    def test_no_cross_contamination_with_random_ordering(self, monkeypatch):
        """Even when order is randomized, values must land in the right list."""
        monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda *a, **kw: 1024)
        monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *a, **kw: None)

        # Use very different latency ranges so any mix-up is obvious
        def _fake_run(cmd, *a, **kw):
            if cmd[0] == "/bin/fast":
                return subprocess.CompletedProcess(cmd, 0, stdout="1.0\n", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="9999.0\n", stderr="")

        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _fake_run)
        la, lb, _ = interleave_timing(
            "/bin/fast", "/bin/slow", "k",
            min_runs=30, max_runs=30, noise_threshold=0.0,
            warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
        )
        # If any swap happened, we'd see 9999.0 in la or 1.0 in lb
        assert max(la) == pytest.approx(1.0), "slow values leaked into fast list"
        assert min(lb) == pytest.approx(9999.0), "fast values leaked into slow list"

    def test_interleaved_min_latency_correctness(self, monkeypatch):
        """
        After interleaved timing, the ProfileResult min_latency_us for each variant
        must correspond to the correct binary's measurements.
        """
        monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda *a, **kw: 1024)
        monkeypatch.setattr("kerndiff.profiler.query_clock_telemetry", lambda *a, **kw: {
            "current_sm_clock_mhz": 1800, "current_mem_clock_mhz": 2000,
            "max_sm_clock_mhz": 1800, "max_mem_clock_mhz": 2000,
            "throttle_reasons": [],
        })
        monkeypatch.setattr("kerndiff.profiler._find_ncu", lambda: None)

        def _fake_run(cmd, *a, **kw):
            if cmd[0] == "/bin/fast":
                return subprocess.CompletedProcess(cmd, 0, stdout="50.0\n", stderr="")
            if cmd[0] == "/bin/slow":
                return subprocess.CompletedProcess(cmd, 0, stdout="200.0\n", stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _fake_run)

        la, lb, _ = interleave_timing(
            "/bin/fast", "/bin/slow", "k",
            min_runs=10, max_runs=10, noise_threshold=0.0,
            warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
        )
        # Verify the latencies themselves before profile() processes them
        assert all(v == pytest.approx(50.0) for v in la)
        assert all(v == pytest.approx(200.0) for v in lb)

        r_fast = profile(binary="/bin/fast", kernel_name="k",
                         max_runs=10, min_runs=10, noise_threshold=0.0,
                         warmup=0, gpu_id=0, hardware=MOCK_HARDWARE, mock=False,
                         pre_collected_latencies=la)
        r_slow = profile(binary="/bin/slow", kernel_name="k",
                         max_runs=10, min_runs=10, noise_threshold=0.0,
                         warmup=0, gpu_id=0, hardware=MOCK_HARDWARE, mock=False,
                         pre_collected_latencies=lb)

        assert r_fast.min_latency_us == pytest.approx(50.0)
        assert r_slow.min_latency_us == pytest.approx(200.0)

        # The speedup computed from these must be exactly 4x
        v = compute_verdict(r_slow, r_fast)
        assert v["speedup"] == pytest.approx(4.0)


# ─────────────────────────────────────────────────────────────────────────────
# 11. NCU cross-validation threshold precision
# ─────────────────────────────────────────────────────────────────────────────

class TestCrossValidationThreshold:
    """
    Warn when ncu_latency / measured_latency is outside [0.8, 1.25].
    These are exact boundaries — off-by-one in the comparison would
    either suppress real warnings or emit false ones.
    """

    def _make_ncu_csv(self, ncu_us):
        return (
            '"ID","PID","Proc","Host","Kernel","Time","Ctx","Stream","Sec",'
            '"Metric Name","Metric Unit","Metric Value"\n'
            f'"0","1","b","h","k","t","1","7","S",'
            f'"gpu__time_duration.sum","nsecond","{int(ncu_us * 1000)}"\n'
        )

    def _run(self, monkeypatch, measured_us, ncu_us):
        monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda *a, **kw: 1024)
        monkeypatch.setattr("kerndiff.profiler.query_clock_telemetry", lambda *a, **kw: {
            "current_sm_clock_mhz": 1800, "current_mem_clock_mhz": 2000,
            "max_sm_clock_mhz": 1800, "max_mem_clock_mhz": 2000,
            "throttle_reasons": [],
        })
        monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *a, **kw: None)
        monkeypatch.setattr("kerndiff.profiler._run_timed_legacy", lambda *a, **kw: [measured_us] * 10)
        monkeypatch.setattr("kerndiff.profiler._find_ncu", lambda: "/ncu")
        monkeypatch.setattr("kerndiff.profiler.shutil.which", lambda cmd: None)
        csv = self._make_ncu_csv(ncu_us)
        def _fake_run(cmd, *a, **kw):
            if "/ncu" in cmd[0]:
                return subprocess.CompletedProcess(cmd, 0, stdout=csv, stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _fake_run)
        return profile(binary="/fake", kernel_name="k",
                       max_runs=10, min_runs=10, noise_threshold=0.0,
                       warmup=0, gpu_id=0, hardware=MOCK_HARDWARE, mock=False)

    def test_ratio_exactly_0_8_no_warning(self, monkeypatch):
        # ncu=80µs, measured=100µs → ratio=0.8 exactly → no warning
        result = self._run(monkeypatch, 100.0, 80.0)
        assert not any("NCU-reported kernel duration" in w for w in result.warnings)

    def test_ratio_0_799_warns(self, monkeypatch):
        # ncu=79.9µs, measured=100µs → ratio=0.799 < 0.8 → warn
        result = self._run(monkeypatch, 100.0, 79.9)
        assert any("NCU-reported kernel duration" in w for w in result.warnings)

    def test_ratio_exactly_1_25_no_warning(self, monkeypatch):
        # ncu=125µs, measured=100µs → ratio=1.25 exactly → no warning
        result = self._run(monkeypatch, 100.0, 125.0)
        assert not any("NCU-reported kernel duration" in w for w in result.warnings)

    def test_ratio_1_251_warns(self, monkeypatch):
        # ncu=125.1µs, measured=100µs → ratio=1.251 > 1.25 → warn
        result = self._run(monkeypatch, 100.0, 125.1)
        assert any("NCU-reported kernel duration" in w for w in result.warnings)

    def test_warning_text_contains_both_values(self, monkeypatch):
        result = self._run(monkeypatch, 100.0, 50.0)
        w = next(w for w in result.warnings if "NCU-reported kernel duration" in w)
        # Warning must name both the NCU value and measured value
        assert "50.0" in w
        assert "100.0" in w


# ─────────────────────────────────────────────────────────────────────────────
# 12. Noise floor — constants and their effect on displayed numbers
# ─────────────────────────────────────────────────────────────────────────────

class TestNoiseFloorValues:
    """
    NOISE_FLOOR_LOCKED  = 0.02  → 2% threshold
    NOISE_FLOOR_UNLOCKED= 0.05  → 5% threshold
    These constants directly determine what gets shown as "~" vs "+"/"-".
    Wrong values would make real improvements invisible or noise look significant.
    """

    def test_locked_constant_value(self):
        assert NOISE_FLOOR_LOCKED == 0.02

    def test_unlocked_constant_value(self):
        assert NOISE_FLOOR_UNLOCKED == 0.05

    def test_locked_threshold_exactly_2_pct(self):
        # With locked floor, exactly 2.0% delta should NOT be noise
        d = compute_delta(METRICS_BY_KEY["latency_us"], 100.0, 98.0,
                          noise_floor=NOISE_FLOOR_LOCKED)
        assert d.symbol != "~"

    def test_locked_threshold_1_99_pct_is_noise(self):
        d = compute_delta(METRICS_BY_KEY["latency_us"], 100.0, 98.01,
                          noise_floor=NOISE_FLOOR_LOCKED)
        assert d.symbol == "~"

    def test_unlocked_threshold_exactly_5_pct(self):
        # With unlocked floor, exactly 5.0% delta should NOT be noise
        d = compute_delta(METRICS_BY_KEY["latency_us"], 100.0, 95.0,
                          noise_floor=NOISE_FLOOR_UNLOCKED)
        assert d.symbol != "~"

    def test_unlocked_threshold_4_99_pct_is_noise(self):
        d = compute_delta(METRICS_BY_KEY["latency_us"], 100.0, 95.01,
                          noise_floor=NOISE_FLOOR_UNLOCKED)
        assert d.symbol == "~"
