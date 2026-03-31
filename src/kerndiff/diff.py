from __future__ import annotations

from dataclasses import dataclass
import math
import statistics

from kerndiff.metrics import METRICS, METRICS_BY_KEY, MetricDef

NOISE_FLOOR_LOCKED = 0.02
NOISE_FLOOR_UNLOCKED = 0.05

# Display order for metric groups
_GROUP_ORDER = {"sol": 0, "arithmetic": 1, "cache": 2, "warp_state": 3, "launch": 4}


@dataclass
class MetricDelta:
    metric: MetricDef
    v1: float
    v2: float
    delta_pct: float
    favorable: bool
    symbol: str


def compute_delta(metric: MetricDef, v1: float, v2: float, noise_floor: float = NOISE_FLOOR_LOCKED) -> MetricDelta:
    import math
    if math.isnan(v1) or math.isnan(v2) or math.isinf(v1) or math.isinf(v2):
        return MetricDelta(metric=metric, v1=v1, v2=v2, delta_pct=0.0, favorable=False, symbol="~")
    denom = abs(v1) if abs(v1) > 1e-12 else 1.0
    delta_pct = ((v2 - v1) / denom) * 100.0
    if metric.lower_is_better is None:
        return MetricDelta(metric=metric, v1=v1, v2=v2, delta_pct=delta_pct, favorable=False, symbol="~")
    favorable = delta_pct < 0 if metric.lower_is_better else delta_pct > 0
    abs_delta = abs(delta_pct)
    if abs_delta < noise_floor * 100:
        symbol = "~"
    elif favorable and abs_delta >= 15:
        symbol = "++"
    elif favorable and abs_delta >= 2:
        symbol = "+"
    elif not favorable and abs_delta >= 15:
        symbol = "--"
    elif not favorable and abs_delta >= 2:
        symbol = "-"
    else:
        symbol = "~"
    return MetricDelta(metric=metric, v1=v1, v2=v2, delta_pct=delta_pct, favorable=favorable, symbol=symbol)


def compute_all_deltas(v1_metrics: dict[str, float], v2_metrics: dict[str, float], noise_floor: float = NOISE_FLOOR_LOCKED) -> list[MetricDelta]:
    deltas: list[MetricDelta] = []
    for metric in METRICS:
        if metric.hidden:
            continue
        if metric.key in v1_metrics and metric.key in v2_metrics:
            deltas.append(compute_delta(metric, v1_metrics[metric.key], v2_metrics[metric.key], noise_floor))
    return deltas


def sort_deltas(deltas: list[MetricDelta]) -> list[MetricDelta]:
    """Sort by group order, then non-noisy before noisy within a group, then by metric definition position."""
    metric_positions = {m.key: i for i, m in enumerate(METRICS)}

    def sort_key(d: MetricDelta) -> tuple:
        group_idx = _GROUP_ORDER.get(d.metric.group, 99)
        is_noisy = 1 if d.symbol == "~" else 0
        pos = metric_positions.get(d.metric.key, 999)
        return (group_idx, is_noisy, pos)

    return sorted(deltas, key=sort_key)


def compute_derived_metrics(metrics: dict) -> dict:
    """Compute derived metrics (arith_intensity, flops_tflops) from raw NCU counters.

    Call after parse_ncu_csv() and after metrics["latency_us"] is set.
    Returns a dict of derived key → value to merge into the metrics dict.
    """
    derived = {}

    # FP32: ffma = 2 FLOPs (mul + add), fadd/fmul = 1 FLOP each
    # FP16: same rule applies
    fp32_flops = (
        2 * metrics.get("raw_ffma", 0)
        + metrics.get("raw_fadd", 0)
        + metrics.get("raw_fmul", 0)
    )
    fp16_flops = (
        2 * metrics.get("raw_hfma", 0)
        + metrics.get("raw_hadd", 0)
        + metrics.get("raw_hmul", 0)
    )
    total_flops = fp32_flops + fp16_flops

    # DRAM bytes: each sector is 32 bytes
    dram_bytes = (
        metrics.get("raw_dram_sectors_rd", 0)
        + metrics.get("raw_dram_sectors_wr", 0)
    ) * 32

    if dram_bytes > 0 and total_flops > 0:
        derived["arith_intensity"] = total_flops / dram_bytes

    latency_us = metrics.get("latency_us", 0)
    if total_flops > 0 and latency_us > 0:
        derived["flops_tflops"] = total_flops / (latency_us * 1e-6) / 1e12

    # SM imbalance proxy: ratio of sm_throughput to sm_occupancy.
    # A value near 100% means every active warp contributes evenly.
    # Low values suggest some SMs are idle while others are saturated.
    sm_throughput = metrics.get("sm_throughput", 0.0)
    sm_occupancy = metrics.get("sm_occupancy", 0.0)
    if sm_throughput > 0 and sm_occupancy > 0:
        derived["sm_imbalance"] = (sm_throughput / sm_occupancy) * 100.0

    raw_sm_cycles = metrics.get("raw_sm_cycles", 0.0)
    if raw_sm_cycles > 0 and latency_us > 0:
        derived["actual_sm_mhz"] = raw_sm_cycles / (latency_us * 1e-6) / 1e6

    return derived


def _pairwise_stats(
    latencies_a: list[float],
    latencies_b: list[float],
) -> tuple[float, float] | None:
    """Return (median_speedup, uncertainty_x) from aligned paired latency lists.

    Each pair (a_i, b_i) was measured back-to-back under the same GPU thermal
    state, so common-mode noise cancels in the ratio.  The median of the ratios
    is a robust point estimate of speedup; its sample stdev is the uncertainty.

    Returns None when lists are mismatched or too short for statistics.
    """
    if len(latencies_a) != len(latencies_b) or len(latencies_a) < 2:
        return None
    ratios = [a / b for a, b in zip(latencies_a, latencies_b) if b > 0]
    if len(ratios) < 2:
        return None
    return statistics.median(ratios), statistics.stdev(ratios)


def compute_verdict(
    v1: "ProfileResult",
    v2: "ProfileResult",
    noise_floor: float = NOISE_FLOOR_LOCKED,
    paired_latencies_a: list[float] | None = None,
    paired_latencies_b: list[float] | None = None,
) -> dict:
    paired = None
    if paired_latencies_a and paired_latencies_b:
        paired = _pairwise_stats(paired_latencies_a, paired_latencies_b)

    if paired:
        speedup, speedup_uncertainty_x = paired
        # Floor at noise_floor so zero-variance inputs don't treat float rounding
        # as a significant change, and to maintain a practical minimum threshold.
        unchanged_threshold = max(speedup_uncertainty_x, noise_floor)
        latency_delta_pct = (1.0 / speedup - 1.0) * 100.0 if speedup > 0 else 0.0
        used_pairwise = True
    else:
        speedup = v1.min_latency_us / v2.min_latency_us if v2.min_latency_us else float("inf")
        rel_err = math.sqrt((v1.cv_pct / 100.0) ** 2 + (v2.cv_pct / 100.0) ** 2)
        speedup_uncertainty_x = speedup * rel_err
        unchanged_threshold = noise_floor
        latency_delta_pct = ((v2.min_latency_us - v1.min_latency_us) / (v1.min_latency_us or 1.0)) * 100.0
        used_pairwise = False

    if abs(speedup - 1.0) <= unchanged_threshold:
        direction = "unchanged"
        label = "no significant change"
    elif speedup > 1.0:
        direction = "improvement"
        label = f"v2 is {speedup:.2f}x faster"
    else:
        direction = "regression"
        label = f"v2 is {(1.0 / speedup):.2f}x slower"
    return {
        "speedup": speedup,
        "direction": direction,
        "label": label,
        "v1_latency_us": v1.min_latency_us,
        "v2_latency_us": v2.min_latency_us,
        "v1_min_us": min(v1.all_latencies_us) if v1.all_latencies_us else v1.min_latency_us,
        "v1_max_us": max(v1.all_latencies_us) if v1.all_latencies_us else v1.min_latency_us,
        "v1_cv_pct": v1.cv_pct,
        "v1_p20_us": v1.p20_latency_us,
        "v1_p50_us": v1.median_latency_us,
        "v1_p80_us": v1.p80_latency_us,
        "v1_n_outliers": v1.n_outliers,
        "v2_min_us": min(v2.all_latencies_us) if v2.all_latencies_us else v2.min_latency_us,
        "v2_max_us": max(v2.all_latencies_us) if v2.all_latencies_us else v2.min_latency_us,
        "v2_cv_pct": v2.cv_pct,
        "v2_p20_us": v2.p20_latency_us,
        "v2_p50_us": v2.median_latency_us,
        "v2_p80_us": v2.p80_latency_us,
        "v2_n_outliers": v2.n_outliers,
        "speedup_uncertainty_x": speedup_uncertainty_x,
        "paired_uncertainty": used_pairwise,
        "noise_floor_pct": noise_floor * 100.0,
        "latency_delta_pct": latency_delta_pct,
    }


__all__ = [
    "MetricDef",
    "MetricDelta",
    "METRICS",
    "METRICS_BY_KEY",
    "NOISE_FLOOR_LOCKED",
    "NOISE_FLOOR_UNLOCKED",
    "compute_delta",
    "compute_all_deltas",
    "sort_deltas",
    "compute_derived_metrics",
    "compute_verdict",
]
