from __future__ import annotations

from dataclasses import dataclass

from kerndiff.metrics import METRICS, METRICS_BY_KEY, MetricDef

NOISE_FLOOR_LOCKED = 0.02
NOISE_FLOOR_UNLOCKED = 0.05


@dataclass
class MetricDelta:
    metric: MetricDef
    v1: float
    v2: float
    delta_pct: float
    favorable: bool
    symbol: str


def compute_delta(metric: MetricDef, v1: float, v2: float, noise_floor: float = NOISE_FLOOR_LOCKED) -> MetricDelta:
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
        if metric.key in v1_metrics and metric.key in v2_metrics:
            deltas.append(compute_delta(metric, v1_metrics[metric.key], v2_metrics[metric.key], noise_floor))
    return deltas


def sort_deltas(deltas: list[MetricDelta]) -> list[MetricDelta]:
    latency = [d for d in deltas if d.metric.key == "latency_us"]
    rest = [d for d in deltas if d.metric.key != "latency_us"]
    changed = sorted((d for d in rest if d.symbol != "~"), key=lambda d: abs(d.delta_pct), reverse=True)
    unchanged = sorted((d for d in rest if d.symbol == "~"), key=lambda d: d.metric.key)
    return latency + changed + unchanged


def compute_verdict(v1: "ProfileResult", v2: "ProfileResult", noise_floor: float = NOISE_FLOOR_LOCKED) -> dict:
    speedup = v1.min_latency_us / v2.min_latency_us if v2.min_latency_us else float("inf")
    if abs(speedup - 1.0) < noise_floor:
        direction = "unchanged"
        pct = abs((v2.min_latency_us - v1.min_latency_us) / (v1.min_latency_us or 1.0)) * 100.0
        label = f"no significant change"
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
        "v2_min_us": min(v2.all_latencies_us) if v2.all_latencies_us else v2.min_latency_us,
        "v2_max_us": max(v2.all_latencies_us) if v2.all_latencies_us else v2.min_latency_us,
        "v2_cv_pct": v2.cv_pct,
        "noise_floor_pct": noise_floor * 100.0,
        "latency_delta_pct": ((v2.min_latency_us - v1.min_latency_us) / (v1.min_latency_us or 1.0)) * 100.0,
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
    "compute_verdict",
]
