from kerndiff.diff import compute_all_deltas, compute_delta, compute_derived_metrics, compute_verdict, sort_deltas
from kerndiff.metrics import METRICS_BY_KEY

from conftest import make_result


def test_lower_is_better_decrease_is_positive_signal():
    delta = compute_delta(METRICS_BY_KEY["latency_us"], 100, 80)
    assert delta.symbol == "++"


def test_lower_is_better_increase_is_negative_signal():
    delta = compute_delta(METRICS_BY_KEY["latency_us"], 100, 120)
    assert delta.symbol == "--"


def test_higher_is_better_increase_is_positive_signal():
    delta = compute_delta(METRICS_BY_KEY["l2_hit_rate"], 100, 120)
    assert delta.symbol == "++"


def test_small_delta_is_noise():
    delta = compute_delta(METRICS_BY_KEY["l2_hit_rate"], 100, 101.5)
    assert delta.symbol == "~"


def test_exactly_fifteen_percent_favorable_is_double_plus():
    delta = compute_delta(METRICS_BY_KEY["latency_us"], 100, 85)
    assert delta.symbol == "++"


def test_sort_deltas_latency_first_and_noise_last():
    v1 = {"latency_us": 100, "l2_hit_rate": 50, "stall_memqueue": 3.0}
    v2 = {"latency_us": 90, "l2_hit_rate": 60, "stall_memqueue": 3.03}
    deltas = sort_deltas(compute_all_deltas(v1, v2))
    assert deltas[0].metric.key == "latency_us"
    noisy_seen = False
    for delta in deltas[1:]:
        if delta.symbol == "~":
            noisy_seen = True
        assert not noisy_seen or delta.symbol == "~"


def test_compute_verdict_returns_required_keys():
    verdict = compute_verdict(make_result(247.0), make_result(189.1))
    for key in [
        "speedup",
        "direction",
        "label",
        "v1_latency_us",
        "v2_latency_us",
        "v1_min_us",
        "v1_max_us",
        "v1_cv_pct",
        "v2_min_us",
        "v2_max_us",
        "v2_cv_pct",
    ]:
        assert key in verdict


def test_compute_verdict_improvement():
    verdict = compute_verdict(make_result(247.0), make_result(189.1))
    assert verdict["direction"] == "improvement"
    assert verdict["speedup"] > 1


def test_compute_verdict_unchanged():
    verdict = compute_verdict(make_result(247.0), make_result(250.952))
    assert verdict["direction"] == "unchanged"


def test_sm_imbalance_derived_metric():
    """sm_imbalance should be computed as (sm_throughput / sm_occupancy) * 100."""
    metrics = {"sm_throughput": 50.0, "sm_occupancy": 80.0, "latency_us": 100.0}
    derived = compute_derived_metrics(metrics)
    assert "sm_imbalance" in derived
    assert abs(derived["sm_imbalance"] - 62.5) < 0.01


def test_sm_imbalance_not_computed_when_occupancy_zero():
    metrics = {"sm_throughput": 50.0, "sm_occupancy": 0.0, "latency_us": 100.0}
    derived = compute_derived_metrics(metrics)
    assert "sm_imbalance" not in derived
