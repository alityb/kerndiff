"""Test significance ladder at exact boundary values."""
import math
import pytest
from kerndiff.diff import compute_delta, compute_verdict
from kerndiff.metrics import MetricDef

from conftest import make_result


def _metric(key="latency", lower_is_better=True, unit="%"):
    return MetricDef(
        key=key, display_name=key, ncu_metric="", unit=unit,
        group="test", lower_is_better=lower_is_better,
        format_fn=lambda v: f"{v:.1f}",
    )


m = _metric()


# Exact boundary: 15% is the threshold between + and ++
def test_exactly_at_15pct_threshold():
    d = compute_delta(m, v1=100.0, v2=85.0, noise_floor=0.02)
    assert d.symbol in ("+", "++")  # 15% exactly — boundary


def test_just_below_15pct():
    d = compute_delta(m, v1=100.0, v2=85.1, noise_floor=0.02)
    assert d.symbol == "+"


def test_just_above_15pct():
    d = compute_delta(m, v1=100.0, v2=84.9, noise_floor=0.02)
    assert d.symbol == "++"


# Noise floor boundary
def test_exactly_at_noise_floor():
    # noise_floor comparison is strict (<), so exactly 2% is NOT within noise
    d = compute_delta(m, v1=100.0, v2=98.0, noise_floor=0.02)  # exactly 2%
    assert d.symbol == "+"  # boundary is exclusive: 2% is not < 2%


def test_just_above_noise_floor():
    d = compute_delta(m, v1=100.0, v2=97.9, noise_floor=0.02)  # 2.1%
    assert d.symbol in ("+", "-")


# Zero values
def test_v1_zero():
    d = compute_delta(m, v1=0.0, v2=5.0, noise_floor=0.02)
    # Should not divide by zero
    assert d.delta_pct is not None


def test_both_zero():
    d = compute_delta(m, v1=0.0, v2=0.0, noise_floor=0.02)
    assert d.symbol == "~"


def test_v2_zero():
    d = compute_delta(m, v1=5.0, v2=0.0, noise_floor=0.02)
    assert d.symbol in ("++", "+")  # lower is better, v2 went down


# NaN and inf in metric values
def test_nan_metric():
    d = compute_delta(m, v1=float("nan"), v2=5.0, noise_floor=0.02)
    assert d.symbol == "~"


def test_inf_metric():
    d = compute_delta(m, v1=float("inf"), v2=5.0, noise_floor=0.02)
    assert d.symbol == "~"


def test_nan_v2():
    d = compute_delta(m, v1=5.0, v2=float("nan"), noise_floor=0.02)
    assert d.symbol == "~"


def test_inf_v2():
    d = compute_delta(m, v1=5.0, v2=float("inf"), noise_floor=0.02)
    assert d.symbol == "~"


# lower_is_better=None (shared_mem case)
def test_neutral_metric():
    m_neutral = _metric("shared_mem", lower_is_better=None)
    d = compute_delta(m_neutral, v1=16.0, v2=32.0, noise_floor=0.02)
    assert d.symbol == "~"


# Verdict boundary: exactly equal latencies
def test_verdict_equal_latency():
    r1 = make_result(200.0)
    r2 = make_result(200.0)
    v = compute_verdict(r1, r2, noise_floor=0.02)
    assert v["direction"] == "unchanged"
    assert "no significant" in v["label"].lower()


# Verdict: speedup exactly at noise floor
def test_verdict_at_noise_floor():
    r1 = make_result(100.0)
    r2 = make_result(98.0)
    v = compute_verdict(r1, r2, noise_floor=0.02)
    # speedup = 100/98 = 1.0204, abs(1.0204-1.0) = 0.0204 > 0.02 (strict <)
    assert v["direction"] == "improvement"

def test_verdict_within_noise_floor():
    r1 = make_result(100.0)
    r2 = make_result(99.0)
    v = compute_verdict(r1, r2, noise_floor=0.02)
    # speedup = 100/99 = 1.0101, abs(1.0101-1.0) = 0.0101 < 0.02
    assert v["direction"] == "unchanged"


# Verdict: speedup just beyond noise floor
def test_verdict_beyond_noise_floor():
    r1 = make_result(100.0)
    r2 = make_result(50.0)
    v = compute_verdict(r1, r2, noise_floor=0.02)
    assert v["direction"] == "improvement"
    assert "faster" in v["label"]


# Verdict: regression
def test_verdict_regression():
    r1 = make_result(100.0)
    r2 = make_result(200.0)
    v = compute_verdict(r1, r2, noise_floor=0.02)
    assert v["direction"] == "regression"
    assert "slower" in v["label"]
