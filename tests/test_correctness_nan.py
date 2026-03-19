"""Test NaN handling in correctness checking."""
import math
from kerndiff.compiler import _safe_diff


def test_both_nan_is_zero():
    assert _safe_diff(float("nan"), float("nan")) == 0.0


def test_one_nan_is_inf():
    assert _safe_diff(float("nan"), 1.0) == float("inf")
    assert _safe_diff(1.0, float("nan")) == float("inf")


def test_both_pos_inf_is_zero():
    assert _safe_diff(float("inf"), float("inf")) == 0.0


def test_both_neg_inf_is_zero():
    assert _safe_diff(float("-inf"), float("-inf")) == 0.0


def test_pos_neg_inf_is_large():
    result = _safe_diff(float("inf"), float("-inf"))
    assert result == float("inf") or result > 1e30


def test_normal_values():
    assert abs(_safe_diff(1.0, 1.5) - 0.5) < 1e-10


def test_zero_diff():
    assert _safe_diff(3.14, 3.14) == 0.0
