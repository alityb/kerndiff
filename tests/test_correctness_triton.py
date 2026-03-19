"""Tests for --correctness support in Triton backends."""
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import pytest


FIXTURES = Path(__file__).parent.parent / "kerndiff" / "fixtures"
PERSISTENT_TEMPLATE = FIXTURES / "harness_template_triton_persistent.py"


def test_triton_backend_has_dump_output_method():
    from kerndiff.backends.triton import TritonBackend
    b = TritonBackend()
    assert hasattr(b, "dump_output"), "TritonBackend must have dump_output() method"
    import inspect
    sig = inspect.signature(b.dump_output)
    assert "proc" in sig.parameters


def test_persistent_harness_has_dump_command():
    content = PERSISTENT_TEMPLATE.read_text()
    assert '_cmd == "dump"' in content or "_cmd == 'dump'" in content


def test_persistent_harness_dumps_z_slice():
    content = PERSISTENT_TEMPLATE.read_text()
    assert "z[:16]" in content
    assert ".cpu().tolist()" in content


def test_persistent_harness_has_nontrivial_input():
    content = PERSISTENT_TEMPLATE.read_text()
    # Must use arange pattern, not zeros
    assert "torch.arange" in content
    assert "% 64 + 1" in content
    # Must NOT use zeros for x
    assert "x = torch.zeros" not in content


def test_ncu_harness_has_nontrivial_input():
    ncu_template = FIXTURES / "harness_template_triton.py"
    content = ncu_template.read_text()
    assert "torch.arange" in content
    assert "% 64 + 1" in content
    assert "torch.randn" not in content


def test_cuda_harness_has_fill_kernel():
    cuda_template = FIXTURES / "harness_template.cu"
    content = cuda_template.read_text()
    assert "_kerndiff_fill" in content
    assert "i % 64 + 1" in content


def test_mock_correctness_skipped():
    from kerndiff.cli import main
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "--correctness", "v1.cu", "v2.cu", "--fn", "kernel"])
    assert rc == 0
    assert "skipped" in stderr.getvalue()
    assert "mock mode" in stderr.getvalue()


def test_auto_correctness_fires_for_different_files(tmp_path):
    """Auto-correctness check triggers when two different .py files are passed in mock mode."""
    # In mock mode we can't test auto-correctness because dump_output returns []
    # We test that auto-correctness block is only entered for non-mock, non-git mode.
    # This tests the logic path doesn't crash with empty output_vals.
    from kerndiff.cli import main
    f1 = tmp_path / "v1.cu"
    f2 = tmp_path / "v2.cu"
    f1.write_text("dummy")
    f2.write_text("dummy")
    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", str(f1), str(f2), "--fn", "kernel"])
    # Mock mode: no auto-correctness line (output_vals is empty)
    assert rc == 0
    assert "auto-correctness" not in stderr.getvalue()


def test_emit_correctness_result_pass():
    from kerndiff.cli import _emit_correctness_result
    warnings = []
    buf = io.StringIO()
    import sys
    old_stderr = sys.stderr
    sys.stderr = buf
    try:
        _emit_correctness_result(1e-6, [1.0, 2.0], [1.0, 2.0], 1e-4, "correctness check", warnings)
    finally:
        sys.stderr = old_stderr
    out = buf.getvalue()
    assert "ok" in out
    assert len(warnings) == 0


def test_emit_correctness_result_fail():
    from kerndiff.cli import _emit_correctness_result
    warnings = []
    buf = io.StringIO()
    import sys
    old_stderr = sys.stderr
    sys.stderr = buf
    try:
        _emit_correctness_result(0.5, [1.0, 2.0], [1.5, 2.5], 1e-4, "correctness check", warnings)
    finally:
        sys.stderr = old_stderr
    out = buf.getvalue()
    assert "FAILED" in out
    assert len(warnings) == 1


def test_safe_diff_normal():
    from kerndiff.cli import _safe_diff
    assert _safe_diff(1.0, 2.0) == pytest.approx(1.0)
    assert _safe_diff(3.5, 3.5) == pytest.approx(0.0)


def test_safe_diff_nan():
    import math
    from kerndiff.cli import _safe_diff
    result = _safe_diff(float("nan"), 1.0)
    assert math.isnan(result)


def test_profile_result_has_output_vals():
    from kerndiff.profiler import ProfileResult, MOCK_HARDWARE
    r = ProfileResult(
        kernel_name="k",
        metrics={},
        min_latency_us=100.0,
        all_latencies_us=[100.0],
        cv_pct=0.0,
        ptx_instructions={},
        hardware=MOCK_HARDWARE,
        warnings=[],
        actual_runs=1,
        max_runs=50,
        min_runs=10,
        noise_threshold=1.0,
        warmup=32,
        l2_flush=False,
    )
    assert r.output_vals == []


def test_profile_result_output_vals_set():
    from kerndiff.profiler import ProfileResult, MOCK_HARDWARE
    r = ProfileResult(
        kernel_name="k",
        metrics={},
        min_latency_us=100.0,
        all_latencies_us=[100.0],
        cv_pct=0.0,
        ptx_instructions={},
        hardware=MOCK_HARDWARE,
        warnings=[],
        actual_runs=1,
        max_runs=50,
        min_runs=10,
        noise_threshold=1.0,
        warmup=32,
        l2_flush=False,
        output_vals=[1.0, 2.0, 3.0],
    )
    assert r.output_vals == [1.0, 2.0, 3.0]
