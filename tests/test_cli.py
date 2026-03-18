import io
from contextlib import redirect_stderr, redirect_stdout

import pytest

from kerndiff.cli import main


SINGLE_KERNEL = """
__global__ void chunked_scan_kernel(float* a, float* b, float* c, int n) {}
"""

MULTI_KERNEL = """
__global__ void chunked_scan_kernel(float* a, float* b, float* c, int n) {}
__global__ void prefill_kernel(float* a, float* b, float* c, int n) {}
"""


def test_fn_auto_detect_single_matching_kernel(tmp_path):
    file_a = tmp_path / "v1.cu"
    file_b = tmp_path / "v2.cu"
    file_a.write_text(SINGLE_KERNEL)
    file_b.write_text(SINGLE_KERNEL)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", str(file_a), str(file_b)])
    assert rc == 0
    assert "could not auto-detect" not in stderr.getvalue()


def test_fn_auto_detect_error_lists_kernels(tmp_path):
    file_a = tmp_path / "v1.cu"
    file_b = tmp_path / "v2.cu"
    file_a.write_text(MULTI_KERNEL)
    file_b.write_text(SINGLE_KERNEL)
    with pytest.raises(SystemExit) as exc:
        main(["--mock", str(file_a), str(file_b)])
    message = str(exc.value)
    assert "error: could not auto-detect kernel" in message
    assert "prefill_kernel" in message
    assert "chunked_scan_kernel" in message


def test_single_file_mock_mode_skips_git(tmp_path):
    file_a = tmp_path / "kernel.cu"
    file_a.write_text(SINGLE_KERNEL)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", str(file_a), "--fn", "chunked_scan_kernel"])
    assert rc == 0
    assert "mock mode -- no GPU required." in stderr.getvalue()
