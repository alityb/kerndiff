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


def test_all_and_fn_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        main(["--mock", "v1.cu", "v2.cu", "--fn", "kernel", "--all"])
    assert "mutually exclusive" in str(exc.value)


def test_all_flag_profiles_common_kernels(tmp_path):
    file_a = tmp_path / "v1.cu"
    file_b = tmp_path / "v2.cu"
    file_a.write_text(MULTI_KERNEL)
    file_b.write_text(MULTI_KERNEL)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", str(file_a), str(file_b), "--all"])
    assert rc == 0
    output = stdout.getvalue()
    assert "chunked_scan_kernel" in output
    assert "prefill_kernel" in output


def test_all_flag_skips_non_common_kernels(tmp_path):
    file_a = tmp_path / "v1.cu"
    file_b = tmp_path / "v2.cu"
    file_a.write_text(MULTI_KERNEL)
    file_b.write_text(SINGLE_KERNEL)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", str(file_a), str(file_b), "--all"])
    assert rc == 0
    err = stderr.getvalue()
    assert "skipping prefill_kernel" in err
    # Only one common kernel, so output is a diff (no header needed)
    assert "latency" in stdout.getvalue()


def test_dtype_flag_accepted():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel", "--dtype", "half"])
    assert rc == 0


def test_dtype_flag_invalid():
    with pytest.raises(SystemExit):
        main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel", "--dtype", "double"])
