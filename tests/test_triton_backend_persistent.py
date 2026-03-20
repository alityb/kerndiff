"""Tests for persistent Triton harness timing correctness."""
import io
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

import kerndiff

import pytest

from kerndiff.runtimes.triton import TritonBackend
from kerndiff.runtimes.cuda import CUDABackend


FIXTURES = Path(kerndiff.__file__).resolve().parent / "fixtures"
PERSISTENT_TEMPLATE = FIXTURES / "harness_template_triton_persistent.py"


def test_triton_backend_is_persistent():
    b = TritonBackend()
    assert b.is_persistent() is True


def test_cuda_backend_is_not_persistent():
    b = CUDABackend()
    assert not hasattr(b, "is_persistent") or not b.is_persistent()


def test_persistent_harness_template_exists():
    assert PERSISTENT_TEMPLATE.exists()


def test_persistent_harness_template_has_ready_signal():
    content = PERSISTENT_TEMPLATE.read_text()
    assert 'print("ready"' in content


def test_persistent_harness_template_has_command_loop():
    content = PERSISTENT_TEMPLATE.read_text()
    assert 'sys.stdin.readline()' in content
    assert 'cmd == "time"' in content or "_cmd == \"time\"" in content
    assert 'cmd == "quit"' in content or "_cmd == \"quit\"" in content


def test_persistent_harness_has_sleep_before_timing():
    content = PERSISTENT_TEMPLATE.read_text()
    assert "torch.cuda._sleep" in content


def test_persistent_harness_has_l2_flush():
    content = PERSISTENT_TEMPLATE.read_text()
    assert "_flush_l2()" in content
    assert "zero_()" in content
    assert "torch.cuda.synchronize()" in content


def test_persistent_harness_prints_latency_as_float():
    content = PERSISTENT_TEMPLATE.read_text()
    # The harness should print the timing value with flush=True
    assert "flush=True" in content
    assert "_us" in content or "us" in content


def test_ncu_harness_is_separate_from_timing_harness():
    b = TritonBackend()
    assert hasattr(b, "compile_ncu")


def test_compile_generates_persistent_harness(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text(
        "import triton\nimport triton.language as tl\n\n"
        "@triton.jit\n"
        "def my_kernel(x_ptr, y_ptr, z_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
        "    mask = offs < n\n"
        "    x = tl.load(x_ptr + offs, mask=mask, other=0.0)\n"
        "    tl.store(z_ptr + offs, x, mask=mask)\n"
    )
    b = TritonBackend()
    harness_path = b.compile(str(source), "my_kernel", "sm_86", "float", 1024, None)
    content = open(harness_path).read()
    assert 'print("ready"' in content
    assert "torch.cuda._sleep" in content
    assert "my_kernel" in content
    assert "1024" in content


def test_compile_timed_bakes_in_l2_flush(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text(
        "import triton\nimport triton.language as tl\n\n"
        "@triton.jit\n"
        "def k(x_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):\n"
        "    pass\n"
    )
    b = TritonBackend()
    b.compile(str(source), "k", "sm_86", "float", 512, None)
    harness_path = b.compile_timed(
        str(source), "k", "sm_86", "float", 512, None,
        iters=1, l2_flush_bytes=6291456, warmup=8,
    )
    content = open(harness_path).read()
    assert "6291456" in content
    assert "8" in content


def test_compile_ncu_generates_single_run_harness(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text(
        "import triton\nimport triton.language as tl\n\n"
        "@triton.jit\n"
        "def k(x_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):\n"
        "    pass\n"
    )
    b = TritonBackend()
    b.compile(str(source), "k", "sm_86", "float", 512, None)
    ncu_harness_path = b.compile_ncu(str(source), "k", "sm_86", "float", 512, None)
    content = open(ncu_harness_path).read()
    # NCU harness uses old single-run template — does NOT have ready signal
    assert 'print("ready"' not in content
    assert "torch.cuda._sleep" not in content
    # But does have the kernel call
    assert "k" in content


def test_last_compile_args_stored(tmp_path):
    source = tmp_path / "kernel.py"
    source.write_text("import triton\n@triton.jit\ndef k(x): pass\n")
    b = TritonBackend()
    b.compile(str(source), "k", "sm_86", "float", 2048, None)
    assert b._last_compile_args["kernel_name"] == "k"
    assert b._last_compile_args["buf_elems"] == 2048


def test_mock_mode_skips_persistent_harness():
    """Mock mode should not spawn any persistent harness process."""
    from kerndiff.cli import main

    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel"])
    assert rc == 0


def test_mock_mode_works_with_py_files():
    """Mock mode should work even when .py files are passed."""
    from kerndiff.cli import main

    stdout, stderr = io.StringIO(), io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "v1.py", "v2.py", "--fn", "chunked_scan_kernel"])
    assert rc == 0
