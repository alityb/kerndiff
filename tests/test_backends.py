import pytest

from kerndiff.backends import dispatch
from kerndiff.backends.cuda import CUDABackend
from kerndiff.backends.triton import TritonBackend, parse_triton_kernels


def test_dispatch_cuda():
    b = dispatch("kernel.cu")
    assert isinstance(b, CUDABackend)


def test_dispatch_triton():
    b = dispatch("kernel.py")
    assert isinstance(b, TritonBackend)


def test_dispatch_unsupported():
    with pytest.raises(SystemExit) as exc:
        dispatch("kernel.rs")
    assert "unsupported file type" in str(exc.value)


def test_parse_triton_kernels_single():
    source = "@triton.jit\ndef prefix_scan(a_ptr, b_ptr, c_ptr, n: tl.constexpr):\n    pass\n"
    names = parse_triton_kernels(source)
    assert names == ["prefix_scan"]


def test_parse_triton_kernels_multiple():
    source = (
        "@triton.jit\ndef kernel_a(x):\n    pass\n\n"
        "@triton.jit\ndef kernel_b(x):\n    pass\n"
    )
    names = parse_triton_kernels(source)
    assert names == ["kernel_a", "kernel_b"]


def test_parse_triton_kernels_none():
    source = "def regular_function():\n    pass\n"
    names = parse_triton_kernels(source)
    assert names == []


def test_triton_compile_generates_harness(tmp_path):
    """Test that TritonBackend.compile() generates a runnable harness."""
    source = tmp_path / "kernel.py"
    source.write_text(
        "import triton\nimport triton.language as tl\n\n"
        "@triton.jit\ndef my_kernel(a_ptr, b_ptr, c_ptr, n: tl.constexpr, BLOCK_SIZE: tl.constexpr):\n"
        "    pid = tl.program_id(0)\n"
        "    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)\n"
        "    mask = offsets < n\n"
        "    x = tl.load(a_ptr + offsets, mask=mask, other=0.0)\n"
        "    tl.store(c_ptr + offsets, x, mask=mask)\n"
    )
    b = TritonBackend()
    harness_path = b.compile(
        str(source), "my_kernel", "sm_86", "float", 1024, None,
    )
    content = open(harness_path).read()
    assert "torch" in content
    assert "my_kernel" in content
    assert "1024" in content


def test_triton_ncu_cmd_has_kernel_regex():
    b = TritonBackend()
    cmd = b.ncu_cmd("/usr/bin/ncu", "/tmp/h.py", "prefix_scan", "metric1,metric2", 1)
    assert "--kernel-name" in cmd
    assert "regex:prefix_scan.*" in cmd
    assert "--target-processes" in cmd
    assert "--clock-control" in cmd
    assert cmd[cmd.index("--clock-control") + 1] == "none"


def test_cuda_ncu_cmd_has_skip_and_clock_none():
    b = CUDABackend()
    cmd = b.ncu_cmd("/usr/bin/ncu", "/tmp/bench", "vec_add", "m1,m2", 1)
    assert "--launch-skip" in cmd
    assert cmd[cmd.index("--launch-skip") + 1] == "2"
    assert "--clock-control" in cmd
    assert cmd[cmd.index("--clock-control") + 1] == "none"


def test_scan_kernels_triton():
    """Test _scan_kernels works for .py files."""
    from kerndiff.cli import _scan_kernels
    from pathlib import Path
    examples = Path(__file__).resolve().parent.parent / "examples"
    names = _scan_kernels(str(examples / "triton_scan_v1.py"))
    assert "prefix_scan" in names


def test_scan_kernels_cuda():
    """Test _scan_kernels still works for .cu files."""
    from kerndiff.cli import _scan_kernels
    from pathlib import Path
    examples = Path(__file__).resolve().parent.parent / "examples"
    names = _scan_kernels(str(examples / "vec_add_v1.cu"))
    assert "vec_add" in names


def test_triton_examples_have_matching_kernels():
    """Both triton scan examples should have prefix_scan."""
    from kerndiff.cli import _scan_kernels
    from pathlib import Path
    examples = Path(__file__).resolve().parent.parent / "examples"
    v1 = _scan_kernels(str(examples / "triton_scan_v1.py"))
    v2 = _scan_kernels(str(examples / "triton_scan_v2.py"))
    assert "prefix_scan" in v1
    assert "prefix_scan" in v2


def test_mock_mode_still_works_with_py_files():
    """Mock mode should work even with .py files (uses fixture CSV)."""
    import io
    from contextlib import redirect_stdout, redirect_stderr
    from kerndiff.cli import main

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "v1.py", "v2.py", "--fn", "chunked_scan_kernel"])
    assert rc == 0
