"""Tests ensuring harness templates use non-trivial (non-zero) input buffers."""
from pathlib import Path

FIXTURES = Path(__file__).parent.parent / "kerndiff" / "fixtures"


def test_cuda_harness_fill_kernel_exists():
    content = (FIXTURES / "harness_template.cu").read_text()
    assert "__global__ void _kerndiff_fill" in content


def test_cuda_harness_fill_pattern():
    content = (FIXTURES / "harness_template.cu").read_text()
    assert "i % 64 + 1" in content


def test_cuda_harness_fill_called_for_d_a():
    content = (FIXTURES / "harness_template.cu").read_text()
    assert "_kerndiff_fill" in content
    # d_a and d_b should be filled, not just zeroed
    assert "cudaMemset(d_a" not in content
    assert "cudaMemset(d_b" not in content


def test_cuda_harness_d_c_still_zeroed():
    content = (FIXTURES / "harness_template.cu").read_text()
    # Output buffer d_c starts at zero (kernel writes into it)
    assert "cudaMemset(d_c" in content


def test_triton_persistent_harness_nontrivial_input():
    content = (FIXTURES / "harness_template_triton_persistent.py").read_text()
    assert "torch.arange" in content
    assert "% 64 + 1" in content
    assert "x = torch.zeros" not in content
    assert "y = torch.zeros" not in content


def test_triton_persistent_harness_output_z_zeroed():
    content = (FIXTURES / "harness_template_triton_persistent.py").read_text()
    # z is the output buffer and starts at zero
    assert "z = torch.zeros" in content


def test_triton_ncu_harness_nontrivial_input():
    content = (FIXTURES / "harness_template_triton.py").read_text()
    assert "torch.arange" in content
    assert "% 64 + 1" in content
    assert "torch.randn" not in content


def test_triton_ncu_harness_output_z_zeroed():
    content = (FIXTURES / "harness_template_triton.py").read_text()
    assert "z = torch.zeros" in content


def test_cuda_harness_fill_sync_after_fill():
    content = (FIXTURES / "harness_template.cu").read_text()
    # Must synchronize after fill before proceeding
    fill_idx = content.index("_kerndiff_fill")
    sync_idx = content.index("cudaDeviceSynchronize", fill_idx)
    assert sync_idx > fill_idx, "cudaDeviceSynchronize must come after _kerndiff_fill"


def test_triton_harness_x_dtype_matches():
    content = (FIXTURES / "harness_template_triton_persistent.py").read_text()
    # x must be cast to _DTYPE, not left as int64 arange
    assert "x = _idx.to(_DTYPE)" in content
    assert "y = _idx.to(_DTYPE)" in content


def test_triton_ncu_harness_x_dtype_matches():
    content = (FIXTURES / "harness_template_triton.py").read_text()
    assert "x = _idx.to(_DTYPE)" in content
    assert "y = _idx.to(_DTYPE)" in content
