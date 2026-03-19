"""Validate roofline numbers against known GPU specs."""
from kerndiff.roofline import compute_roofline, GPU_SPECS, _find_spec, RooflineResult


def test_h100_sxm_bw_is_3350():
    spec = _find_spec("NVIDIA H100 SXM5 80GB")
    assert spec is not None
    assert spec.peak_bw_gbs == 3350


def test_h100_pcie_bw_is_2000():
    spec = _find_spec("NVIDIA H100 PCIe 80GB")
    assert spec is not None
    assert abs(spec.peak_bw_gbs - 2000) < 50


def test_a10g_bw_is_600():
    spec = _find_spec("NVIDIA A10G")
    assert spec is not None
    assert abs(spec.peak_bw_gbs - 600) < 20


def test_h200_bw_is_4800():
    spec = _find_spec("NVIDIA H200 SXM")
    assert spec is not None
    assert abs(spec.peak_bw_gbs - 4800) < 100


def test_a100_sxm_bw_is_2000():
    spec = _find_spec("NVIDIA A100 SXM4 80GB")
    assert spec is not None
    assert abs(spec.peak_bw_gbs - 2000) < 50


def test_a100_pcie_bw_is_1935():
    spec = _find_spec("NVIDIA A100 PCIe 80GB")
    assert spec is not None
    assert abs(spec.peak_bw_gbs - 1935) < 50


def test_l40s_bw_is_864():
    spec = _find_spec("NVIDIA L40S")
    assert spec is not None
    assert spec.peak_bw_gbs == 864


def test_l40_bw_is_864():
    spec = _find_spec("NVIDIA L40")
    assert spec is not None
    assert spec.peak_bw_gbs == 864


def test_l40s_tflops_higher_than_l40():
    l40s = _find_spec("NVIDIA L40S")
    l40 = _find_spec("NVIDIA L40")
    assert l40s is not None and l40 is not None
    assert l40s.peak_tflops_fp16 > l40.peak_tflops_fp16


def test_v100_bw_is_900():
    spec = _find_spec("NVIDIA V100 SXM2 32GB")
    assert spec is not None
    assert abs(spec.peak_bw_gbs - 900) < 20


def test_unknown_gpu_returns_none():
    spec = _find_spec("NVIDIA GeForce GTX 1080 Ti")
    assert spec is None


def test_nvml_peak_bw_used_over_table():
    """When NVML bandwidth is provided, it takes priority over the table."""
    result = compute_roofline(
        gpu_name="NVIDIA A10G",
        dram_bw_gbs=540.0,
        sm_throughput_pct=91.0,
        nvml_peak_bw=600.0,
    )
    assert result.gpu_matched
    assert result.peak_bw_gbs == 600.0
    assert result.bw_source == "nvml"


def test_table_used_when_nvml_none():
    result = compute_roofline(
        gpu_name="NVIDIA A10G",
        dram_bw_gbs=540.0,
        sm_throughput_pct=91.0,
        nvml_peak_bw=None,
    )
    assert result.gpu_matched
    assert result.bw_source == "table"


def test_roofline_unmatched_when_no_spec_and_no_nvml():
    result = compute_roofline(
        gpu_name="NVIDIA GeForce GTX 1080 Ti",
        dram_bw_gbs=540.0,
        sm_throughput_pct=91.0,
        nvml_peak_bw=None,
    )
    assert not result.gpu_matched


def test_nvml_rescues_unknown_gpu():
    """NVML should provide roofline even for GPUs not in the table."""
    result = compute_roofline(
        gpu_name="NVIDIA GeForce GTX 1080 Ti",
        dram_bw_gbs=300.0,
        sm_throughput_pct=50.0,
        nvml_peak_bw=480.0,
    )
    assert result.gpu_matched
    assert result.bw_source == "nvml"
    assert abs(result.bw_utilization - 300.0 / 480.0) < 0.01


def test_bw_utilization_correct_with_nvml():
    result = compute_roofline(
        gpu_name="NVIDIA A10G",
        dram_bw_gbs=300.0,
        sm_throughput_pct=20.0,
        nvml_peak_bw=600.0,
    )
    assert abs(result.bw_utilization - 0.5) < 0.01
    assert result.bound == "memory"


def test_compute_bound_detection():
    result = compute_roofline(
        gpu_name="NVIDIA A10G",
        dram_bw_gbs=60.0,
        sm_throughput_pct=90.0,
        nvml_peak_bw=600.0,
    )
    assert result.bound == "compute"
    assert result.compute_utilization == 0.9


def test_roofline_result_has_new_fields():
    """RooflineResult should have peak_bw_gbs and bw_source with defaults."""
    r = RooflineResult(bound="memory", bw_utilization=0.5, compute_utilization=0.3,
                       headroom_pct=50.0, gpu_matched=True)
    assert r.peak_bw_gbs == 0.0
    assert r.bw_source == "unknown"


def test_h100_sxm_matches_before_generic_h100():
    """'H100 SXM5' should match the SXM5 entry, not the generic H100."""
    spec = _find_spec("NVIDIA H100 SXM5 80GB HBM3")
    assert spec is not None
    assert spec.peak_bw_gbs == 3350


def test_h100_pcie_does_not_get_sxm():
    """'H100 PCIe' should match PCIe entry with 2000 GB/s, not SXM 3350."""
    spec = _find_spec("NVIDIA H100 PCIe 80GB")
    assert spec is not None
    assert spec.peak_bw_gbs == 2000
