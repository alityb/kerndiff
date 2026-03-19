from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GpuSpec:
    peak_bw_gbs: float
    peak_tflops_fp16: float


# Ordered most-specific first — dict is insertion-ordered in Python 3.7+
GPU_SPECS: dict[str, GpuSpec] = {
    "H100 SXM5": GpuSpec(peak_bw_gbs=3350, peak_tflops_fp16=989),
    "H100 SXM":  GpuSpec(peak_bw_gbs=3350, peak_tflops_fp16=989),
    "H100 PCIe": GpuSpec(peak_bw_gbs=2000, peak_tflops_fp16=800),
    "H100":      GpuSpec(peak_bw_gbs=3350, peak_tflops_fp16=989),  # fallback to SXM
    "H200 SXM":  GpuSpec(peak_bw_gbs=4800, peak_tflops_fp16=989),
    "H200":      GpuSpec(peak_bw_gbs=4800, peak_tflops_fp16=989),
    "A100 SXM4": GpuSpec(peak_bw_gbs=2000, peak_tflops_fp16=312),
    "A100 SXM":  GpuSpec(peak_bw_gbs=2000, peak_tflops_fp16=312),
    "A100 PCIe": GpuSpec(peak_bw_gbs=1935, peak_tflops_fp16=312),
    "A100":      GpuSpec(peak_bw_gbs=2000, peak_tflops_fp16=312),  # fallback to SXM
    "A10G":      GpuSpec(peak_bw_gbs=600, peak_tflops_fp16=31.2),
    "A10":       GpuSpec(peak_bw_gbs=600, peak_tflops_fp16=31.2),
    "L40S":      GpuSpec(peak_bw_gbs=864, peak_tflops_fp16=362),
    "L40":       GpuSpec(peak_bw_gbs=864, peak_tflops_fp16=181),
    "RTX 4090":  GpuSpec(peak_bw_gbs=1008, peak_tflops_fp16=82.6),
    "RTX 3090":  GpuSpec(peak_bw_gbs=936, peak_tflops_fp16=35.6),
    "RTX 3080":  GpuSpec(peak_bw_gbs=760, peak_tflops_fp16=29.8),
    "V100 SXM2": GpuSpec(peak_bw_gbs=900, peak_tflops_fp16=125),
    "V100 PCIe": GpuSpec(peak_bw_gbs=900, peak_tflops_fp16=125),
    "V100":      GpuSpec(peak_bw_gbs=900, peak_tflops_fp16=125),
}


@dataclass
class RooflineResult:
    bound: str
    bw_utilization: float
    compute_utilization: float
    headroom_pct: float
    gpu_matched: bool
    peak_bw_gbs: float = 0.0
    bw_source: str = "unknown"


def _find_spec(gpu_name: str) -> GpuSpec | None:
    """Match GPU name against specs table, most-specific first."""
    name = gpu_name.lower()
    for key, spec in GPU_SPECS.items():
        if key.lower() in name:
            return spec
    return None


def compute_roofline(
    gpu_name: str,
    dram_bw_gbs: float,
    sm_throughput_pct: float,
    nvml_peak_bw: float | None = None,
) -> RooflineResult:
    spec = _find_spec(gpu_name)

    # Priority: NVML > spec table > None
    if nvml_peak_bw is not None and nvml_peak_bw > 0:
        peak_bw = nvml_peak_bw
        bw_source = "nvml"
    elif spec is not None:
        peak_bw = spec.peak_bw_gbs
        bw_source = "table"
    else:
        return RooflineResult(
            bound="unknown", bw_utilization=0.0, compute_utilization=0.0,
            headroom_pct=0.0, gpu_matched=False,
        )

    bw_util = dram_bw_gbs / peak_bw if peak_bw > 0 else 0.0
    compute_util = sm_throughput_pct / 100.0
    bound = "memory" if bw_util > compute_util else "compute"
    headroom_pct = 100.0 * (1.0 - max(bw_util, compute_util))

    return RooflineResult(
        bound=bound,
        bw_utilization=bw_util,
        compute_utilization=compute_util,
        headroom_pct=headroom_pct,
        gpu_matched=True,
        peak_bw_gbs=peak_bw,
        bw_source=bw_source,
    )
