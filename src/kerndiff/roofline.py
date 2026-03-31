from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GpuSpec:
    peak_bw_gbs: float
    peak_tflops_fp16: float
    peak_tflops_fp32: float  # non-tensor-core FP32 throughput


# Ordered most-specific first so the first substring match wins.
GPU_SPECS: dict[str, GpuSpec] = {
    "H100 SXM5": GpuSpec(peak_bw_gbs=3350, peak_tflops_fp16=989,  peak_tflops_fp32=67),
    "H100 SXM":  GpuSpec(peak_bw_gbs=3350, peak_tflops_fp16=989,  peak_tflops_fp32=67),
    "H100 PCIe": GpuSpec(peak_bw_gbs=2000, peak_tflops_fp16=800,  peak_tflops_fp32=51),
    "H100":      GpuSpec(peak_bw_gbs=3350, peak_tflops_fp16=989,  peak_tflops_fp32=67),
    "H200 SXM":  GpuSpec(peak_bw_gbs=4800, peak_tflops_fp16=989,  peak_tflops_fp32=67),
    "H200":      GpuSpec(peak_bw_gbs=4800, peak_tflops_fp16=989,  peak_tflops_fp32=67),
    "A100 SXM4": GpuSpec(peak_bw_gbs=2000, peak_tflops_fp16=312,  peak_tflops_fp32=19.5),
    "A100 SXM":  GpuSpec(peak_bw_gbs=2000, peak_tflops_fp16=312,  peak_tflops_fp32=19.5),
    "A100 PCIe": GpuSpec(peak_bw_gbs=1935, peak_tflops_fp16=312,  peak_tflops_fp32=19.5),
    "A100":      GpuSpec(peak_bw_gbs=2000, peak_tflops_fp16=312,  peak_tflops_fp32=19.5),
    "A10G":      GpuSpec(peak_bw_gbs=600,  peak_tflops_fp16=31.2, peak_tflops_fp32=31.2),
    "A10":       GpuSpec(peak_bw_gbs=600,  peak_tflops_fp16=31.2, peak_tflops_fp32=31.2),
    "L40S":      GpuSpec(peak_bw_gbs=864,  peak_tflops_fp16=362,  peak_tflops_fp32=91.6),
    "L40":       GpuSpec(peak_bw_gbs=864,  peak_tflops_fp16=181,  peak_tflops_fp32=90.5),
    "RTX 4090":  GpuSpec(peak_bw_gbs=1008, peak_tflops_fp16=82.6, peak_tflops_fp32=82.6),
    "RTX 3090":  GpuSpec(peak_bw_gbs=936,  peak_tflops_fp16=35.6, peak_tflops_fp32=35.6),
    "RTX 3080":  GpuSpec(peak_bw_gbs=760,  peak_tflops_fp16=29.8, peak_tflops_fp32=29.8),
    "V100 SXM2": GpuSpec(peak_bw_gbs=900,  peak_tflops_fp16=125,  peak_tflops_fp32=14.0),
    "V100 PCIe": GpuSpec(peak_bw_gbs=900,  peak_tflops_fp16=125,  peak_tflops_fp32=14.0),
    "V100":      GpuSpec(peak_bw_gbs=900,  peak_tflops_fp16=125,  peak_tflops_fp32=14.0),
}


_TENSOR_CORE_THRESHOLD_PCT = 10.0  # above this tensor_core_util, use fp16 ridge point


@dataclass
class RooflineResult:
    bound: str
    bw_utilization: float
    compute_utilization: float
    headroom_pct: float
    gpu_matched: bool
    peak_bw_gbs: float = 0.0
    bw_source: str = "unknown"
    ridge_point: float = 0.0        # FLOPs/byte; 0 when arith_intensity was unavailable
    used_tensor_core_peak: bool = False  # True when fp16 peak was used for ridge


def _find_spec(gpu_name: str) -> GpuSpec | None:
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
    arith_intensity: float = 0.0,
    tensor_core_util: float = 0.0,
) -> RooflineResult:
    spec = _find_spec(gpu_name)

    peak_bw = nvml_peak_bw if (nvml_peak_bw is not None and nvml_peak_bw > 0) else (
        spec.peak_bw_gbs if spec is not None else None
    )
    bw_source = "nvml" if (nvml_peak_bw is not None and nvml_peak_bw > 0) else "table"

    if peak_bw is None:
        return RooflineResult(
            bound="unknown", bw_utilization=0.0, compute_utilization=0.0,
            headroom_pct=0.0, gpu_matched=False,
        )

    bw_util = dram_bw_gbs / peak_bw if peak_bw > 0 else 0.0
    compute_util = sm_throughput_pct / 100.0

    if arith_intensity > 0 and spec is not None:
        # When the kernel meaningfully uses tensor cores, the relevant compute
        # ceiling is fp16 (tensor-core) throughput, not scalar fp32.
        use_tc_peak = tensor_core_util >= _TENSOR_CORE_THRESHOLD_PCT
        peak_tflops = spec.peak_tflops_fp16 if use_tc_peak else spec.peak_tflops_fp32
        ridge_point = (peak_tflops * 1e12) / (peak_bw * 1e9)
        bound = "compute" if arith_intensity > ridge_point else "memory"
        headroom_pct = 100.0 * (1.0 - min(bw_util, 1.0)) if bound == "memory" else \
                       100.0 * (1.0 - min(compute_util, 1.0))
    else:
        use_tc_peak = False
        ridge_point = 0.0
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
        ridge_point=ridge_point,
        used_tensor_core_peak=use_tc_peak,
    )
