from __future__ import annotations

from dataclasses import dataclass

GPU_SPECS = {
    "H100 SXM5": {"peak_bw_gbs": 3350, "peak_tflops_fp16": 1979},
    "H100 PCIe": {"peak_bw_gbs": 2000, "peak_tflops_fp16": 1513},
    "H200 SXM": {"peak_bw_gbs": 4800, "peak_tflops_fp16": 1979},
    "A100 SXM4": {"peak_bw_gbs": 2000, "peak_tflops_fp16": 312},
    "A100 PCIe": {"peak_bw_gbs": 1935, "peak_tflops_fp16": 312},
    "A10G": {"peak_bw_gbs": 600, "peak_tflops_fp16": 125},
    "L40S": {"peak_bw_gbs": 864, "peak_tflops_fp16": 733},
    "RTX 4090": {"peak_bw_gbs": 1008, "peak_tflops_fp16": 82.6},
    "RTX 3090": {"peak_bw_gbs": 936, "peak_tflops_fp16": 35.6},
    "RTX 3080": {"peak_bw_gbs": 760, "peak_tflops_fp16": 29.8},
}


@dataclass
class RooflineResult:
    bound: str
    bw_utilization: float
    compute_utilization: float
    headroom_pct: float
    gpu_matched: bool


def fuzzy_match_gpu(gpu_name: str) -> str | None:
    haystack = gpu_name.lower().replace("-", " ")
    for name in GPU_SPECS:
        needle = name.lower().replace("-", " ")
        if needle in haystack:
            return name
    return None


def compute_roofline(gpu_name: str, dram_bw_gbs: float, sm_throughput_pct: float) -> RooflineResult:
    match = fuzzy_match_gpu(gpu_name)
    if match is None:
        return RooflineResult(bound="unknown", bw_utilization=0.0, compute_utilization=0.0, headroom_pct=0.0, gpu_matched=False)
    spec = GPU_SPECS[match]
    bw_util = dram_bw_gbs / spec["peak_bw_gbs"] if spec["peak_bw_gbs"] else 0.0
    compute_util = sm_throughput_pct / 100.0
    bound = "memory" if bw_util > compute_util else "compute"
    headroom_pct = 100.0 * (1.0 - max(bw_util, compute_util))
    return RooflineResult(bound=bound, bw_utilization=bw_util, compute_utilization=compute_util, headroom_pct=headroom_pct, gpu_matched=True)
