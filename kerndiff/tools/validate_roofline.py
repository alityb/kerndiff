"""
Validates that kerndiff's roofline numbers match NCU's own peak bandwidth report.
Run on a machine with a real GPU and NCU access.

Usage: python3 kerndiff/tools/validate_roofline.py --gpu 0
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from kerndiff.profiler import query_peak_bandwidth_nvml, query_hardware
from kerndiff.roofline import compute_roofline, GPU_SPECS, _find_spec


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    hw = query_hardware(args.gpu)
    print(f"GPU: {hw.gpu_name}")
    print(f"SM clock: {hw.sm_clock_mhz} MHz")
    print(f"Mem clock: {hw.mem_clock_mhz} MHz")

    # NVML bandwidth
    nvml_bw = query_peak_bandwidth_nvml(args.gpu)
    if nvml_bw:
        print(f"\nNVML peak bandwidth: {nvml_bw:.1f} GB/s")
    else:
        print("\nNVML peak bandwidth: unavailable")

    # Table bandwidth
    spec = _find_spec(hw.gpu_name)
    if spec:
        print(f"Table peak bandwidth: {spec.peak_bw_gbs:.1f} GB/s")
        if nvml_bw:
            diff_pct = abs(nvml_bw - spec.peak_bw_gbs) / spec.peak_bw_gbs * 100
            status = "OK" if diff_pct < 10 else "MISMATCH"
            print(f"Difference: {diff_pct:.1f}% [{status}]")
    else:
        print(f"Table: no entry for '{hw.gpu_name}'")

    # Raw NVML params for transparency
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
        bus_width = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
        mem_clk = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        raw_bw_no_ddr = (bus_width / 8) * (mem_clk * 1e6) / 1e9
        raw_bw_ddr = raw_bw_no_ddr * 2
        print(f"\nRaw NVML params:")
        print(f"  bus_width = {bus_width} bits")
        print(f"  mem_clock = {mem_clk} MHz")
        print(f"  computed peak (base clock only) = {raw_bw_no_ddr:.1f} GB/s")
        print(f"  computed peak (with 2x DDR)     = {raw_bw_ddr:.1f} GB/s")
        pynvml.nvmlShutdown()
    except Exception as e:
        print(f"\nRaw NVML query failed: {e}")

    # Test compute_roofline with NVML
    if nvml_bw:
        result = compute_roofline(hw.gpu_name, 500.0, 50.0, nvml_peak_bw=nvml_bw)
        print(f"\nRoofline test (dram=500 GB/s, sm=50%):")
        print(f"  bw_utilization: {result.bw_utilization:.1%}")
        print(f"  bound: {result.bound}")
        print(f"  bw_source: {result.bw_source}")
        print(f"  peak_bw_gbs: {result.peak_bw_gbs:.1f}")


if __name__ == "__main__":
    main()
