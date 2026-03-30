from __future__ import annotations

import os
import random
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean, median, stdev

from kerndiff.diff import compute_derived_metrics
from kerndiff.metrics import METRICS_BY_NCU
from kerndiff.parser import parse_ncu_csv, parse_ncu_csv_pipeline


@dataclass
class HardwareInfo:
    gpu_name: str
    sm_clock_mhz: int
    mem_clock_mhz: int
    driver_version: str
    clocks_locked: bool
    mock: bool


MOCK_HARDWARE = HardwareInfo(
    gpu_name="NVIDIA H100 SXM5 80GB",
    sm_clock_mhz=1980,
    mem_clock_mhz=2619,
    driver_version="535.104.12",
    clocks_locked=True,
    mock=True,
)

GPU_L2_SIZES = {
    "A10G": 6 * 1024 * 1024,
    "A10": 6 * 1024 * 1024,
    "H100": 50 * 1024 * 1024,
    "H200": 50 * 1024 * 1024,
    "A100": 40 * 1024 * 1024,
    "V100": 6 * 1024 * 1024,
    "L40S": 96 * 1024 * 1024,
    "L40": 48 * 1024 * 1024,
    "RTX 4090": 72 * 1024 * 1024,
    "RTX 3090": 6 * 1024 * 1024,
}


@dataclass
class ProfileResult:
    kernel_name: str
    metrics: dict[str, float]
    min_latency_us: float
    all_latencies_us: list[float]
    clean_latencies_us: list[float]
    median_latency_us: float
    p20_latency_us: float
    p80_latency_us: float
    cv_pct: float
    n_outliers: int
    ptx_instructions: dict[str, int]
    hardware: HardwareInfo
    warnings: list[str]
    actual_runs: int
    max_runs: int
    min_runs: int
    noise_threshold: float
    warmup: int
    l2_flush: bool
    output_vals: list[float] = field(default_factory=list)
    trace_events: list[dict] = field(default_factory=list)
    clock_telemetry: dict[str, object] = field(default_factory=dict)


def query_hardware(gpu_id: int) -> HardwareInfo:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,clocks.sm,clocks.mem,driver_version",
            "--format=csv,noheader,nounits",
            "-i",
            str(gpu_id),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return HardwareInfo("unknown", 0, 0, "unknown", False, False)
    parts = [p.strip() for p in result.stdout.strip().split(",")]
    if len(parts) < 4:
        return HardwareInfo("unknown", 0, 0, "unknown", False, False)
    return HardwareInfo(
        gpu_name=parts[0],
        sm_clock_mhz=int(parts[1]) if parts[1].isdigit() else 0,
        mem_clock_mhz=int(parts[2]) if parts[2].isdigit() else 0,
        driver_version=parts[3],
        clocks_locked=False,
        mock=False,
    )


def query_l2_size(gpu_id: int, gpu_name: str = "") -> int:
    # Try fuzzy match against known GPU L2 sizes first
    for name, size in GPU_L2_SIZES.items():
        if name.lower() in gpu_name.lower():
            return size
    # Fall back to nvidia-smi
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=l2_cache", "--format=csv,noheader,nounits", "-i", str(gpu_id)],
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return 6 * 1024 * 1024
    if result.returncode != 0 or not result.stdout.strip():
        return 6 * 1024 * 1024  # 6MB safe fallback
    try:
        return int(result.stdout.strip()) * 1024
    except ValueError:
        return 6 * 1024 * 1024


def _find_ncu() -> str | None:
    ncu = shutil.which("ncu")
    if ncu:
        return ncu
    # Check common installation paths
    search_dirs = [
        "/opt/nvidia/nsight-compute",
        "/usr/local/cuda/bin",
        "/usr/local/cuda/nsight-compute",
    ]
    for base in search_dirs:
        if not os.path.isdir(base):
            continue
        # Find the latest version
        for entry in sorted(os.listdir(base), reverse=True):
            candidate = os.path.join(base, entry, "ncu")
            if os.path.isfile(candidate):
                return candidate
        candidate = os.path.join(base, "ncu")
        if os.path.isfile(candidate):
            return candidate
    return None


def query_peak_bandwidth_nvml(gpu_id: int) -> float | None:
    """Return peak memory bandwidth in GB/s using NVML, or None if unavailable.

    Uses bus_width * mem_clock * 2 (DDR). NVML returns base clock on most drivers,
    so we always apply the 2x DDR multiplier.
    """
    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.filterwarnings("ignore", category=FutureWarning)
            import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        bus_width_bits = pynvml.nvmlDeviceGetMemoryBusWidth(handle)
        mem_clock_mhz = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        pynvml.nvmlShutdown()
        # DDR multiplier: NVML returns base clock, multiply by 2
        peak_bw = (bus_width_bits / 8) * (mem_clock_mhz * 1e6) * 2 / 1e9
        return peak_bw if peak_bw > 0 else None
    except Exception:
        return None


def _decode_throttle_reasons(pynvml, reasons: int) -> list[str]:
    known_reasons = [
        ("gpu_idle", getattr(pynvml, "nvmlClocksThrottleReasonGpuIdle", 0)),
        ("applications_clocks", getattr(pynvml, "nvmlClocksThrottleReasonApplicationsClocksSetting", 0)),
        ("sw_power_cap", getattr(pynvml, "nvmlClocksThrottleReasonSwPowerCap", 0)),
        ("hw_slowdown", getattr(pynvml, "nvmlClocksThrottleReasonHwSlowdown", 0)),
        ("hw_thermal_slowdown", getattr(pynvml, "nvmlClocksThrottleReasonHwThermalSlowdown", 0)),
        ("sw_thermal_slowdown", getattr(pynvml, "nvmlClocksThrottleReasonSwThermalSlowdown", 0)),
        ("sync_boost", getattr(pynvml, "nvmlClocksThrottleReasonSyncBoost", 0)),
        ("display_clock_setting", getattr(pynvml, "nvmlClocksThrottleReasonDisplayClockSetting", 0)),
    ]
    active = [name for name, mask in known_reasons if mask and (reasons & mask)]
    return active


def query_clock_telemetry(gpu_id: int, hardware: HardwareInfo | None = None) -> dict[str, object]:
    fallback = {
        "current_sm_clock_mhz": hardware.sm_clock_mhz if hardware else 0,
        "current_mem_clock_mhz": hardware.mem_clock_mhz if hardware else 0,
        "max_sm_clock_mhz": 0,
        "max_mem_clock_mhz": 0,
        "throttle_reasons": [],
    }
    try:
        import warnings as _w
        with _w.catch_warnings():
            _w.filterwarnings("ignore", category=FutureWarning)
            import pynvml

        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_id)
        current_sm = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
        current_mem = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        max_sm = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_SM)
        max_mem = pynvml.nvmlDeviceGetMaxClockInfo(handle, pynvml.NVML_CLOCK_MEM)
        throttle_bits = pynvml.nvmlDeviceGetCurrentClocksThrottleReasons(handle)
        pynvml.nvmlShutdown()
        return {
            "current_sm_clock_mhz": current_sm,
            "current_mem_clock_mhz": current_mem,
            "max_sm_clock_mhz": max_sm,
            "max_mem_clock_mhz": max_mem,
            "throttle_reasons": _decode_throttle_reasons(pynvml, throttle_bits),
        }
    except Exception:
        return fallback


def lock_clocks(gpu_id: int) -> bool:
    r1 = subprocess.run(
        ["nvidia-smi", "--lock-gpu-clocks=tdp,tdp", "-i", str(gpu_id)],
        capture_output=True,
    )
    if r1.returncode != 0:
        return False
    # Query max supported memory clock and lock to it
    query = subprocess.run(
        ["nvidia-smi", "--query-supported-clocks=mem", "--format=csv,noheader", "-i", str(gpu_id)],
        capture_output=True, text=True,
    )
    if query.returncode == 0:
        for line in query.stdout.splitlines():
            try:
                max_mem_mhz = int(line.strip().split()[0])
                subprocess.run(
                    ["nvidia-smi", f"--lock-memory-clocks={max_mem_mhz}", "-i", str(gpu_id)],
                    capture_output=True,
                )
                break
            except (ValueError, IndexError):
                pass
    return True


def unlock_clocks(gpu_id: int) -> None:
    subprocess.run(["nvidia-smi", "--reset-gpu-clocks", "-i", str(gpu_id)], capture_output=True)
    subprocess.run(["nvidia-smi", "--reset-memory-clocks", "-i", str(gpu_id)], capture_output=True)


def _compute_cv(values: list[float]) -> float:
    return (stdev(values) / mean(values)) * 100.0 if len(values) > 1 else 0.0


def _percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    if len(values) == 1:
        return values[0]
    s = sorted(values)
    pos = (len(s) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(s) - 1)
    frac = pos - lo
    return s[lo] * (1.0 - frac) + s[hi] * frac


def _remove_outliers(latencies: list[float]) -> tuple[list[float], int]:
    """Remove >2x-median runs when safe; keep original list otherwise."""
    if len(latencies) < 5:
        return latencies, 0
    med = median(latencies)
    clean = [x for x in latencies if x <= med * 2.0]
    n_removed = len(latencies) - len(clean)
    if n_removed >= len(latencies) / 2:
        return latencies, 0
    return clean, n_removed


def _emit_progress(label: str, run_idx: int, max_runs: int, cv_pct: float, enabled: bool) -> None:
    if not enabled:
        return
    cv_text = f"{cv_pct:.1f}%" if run_idx >= 2 else "n/a"
    print(
        f"\r  profiling {label}...   run {run_idx}/{max_runs}  current cv={cv_text}",
        end="",
        file=sys.stderr,
        flush=True,
    )


def _end_progress(enabled: bool) -> None:
    if enabled:
        print(file=sys.stderr)


def _synthesize_latencies(min_latency_us: float, runs: int) -> list[float]:
    rng = random.Random(42)
    latencies = [min_latency_us]
    for _ in range(max(runs - 1, 0)):
        latencies.append(min_latency_us * (1.0 + rng.uniform(0.0, 0.04)))
    return latencies


def _trace_event(
    *,
    lane: str,
    name: str,
    category: str,
    ts_us: float,
    dur_us: float,
    args: dict | None = None,
) -> dict:
    return {
        "lane": lane,
        "name": name,
        "category": category,
        "ts_us": ts_us,
        "dur_us": max(dur_us, 0.0),
        "args": args or {},
    }


def _sample_trace_events(latencies_us: list[float], start_ts_us: float) -> list[dict]:
    events: list[dict] = []
    cursor_us = start_ts_us
    for idx, latency_us in enumerate(latencies_us, start=1):
        events.append(_trace_event(
            lane="samples",
            name=f"run_{idx:03d}",
            category="timing_sample",
            ts_us=cursor_us,
            dur_us=latency_us,
            args={"latency_us": latency_us, "run_index": idx},
        ))
        cursor_us += latency_us
    return events


def profile(
    binary: str,
    kernel_name: str,
    max_runs: int,
    min_runs: int,
    noise_threshold: float,
    warmup: int,
    gpu_id: int,
    hardware: HardwareInfo,
    mock: bool = False,
    mock_prefix: str = "v1",
    env: dict | None = None,
    pipeline: int = 1,
    backend=None,
    dump_output: bool = False,
    show_progress: bool = False,
    progress_label: str = "",
) -> ProfileResult:
    warnings: list[str] = []
    trace_events: list[dict] = []
    clock_telemetry: dict[str, object] = {}

    if mock:
        fixture_path = Path(__file__).resolve().parent / "fixtures" / f"{mock_prefix}_ncu.csv"
        metrics = parse_ncu_csv(fixture_path.read_text())
        min_latency_us = metrics.get("latency_us", 0.0)
        all_latencies_us = _synthesize_latencies(min_latency_us, 20)
        cv_pct = _compute_cv(all_latencies_us)
        metrics["latency_us"] = min_latency_us
        metrics.update(compute_derived_metrics(metrics))
        timed_total_us = sum(all_latencies_us)
        trace_events.append(_trace_event(
            lane="phases",
            name="timed_runs",
            category="profile_phase",
            ts_us=0.0,
            dur_us=timed_total_us,
            args={"runs": len(all_latencies_us), "pipeline": pipeline, "mock": True},
        ))
        trace_events.extend(_sample_trace_events(all_latencies_us, 0.0))
        return ProfileResult(
            kernel_name=kernel_name,
            metrics=metrics,
            min_latency_us=min_latency_us,
            all_latencies_us=all_latencies_us,
            clean_latencies_us=all_latencies_us,
            median_latency_us=median(all_latencies_us) if all_latencies_us else min_latency_us,
            p20_latency_us=_percentile(all_latencies_us, 0.2),
            p80_latency_us=_percentile(all_latencies_us, 0.8),
            cv_pct=cv_pct,
            n_outliers=0,
            ptx_instructions={},
            hardware=MOCK_HARDWARE,
            warnings=warnings,
            actual_runs=20,
            max_runs=50,
            min_runs=10,
            noise_threshold=1.0,
            warmup=warmup,
            l2_flush=False,
            trace_events=trace_events,
            clock_telemetry=query_clock_telemetry(gpu_id, MOCK_HARDWARE),
        )

    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    l2_size_bytes = query_l2_size(gpu_id, gpu_name=hardware.gpu_name)

    warmup_start = time.perf_counter_ns()
    if backend is not None:
        _run_warmup_backend(backend, binary, kernel_name, warmup, run_env)
        warmup_dur_us = (time.perf_counter_ns() - warmup_start) / 1000.0
        trace_events.append(_trace_event(
            lane="phases",
            name="warmup",
            category="profile_phase",
            ts_us=0.0,
            dur_us=warmup_dur_us,
            args={"iters": warmup},
        ))
        timed_start = time.perf_counter_ns()
        latencies, output_vals = _run_timed_backend(
            backend, binary, kernel_name, l2_size_bytes,
            min_runs, max_runs, noise_threshold, warmup, run_env,
            dump_output=dump_output,
            show_progress=show_progress,
            progress_label=progress_label,
        )
        timed_dur_us = (time.perf_counter_ns() - timed_start) / 1000.0
    else:
        output_vals: list[float] = []
        _run_warmup_legacy(binary, kernel_name, warmup, run_env)
        warmup_dur_us = (time.perf_counter_ns() - warmup_start) / 1000.0
        trace_events.append(_trace_event(
            lane="phases",
            name="warmup",
            category="profile_phase",
            ts_us=0.0,
            dur_us=warmup_dur_us,
            args={"iters": warmup},
        ))
        timed_start = time.perf_counter_ns()
        latencies = _run_timed_legacy(
            binary, kernel_name, l2_size_bytes,
            min_runs, max_runs, noise_threshold, run_env,
            show_progress=show_progress,
            progress_label=progress_label,
        )
        timed_dur_us = (time.perf_counter_ns() - timed_start) / 1000.0

    trace_events.append(_trace_event(
        lane="phases",
        name="timed_runs",
        category="profile_phase",
        ts_us=warmup_dur_us,
        dur_us=timed_dur_us,
        args={"runs": len(latencies), "pipeline": pipeline},
    ))
    trace_events.extend(_sample_trace_events(latencies, warmup_dur_us))
    clock_telemetry = query_clock_telemetry(gpu_id, hardware)
    active_throttle_reasons = [
        reason
        for reason in clock_telemetry.get("throttle_reasons", [])
        if reason not in {"applications_clocks", "gpu_idle"}
    ]
    if active_throttle_reasons:
        warnings.append(
            "GPU reported active clock-throttle reasons during timing: "
            + ", ".join(active_throttle_reasons)
        )

    if not latencies:
        raise SystemExit(
            "error: no timing samples were collected\n"
            f"  binary/artifact: {binary}\n"
            "  this usually means the harness exited before producing latency output"
        )

    clean_latencies, n_outliers = _remove_outliers(latencies)
    if n_outliers > 0:
        warnings.append(
            f"{n_outliers} outlier run(s) detected (>2x median), excluded from statistics"
        )
    stats_latencies = clean_latencies if clean_latencies else latencies

    min_latency_us = min(stats_latencies)
    cv_pct = _compute_cv(stats_latencies)
    median_latency_us = median(stats_latencies)
    p20_latency_us = _percentile(stats_latencies, 0.2)
    p80_latency_us = _percentile(stats_latencies, 0.8)

    if cv_pct > noise_threshold and len(latencies) >= max_runs:
        warnings.append(
            f"noise threshold ({noise_threshold}%) not reached after {max_runs} runs "
            f"(cv={cv_pct:.1f}%). Consider clock locking or --noise-threshold {cv_pct:.0f}."
        )

    med = median(latencies)
    if latencies[0] > med * 1.3:
        warnings.append(
            f"first run was {int((latencies[0] / med - 1) * 100)}% slower than median. "
            f"Consider --warmup (currently {warmup})."
        )

    if max(latencies) / min_latency_us > 1.15:
        warnings.append(
            f"high variance detected (min={min_latency_us:.1f}us, max={max(latencies):.1f}us). Clock locking recommended."
        )

    ncu_path = _find_ncu()
    if ncu_path is None:
        warnings.append(
            "NCU (nsight-compute) not found. Install with: sudo apt-get install nsight-compute-2026.1.0\n"
            "  continuing with latency-only diff (no hardware counters)."
        )
        metrics: dict[str, float] = {}
    else:
        ncu_metric_names = ",".join(METRICS_BY_NCU.keys())
        ncu_start = time.perf_counter_ns()

        if backend is not None:
            # Triton: use single-run NCU harness (persistent harness can't be used by NCU)
            ncu_artifact = binary
            if hasattr(backend, "compile_ncu") and hasattr(backend, "_last_compile_args"):
                a = backend._last_compile_args
                ncu_artifact = backend.compile_ncu(
                    a["source_path"], a["kernel_name"], a["arch"],
                    a["dtype"], a["buf_elems"], a["call_expr"],
                )
            ncu_cmd = backend.ncu_cmd(ncu_path, ncu_artifact, kernel_name, ncu_metric_names, pipeline)
        else:
            ncu_cmd = [
                ncu_path,
                "--csv",
                "--metrics", ncu_metric_names,
                "--launch-count", str(pipeline),
                # Harness setup launches two _kerndiff_fill kernels before user kernel.
                "--launch-skip", "2",
                "--cache-control", "all",
                # Keep NCU counters at the GPU's live frequency to match timed behavior.
                "--clock-control", "none",
                binary,
                "--kernel", kernel_name,
                "--iters", "1",
            ]

        ncu_result = subprocess.run(ncu_cmd, capture_output=True, text=True, env=run_env)

        ncu_all_output = ((ncu_result.stdout or "") + (ncu_result.stderr or "")).lower()

        if ncu_result.returncode != 0 or "err_nvgpuctrperm" in ncu_all_output or ("permission" in ncu_all_output and "\"Metric Name\"" not in (ncu_result.stdout or "")):
            sudo = shutil.which("sudo")
            if sudo:
                print("  ncu: retrying with sudo...", file=sys.stderr)
                ncu_result = subprocess.run(
                    [sudo] + ncu_cmd, capture_output=True, text=True, env=run_env,
                )

        ncu_has_csv = "\"Metric Name\"" in (ncu_result.stdout or "")
        if not ncu_has_csv:
            ncu_all_text = ((ncu_result.stdout or "") + (ncu_result.stderr or "")).lower()
            if "permission" in ncu_all_text or "perf_event_paranoid" in ncu_all_text or "err_nvgpuctrperm" in ncu_all_text:
                warnings.append(
                    "NCU requires elevated permissions on this system.\n"
                    "  fix: sudo sh -c 'echo 0 > /proc/sys/kernel/perf_event_paranoid'\n"
                    "  or:  sudo ncu ... (run kerndiff with sudo)\n"
                    "  continuing with latency-only diff (no hardware counters)."
                )
            else:
                err_text = (ncu_result.stderr or ncu_result.stdout or "(empty)").strip()[:200]
                warnings.append(
                    f"NCU returned no usable output.\n"
                    f"  {err_text}\n"
                    "  continuing with latency-only diff (no hardware counters)."
                )
            metrics = {}
        else:
            if kernel_name and kernel_name not in (ncu_result.stdout or "") and "_kerndiff_fill" in (ncu_result.stdout or ""):
                warnings.append(
                    "NCU appears to have profiled _kerndiff_fill instead of the target kernel."
                )
            if pipeline > 1:
                metrics = parse_ncu_csv_pipeline(ncu_result.stdout, pipeline)
            else:
                metrics = parse_ncu_csv(ncu_result.stdout)
        ncu_dur_us = (time.perf_counter_ns() - ncu_start) / 1000.0
        trace_events.append(_trace_event(
            lane="phases",
            name="ncu",
            category="profile_phase",
            ts_us=warmup_dur_us + timed_dur_us,
            dur_us=ncu_dur_us,
            args={"metrics_collected": len(metrics), "pipeline": pipeline},
        ))

    metrics["latency_us"] = min_latency_us
    metrics.update(compute_derived_metrics(metrics))

    return ProfileResult(
        kernel_name=kernel_name,
        metrics=metrics,
        min_latency_us=min_latency_us,
        all_latencies_us=latencies,
        clean_latencies_us=stats_latencies,
        median_latency_us=median_latency_us,
        p20_latency_us=p20_latency_us,
        p80_latency_us=p80_latency_us,
        cv_pct=cv_pct,
        n_outliers=n_outliers,
        ptx_instructions={},
        hardware=hardware,
        warnings=warnings,
        actual_runs=len(latencies),
        max_runs=max_runs,
        min_runs=min_runs,
        noise_threshold=noise_threshold,
        warmup=warmup,
        l2_flush=True,
        output_vals=output_vals,
        trace_events=trace_events,
        clock_telemetry=clock_telemetry,
    )


def _run_warmup_legacy(binary, kernel_name, warmup, run_env):
    try:
        subprocess.run(
            [binary, "--kernel", kernel_name, "--iters", str(warmup)],
            capture_output=True, text=True, check=True, env=run_env,
        )
    except subprocess.CalledProcessError as e:
        stderr_text = (e.stderr or "").strip()
        raise SystemExit(
            f"error: kernel crashed during warmup (exit code {e.returncode})\n\n"
            f"  binary: {binary}\n"
            f"  stderr: {stderr_text}\n\n"
            f"  this usually means:\n"
            f"  - buffer size is too small for the kernel's access pattern\n"
            f"  - the kernel requires a specific launch config (use --call)\n"
            f"  - the kernel has a bug"
        )


def _run_warmup_backend(backend, binary, kernel_name, warmup, run_env):
    """For backend-aware mode, warmup is handled inside the harness."""
    if hasattr(backend, "is_persistent") and backend.is_persistent():
        return
    cmd = backend.run_cmd(binary, kernel_name, iters=warmup, l2_flush=0)
    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True, env=run_env)
    except subprocess.CalledProcessError as e:
        stderr_text = (e.stderr or "").strip()
        raise SystemExit(
            f"error: kernel crashed during warmup (exit code {e.returncode})\n\n"
            f"  artifact: {binary}\n"
            f"  stderr: {stderr_text[:500]}\n\n"
            f"  this usually means:\n"
            f"  - the kernel has a bug or import error\n"
            f"  - buffer size is too small for the kernel's access pattern\n"
            f"  - the kernel requires a specific launch config (use --call)"
        )


def _run_timed_legacy(
    binary,
    kernel_name,
    l2_size_bytes,
    min_runs,
    max_runs,
    noise_threshold,
    run_env,
    show_progress: bool = False,
    progress_label: str = "",
):
    latencies: list[float] = []
    cv_pct = float("inf")
    try:
        while len(latencies) < min_runs or (cv_pct > noise_threshold and len(latencies) < max_runs):
            try:
                r = subprocess.run(
                    [binary, "--kernel", kernel_name, "--iters", "1", "--l2-flush", str(l2_size_bytes)],
                    capture_output=True, text=True, check=True, env=run_env,
                )
            except subprocess.CalledProcessError as e:
                stderr_text = (e.stderr or "").strip()
                raise SystemExit(
                    f"error: kernel crashed during timing run (exit code {e.returncode})\n\n"
                    f"  binary: {binary}\n"
                    f"  stderr: {stderr_text}\n\n"
                    f"  this usually means:\n"
                    f"  - buffer size is too small for the kernel's access pattern\n"
                    f"  - the kernel requires a specific launch config (use --call)\n"
                    f"  - the kernel has a bug"
                )
            latencies.append(float(r.stdout.strip()))
            if len(latencies) >= 2:
                cv_pct = _compute_cv(latencies)
            _emit_progress(progress_label or kernel_name, len(latencies), max_runs, cv_pct, show_progress)
    finally:
        _end_progress(show_progress)
    return latencies


def _run_timed_backend(
    backend,
    binary,
    kernel_name,
    l2_size_bytes,
    min_runs,
    max_runs,
    noise_threshold,
    warmup,
    run_env,
    dump_output: bool = False,
    show_progress: bool = False,
    progress_label: str = "",
):
    """Timed runs using backend. Persistent backends use pipe protocol; others spawn per-run."""
    if hasattr(backend, "compile_timed") and hasattr(backend, "_last_compile_args"):
        args = backend._last_compile_args
        binary = backend.compile_timed(
            args["source_path"], args["kernel_name"], args["arch"],
            args["dtype"], args["buf_elems"], args["call_expr"],
            iters=1, l2_flush_bytes=l2_size_bytes, warmup=warmup,
        )

    if hasattr(backend, "is_persistent") and backend.is_persistent():
        proc = backend.spawn_persistent(binary, env=run_env)
        try:
            captured_output: list[float] = []
            if dump_output and hasattr(backend, "dump_output"):
                captured_output = backend.dump_output(proc)
            latencies: list[float] = []
            cv_pct = float("inf")
            while len(latencies) < min_runs or (
                cv_pct > noise_threshold and len(latencies) < max_runs
            ):
                us = backend.send_time(proc)
                latencies.append(us)
                if len(latencies) >= 2:
                    cv_pct = _compute_cv(latencies)
                _emit_progress(progress_label or kernel_name, len(latencies), max_runs, cv_pct, show_progress)
        finally:
            backend.shutdown(proc)
            _end_progress(show_progress)
        return latencies, captured_output

    latencies = []
    cv_pct = float("inf")
    try:
        while len(latencies) < min_runs or (cv_pct > noise_threshold and len(latencies) < max_runs):
            cmd = backend.run_cmd(binary, kernel_name, iters=1, l2_flush=l2_size_bytes)
            try:
                r = subprocess.run(cmd, capture_output=True, text=True, check=True, env=run_env)
            except subprocess.CalledProcessError as e:
                stderr_text = (e.stderr or "").strip()
                raise SystemExit(
                    f"error: kernel crashed during timing run (exit code {e.returncode})\n\n"
                    f"  artifact: {binary}\n"
                    f"  stderr: {stderr_text[:500]}\n\n"
                    f"  this usually means:\n"
                    f"  - the kernel has a bug or import error\n"
                    f"  - buffer size is too small for the kernel's access pattern\n"
                    f"  - the kernel requires a specific launch config (use --call)"
                )
            try:
                latencies.append(float(r.stdout.strip().splitlines()[-1]))
            except (ValueError, IndexError):
                raise SystemExit(
                    f"error: could not parse latency from output\n"
                    f"  stdout: {r.stdout[:200]}"
                )
            if len(latencies) >= 2:
                cv_pct = _compute_cv(latencies)
            _emit_progress(progress_label or kernel_name, len(latencies), max_runs, cv_pct, show_progress)
    finally:
        _end_progress(show_progress)
    return latencies, []
