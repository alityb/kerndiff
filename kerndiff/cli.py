from __future__ import annotations

import argparse
import atexit
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from kerndiff.compiler import compile_kernel
from kerndiff.diff import NOISE_FLOOR_LOCKED, NOISE_FLOOR_UNLOCKED, compute_all_deltas, compute_verdict, sort_deltas
from kerndiff.profiler import MOCK_HARDWARE, lock_clocks, profile, query_hardware, unlock_clocks
from kerndiff import ptx
from kerndiff.renderer import build_json_payload, render_json, render_metric_table, render_ptx_diff, render_verdict, write_output
from kerndiff.roofline import compute_roofline

KERNEL_RE = re.compile(r"__global__\s+\w+\s+(\w+)\s*\(")
_TEMP_PATHS: list[str] = []


def _cleanup_temp_paths() -> None:
    for path in _TEMP_PATHS:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


atexit.register(_cleanup_temp_paths)


def detect_sm_arch(gpu_name: str) -> str | None:
    name = gpu_name.lower()
    if "h100" in name:
        return "sm_90"
    if "h200" in name:
        return "sm_90"
    if "a100" in name:
        return "sm_80"
    if "a10" in name:
        return "sm_86"
    if "l40" in name:
        return "sm_89"
    if "4090" in name:
        return "sm_89"
    if "4080" in name:
        return "sm_89"
    if "3090" in name:
        return "sm_86"
    if "3080" in name:
        return "sm_86"
    if "v100" in name:
        return "sm_70"
    return None


def _scan_kernels(path: str) -> list[str]:
    return KERNEL_RE.findall(Path(path).read_text())


def resolve_kernel_name(file_a: str, file_b: str, fn_name: str | None) -> str:
    if fn_name:
        return fn_name
    kernels_a = _scan_kernels(file_a)
    kernels_b = _scan_kernels(file_b)
    if len(kernels_a) == 1 and len(kernels_b) == 1 and kernels_a[0] == kernels_b[0]:
        return kernels_a[0]
    msg = [
        "error: could not auto-detect kernel — please specify --fn",
        f"  {file_a}: {', '.join(kernels_a) or '(none)'}",
        f"  {file_b}: {', '.join(kernels_b) or '(none)'}",
    ]
    raise SystemExit("\n".join(msg))


def _status_line(label: str, status: str, suffix: str = "") -> str:
    return f"  {label:<34}{status:>8}{('  ' + suffix) if suffix else ''}"


def _emit_status(label: str, status: str, suffix: str = "") -> None:
    print(_status_line(label, status, suffix), file=sys.stderr)


def _warn(msg: str, warnings: list[str]) -> None:
    warnings.append(msg)
    print(f"warning: {msg}", file=sys.stderr)


def _prepare_git_mode(file_a: str) -> tuple[str, str]:
    repo = subprocess.run(["git", "rev-parse", "--show-toplevel"], capture_output=True, text=True)
    if repo.returncode != 0:
        raise SystemExit("error: not inside a git repository (required for single-file mode)")
    repo_root = repo.stdout.strip()
    rel_path = os.path.relpath(os.path.abspath(file_a), repo_root)
    committed = subprocess.run(["git", "show", f"HEAD:{rel_path}"], capture_output=True, text=True)
    if committed.returncode != 0:
        raise SystemExit(f"error: {os.path.basename(file_a)} has no prior git commits to compare against")
    temp_dir = tempfile.mkdtemp(prefix="kerndiff_")
    _TEMP_PATHS.append(temp_dir)
    head_file = os.path.join(temp_dir, "head_kernel.cu")
    Path(head_file).write_text(committed.stdout)
    return head_file, file_a


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kerndiff", add_help=True)
    parser.add_argument("files", nargs="+", help=".cu file(s)")
    parser.add_argument("--fn", dest="fn_name")
    parser.add_argument("--call", dest="call_expr", default=None,
                        help="kernel launch expression (for non-standard signatures)")
    parser.add_argument("--max-runs", type=int, default=50)
    parser.add_argument("--min-runs", type=int, default=10)
    parser.add_argument("--noise-threshold", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--format", choices=["term", "json"], default="term")
    parser.add_argument("--output")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arch", default="sm_90")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if len(args.files) not in {1, 2}:
        parser.error("expected one or two .cu files")
    file_a = args.files[0]
    file_b = args.files[1] if len(args.files) == 2 else None
    use_color = not args.no_color and args.format == "term" and sys.stdout.isatty()
    aggregate_warnings: list[str] = []

    # --- GPU / CUDA_VISIBLE_DEVICES resolution ---
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.mock:
        hardware = MOCK_HARDWARE
        print(f"  gpu: {hardware.gpu_name} (mock)", file=sys.stderr)
        _warn("mock mode -- no GPU required.", aggregate_warnings)
        physical_gpu_id = args.gpu
        binary_env: dict | None = None
    else:
        if cuda_visible is not None:
            # Env var already set by scheduler or user — respect it.
            if args.gpu != 0:
                _warn(
                    f"CUDA_VISIBLE_DEVICES={cuda_visible} is set; --gpu {args.gpu} ignored.",
                    aggregate_warnings,
                )
            try:
                physical_gpu_id = int(cuda_visible.split(",")[0])
            except (ValueError, IndexError):
                physical_gpu_id = 0
            binary_env = None  # already in env
        else:
            physical_gpu_id = args.gpu
            binary_env = {"CUDA_VISIBLE_DEVICES": str(args.gpu)}

        hardware = query_hardware(gpu_id=physical_gpu_id)
        if args.arch == "sm_90":
            detected_arch = detect_sm_arch(hardware.gpu_name)
            if detected_arch:
                args.arch = detected_arch
        state = "clocks locked" if hardware.clocks_locked else "clocks unlocked"
        print(f"  gpu: {hardware.gpu_name} (device {physical_gpu_id}, {args.arch}, {state})", file=sys.stderr)

    # --- File resolution ---
    if file_b is None:
        if args.mock:
            file_b = file_a
            file_a = "fixtures/v1_ncu.csv"
        else:
            file_a, file_b = _prepare_git_mode(file_a)

    if not args.mock:
        for path in (file_a, file_b):
            if not Path(path).exists():
                raise SystemExit(f"error: file not found: {path}")

    kernel_name = args.fn_name
    if kernel_name is None and Path(file_a).exists() and Path(file_b).exists():
        kernel_name = resolve_kernel_name(file_a, file_b, None)
    elif kernel_name is None:
        raise SystemExit("error: could not auto-detect kernel — please specify --fn")

    # --- Compilation ---
    _emit_status("compiling...", "ok", f"{0.0:.1f}s" if args.mock else "")
    if not args.mock:
        compile_start = time.perf_counter()
        binary_a = compile_kernel(file_a, kernel_name, arch=args.arch, mock=False, kernel_call=args.call_expr)
        binary_b = compile_kernel(file_b, kernel_name, arch=args.arch, mock=False, kernel_call=args.call_expr)
        _emit_status("compiling...", "ok", f"{time.perf_counter() - compile_start:.1f}s")
    else:
        binary_a = file_a
        binary_b = file_b

    # --- Clock locking ---
    noise_floor = NOISE_FLOOR_LOCKED
    if args.mock:
        hardware.clocks_locked = True
    else:
        locked = False
        try:
            locked = lock_clocks(physical_gpu_id)
        except FileNotFoundError:
            locked = False
        hardware.clocks_locked = locked
        if locked:
            _emit_status("locking clocks...", "ok")
        else:
            noise_floor = NOISE_FLOOR_UNLOCKED
            _emit_status("locking clocks...", "skipped", "-- results may vary +/-10%")
            _warn("clock locking unavailable (requires sudo). Results may vary +/-10%.", aggregate_warnings)

    # --- Profiling ---
    try:
        _emit_status(f"warming up ({args.warmup} iters)...", "ok")

        result_a = profile(
            binary_a, kernel_name,
            max_runs=args.max_runs, min_runs=args.min_runs,
            noise_threshold=args.noise_threshold, warmup=args.warmup,
            gpu_id=physical_gpu_id, hardware=hardware,
            mock=args.mock, mock_prefix="v1", env=binary_env,
        )
        _emit_status(
            "profiling v1 (cold, adaptive)...",
            "ok",
            f"{result_a.actual_runs} runs  {result_a.min_latency_us:.0f}us  cv={result_a.cv_pct:.1f}%",
        )

        result_b = profile(
            binary_b, kernel_name,
            max_runs=args.max_runs, min_runs=args.min_runs,
            noise_threshold=args.noise_threshold, warmup=args.warmup,
            gpu_id=physical_gpu_id, hardware=hardware,
            mock=args.mock, mock_prefix="v2", env=binary_env,
        )
        _emit_status(
            "profiling v2 (cold, adaptive)...",
            "ok",
            f"{result_b.actual_runs} runs  {result_b.min_latency_us:.0f}us  cv={result_b.cv_pct:.1f}%",
        )
    finally:
        if not args.mock:
            unlock_clocks(physical_gpu_id)

    for warning in result_a.warnings + result_b.warnings:
        _warn(warning, aggregate_warnings)

    # --- PTX ---
    _emit_status("extracting ptx...", "ok")
    if args.mock:
        ptx_a = ptx.load_fixture("v1")
        ptx_b = ptx.load_fixture("v2")
    else:
        ptx_a = ptx.extract_ptx(file_a, arch=args.arch)
        ptx_b = ptx.extract_ptx(file_b, arch=args.arch)
    result_a.ptx_instructions = ptx_a
    result_b.ptx_instructions = ptx_b

    deltas = sort_deltas(compute_all_deltas(result_a.metrics, result_b.metrics, noise_floor=noise_floor))
    verdict = compute_verdict(result_a, result_b, noise_floor=noise_floor)
    roofline_v1 = compute_roofline(
        hardware.gpu_name,
        result_a.metrics.get("dram_bw_gbs", 0.0),
        result_a.metrics.get("sm_throughput", 0.0),
    )
    roofline = compute_roofline(
        hardware.gpu_name,
        result_b.metrics.get("dram_bw_gbs", 0.0),
        result_b.metrics.get("sm_throughput", 0.0),
    )
    ptx_diff = ptx.diff_ptx(ptx_a, ptx_b)

    if args.format == "json":
        payload = build_json_payload(
            hardware=hardware,
            kernel_name=kernel_name,
            file_a=file_a,
            file_b=file_b,
            actual_runs=max(result_a.actual_runs, result_b.actual_runs),
            max_runs=args.max_runs,
            min_runs=args.min_runs,
            noise_threshold=args.noise_threshold,
            warmup=args.warmup,
            l2_flush=result_a.l2_flush or result_b.l2_flush,
            verdict=verdict,
            deltas=deltas,
            roofline=roofline,
            roofline_v1_bw=roofline_v1.bw_utilization,
            ptx_diff=ptx_diff,
            warnings=aggregate_warnings,
        )
        output = render_json(payload)
    else:
        sections = [
            render_verdict(verdict, use_color=use_color),
            render_metric_table(
                deltas,
                result_a,
                result_b,
                roofline=roofline,
                roofline_v1_bw=roofline_v1.bw_utilization,
                roofline_v1_compute=roofline_v1.compute_utilization,
                use_color=use_color,
            ),
        ]
        ptx_section = render_ptx_diff(ptx_diff)
        if ptx_section:
            sections.append(ptx_section)
        output = "\n".join(sections)

    if args.output:
        write_output(output, args.output)
    else:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
