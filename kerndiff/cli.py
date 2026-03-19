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

from kerndiff.backends import dispatch as dispatch_backend
from kerndiff.compiler import compile_kernel, verify_correctness
from kerndiff.config import apply_config, find_config, load_config
from kerndiff.diff import NOISE_FLOOR_LOCKED, NOISE_FLOOR_UNLOCKED, compute_all_deltas, compute_verdict, sort_deltas
from kerndiff.profiler import MOCK_HARDWARE, lock_clocks, profile, query_hardware, query_peak_bandwidth_nvml, unlock_clocks
from kerndiff import ptx
from kerndiff.renderer import (
    build_json_payload, render_json, render_metric_table, render_ptx_diff,
    render_verdict, render_shape_table, write_output,
)
from kerndiff.roofline import compute_roofline

KERNEL_RE = re.compile(r"__global__\s+\w+\s+(\w+)\s*\(")
TRITON_KERNEL_RE = re.compile(r"@triton\.jit\s+def\s+(\w+)")
_TEMP_PATHS: list[str] = []
_SUPPRESS_STDERR = False


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
    text = Path(path).read_text()
    if path.endswith(".py"):
        return TRITON_KERNEL_RE.findall(text)
    return KERNEL_RE.findall(text)


def resolve_kernel_name(file_a: str, file_b: str, fn_name: str | None) -> str:
    if fn_name:
        return fn_name
    kernels_a = _scan_kernels(file_a)
    kernels_b = _scan_kernels(file_b)
    if len(kernels_a) == 1 and len(kernels_b) == 1 and kernels_a[0] == kernels_b[0]:
        return kernels_a[0]

    # Find common kernels for picker
    common = sorted(set(kernels_a) & set(kernels_b))

    # Interactive picker if tty and there are common kernels
    if common and sys.stdin.isatty():
        print("multiple kernels found — pick one (or use --fn):", file=sys.stderr)
        for i, name in enumerate(common, 1):
            print(f"  [{i}] {name}", file=sys.stderr)
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\naborted")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(common):
                return common[idx]
        # Try as a name
        if choice in common:
            return choice
        raise SystemExit(f"error: invalid selection '{choice}'")

    msg = [
        "error: could not auto-detect kernel — please specify --fn",
        f"  {file_a}: {', '.join(kernels_a) or '(none)'}",
        f"  {file_b}: {', '.join(kernels_b) or '(none)'}",
    ]
    raise SystemExit("\n".join(msg))


def resolve_all_kernels(file_a: str, file_b: str) -> list[str]:
    kernels_a = set(_scan_kernels(file_a))
    kernels_b = set(_scan_kernels(file_b))
    if not kernels_a:
        raise SystemExit(f"error: no kernels found in {os.path.basename(file_a)}")
    if not kernels_b:
        raise SystemExit(f"error: no kernels found in {os.path.basename(file_b)}")
    common = sorted(kernels_a & kernels_b)
    if not _SUPPRESS_STDERR:
        for name in sorted(kernels_a - kernels_b):
            print(f"  skipping {name} (not in {os.path.basename(file_b)})", file=sys.stderr)
        for name in sorted(kernels_b - kernels_a):
            print(f"  skipping {name} (not in {os.path.basename(file_a)})", file=sys.stderr)
    if not common:
        raise SystemExit(
            f"error: no kernels in common between {os.path.basename(file_a)} and {os.path.basename(file_b)}"
        )
    return common


def _status_line(label: str, status: str, suffix: str = "") -> str:
    return f"  {label:<34}{status:>8}{('  ' + suffix) if suffix else ''}"


def _emit_status(label: str, status: str, suffix: str = "") -> None:
    if _SUPPRESS_STDERR:
        return
    print(_status_line(label, status, suffix), file=sys.stderr)


def _warn(msg: str, warnings: list[str]) -> None:
    warnings.append(msg)
    if _SUPPRESS_STDERR:
        return
    print(f"warning: {msg}", file=sys.stderr)


def resolve_git_baseline(filepath: str, at_ref: str = "HEAD") -> tuple[str, str]:
    """Extract a git ref version of a file to a temp path.

    Returns (temp_path, display_label) where display_label is like "HEAD:src/kernel.cu".
    """
    abs_path = Path(filepath).resolve()

    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True, text=True, cwd=abs_path.parent,
    )
    if result.returncode != 0:
        raise SystemExit("error: single-file mode requires a git repo (run inside a git repo, or pass two files)")

    repo_root = Path(result.stdout.strip())
    rel_path = abs_path.relative_to(repo_root)

    # Check file is tracked
    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(rel_path)],
        capture_output=True, cwd=repo_root,
    )
    if tracked.returncode != 0:
        raise SystemExit(f"error: {rel_path} is not tracked by git (git add it first)")

    # Resolve short hash for display
    if at_ref == "HEAD":
        display_ref = "HEAD"
    else:
        short = subprocess.run(
            ["git", "rev-parse", "--short", at_ref],
            capture_output=True, text=True, cwd=repo_root,
        )
        display_ref = short.stdout.strip() if short.returncode == 0 else at_ref

    head_content = subprocess.run(
        ["git", "show", f"{at_ref}:{rel_path}"],
        capture_output=True, text=True, cwd=repo_root,
    )
    if head_content.returncode != 0:
        raise SystemExit(
            f"error: file not found in {at_ref}: {rel_path}\n"
            f"  (commit it first, or pass two files explicitly)"
        )

    suffix = abs_path.suffix or ".cu"
    tmp = tempfile.NamedTemporaryFile(
        suffix=suffix, prefix="kerndiff_head_", delete=False, mode="w",
    )
    tmp.write(head_content.stdout)
    tmp.close()
    _TEMP_PATHS.append(tmp.name)

    display_label = f"{display_ref}:{rel_path}"
    return tmp.name, display_label


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kerndiff", add_help=True)
    parser.add_argument("files", nargs="+", help=".cu or .py file(s)")
    parser.add_argument("--fn", dest="fn_name")
    parser.add_argument("--all", dest="all_kernels", action="store_true",
                        help="profile all common kernels in both files")
    parser.add_argument("--call", dest="call_expr", default=None,
                        help="kernel launch expression (for non-standard signatures)")
    parser.add_argument("--dtype", choices=["float", "half", "int", "int4"], default="float",
                        help="element type for harness buffers (default: float)")
    parser.add_argument("--elems", type=int, default=1 << 22,
                        help="number of buffer elements in harness (default: 4194304)")
    parser.add_argument("--pipeline", type=int, default=1,
                        help="number of kernel launches in a pipeline (NCU profiles all N)")
    parser.add_argument("--shape", default=None,
                        help="comma-separated list of buffer sizes for sweep (e.g. 1024,2048,4096)")
    parser.add_argument("--correctness", action="store_true",
                        help="verify v1 and v2 produce the same output before profiling")
    parser.add_argument("--tol", type=float, default=1e-4,
                        help="tolerance for correctness check (default: 1e-4)")
    parser.add_argument("--watch", action="store_true",
                        help="re-run on file change (poll every 500ms)")
    parser.add_argument("--validate", action="store_true",
                        help="run forward and reverse profiles to check consistency")
    parser.add_argument("--at", dest="at_ref", default=None,
                        help="git ref to compare against in single-file mode (default: HEAD)")
    parser.add_argument("--max-runs", type=int, default=50)
    parser.add_argument("--min-runs", type=int, default=10)
    parser.add_argument("--noise-threshold", type=float, default=1.0)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--format", choices=["term", "json"], default="term")
    parser.add_argument("--output")
    parser.add_argument(
        "--export-json",
        dest="export_json",
        metavar="FILE",
        help="write JSON output to FILE (implies --format json). Progress still shown on stderr.",
    )
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arch", default="sm_90")
    return parser


def _safe_diff(a: float, b: float) -> float:
    import math
    if math.isnan(a) or math.isnan(b):
        return float("nan")
    if math.isinf(a) or math.isinf(b):
        return float("inf")
    return abs(a - b)


def _emit_correctness_result(
    max_diff: float,
    v1_vals: list[float],
    v2_vals: list[float],
    tol: float,
    label: str,
    aggregate_warnings: list[str],
) -> None:
    import math
    if math.isnan(max_diff) or max_diff > tol:
        summary = f"max_diff={max_diff:.4g}"
        if v1_vals and v2_vals:
            summary += f"  (first 4: v1={v1_vals[:4]} v2={v2_vals[:4]})"
        _emit_status(f"{label}...", "FAILED", summary)
        _warn(f"outputs differ ({summary}) — speedup may reflect a bug, not an optimization", aggregate_warnings)
    else:
        _emit_status(f"{label}...", "ok", f"max_diff={max_diff:.1e}")


def _dump_output_from_binary(binary: str, env: dict | None, dump_count: int = 16) -> list[float]:
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    try:
        result = subprocess.run(
            [binary, "--dump-output", str(dump_count)],
            capture_output=True,
            text=True,
            env=run_env,
        )
    except Exception:
        return []

    values: list[float] = []
    for line in (result.stdout or "").splitlines():
        try:
            values.append(float(line.strip()))
        except ValueError:
            continue
    return values


def _run_correctness_check(
    args,
    binary_a: str,
    binary_b: str,
    backend_a,
    binary_env: dict | None,
    result_a,
    result_b,
    aggregate_warnings: list[str],
    backend_b=None,
) -> None:
    """Run correctness check — routes through dump path for Triton, verify_correctness for CUDA."""
    def _is_persistent(b):
        return b is not None and hasattr(b, "is_persistent") and b.is_persistent()

    persistent_a = _is_persistent(backend_a)
    persistent_b = _is_persistent(backend_b)

    if persistent_a or persistent_b:
        v1_vals = result_a.output_vals
        v2_vals = result_b.output_vals
        if not v1_vals and not persistent_a:
            v1_vals = _dump_output_from_binary(binary_a, binary_env)
        if not v2_vals and not persistent_b:
            v2_vals = _dump_output_from_binary(binary_b, binary_env)
        if not v1_vals or not v2_vals:
            _emit_status("correctness check...", "skipped", "(dump returned no values)")
            return
        diffs = [_safe_diff(a, b) for a, b in zip(v1_vals, v2_vals)]
        max_diff = max(diffs)
        _emit_correctness_result(max_diff, v1_vals, v2_vals, args.tol, "correctness check", aggregate_warnings)
    else:
        max_diff, v1_vals, v2_vals = verify_correctness(binary_a, binary_b, env=binary_env)
        _emit_correctness_result(max_diff, v1_vals, v2_vals, args.tol, "correctness check", aggregate_warnings)


def _profile_variant(
    *,
    label: str,
    mock_prefix: str,
    binary: str,
    kernel_name: str,
    args,
    physical_gpu_id: int,
    hardware,
    binary_env: dict | None,
    backend,
    pipeline: int,
    dump_output: bool = False,
    show_progress: bool = False,
):
    result = profile(
        binary, kernel_name,
        max_runs=args.max_runs, min_runs=args.min_runs,
        noise_threshold=args.noise_threshold, warmup=args.warmup,
        gpu_id=physical_gpu_id, hardware=hardware,
        mock=args.mock, mock_prefix=mock_prefix, env=binary_env,
        pipeline=pipeline, backend=backend,
        dump_output=dump_output,
        show_progress=show_progress,
        progress_label=label,
    )
    _emit_status(
        f"profiling {label} {kernel_name}...",
        "ok",
        f"{result.actual_runs} runs  {result.min_latency_us:.0f}us  cv={result.cv_pct:.1f}%",
    )
    return result


def _run_single_kernel(
    args,
    kernel_name: str,
    file_a: str,
    file_b: str,
    hardware,
    physical_gpu_id: int,
    binary_env: dict | None,
    noise_floor: float,
    aggregate_warnings: list[str],
    use_color: bool,
    buf_elems: int | None = None,
    nvml_peak_bw: float | None = None,
    git_display_label: str | None = None,
) -> str:
    elems = buf_elems if buf_elems is not None else args.elems
    pipeline = getattr(args, "pipeline", 1)

    backend_a = None
    backend_b = None
    if not args.mock:
        try:
            backend_a = dispatch_backend(file_a)
        except SystemExit:
            raise
        try:
            backend_b = dispatch_backend(file_b)
        except SystemExit:
            raise

    _emit_status(f"compiling {kernel_name}...", "ok")
    if not args.mock:
        binary_a = backend_a.compile(file_a, kernel_name, arch=args.arch, dtype=args.dtype, buf_elems=elems, call_expr=args.call_expr)
        binary_b = backend_b.compile(file_b, kernel_name, arch=args.arch, dtype=args.dtype, buf_elems=elems, call_expr=args.call_expr)
    else:
        binary_a = file_a
        binary_b = file_b

    do_correctness = getattr(args, "correctness", False)
    if do_correctness and args.mock:
        _emit_status("correctness check...", "skipped", "(mock mode)")
        do_correctness = False

    # Also dump output for auto-correctness (two different, non-mock, non-git files)
    _auto_check = file_a != file_b and not args.mock and git_display_label is None
    _want_dump = do_correctness or _auto_check
    if (
        args.warmup < 25
        and (
            (backend_a is not None and backend_a.__class__.__name__ == "TritonBackend")
            or (backend_b is not None and backend_b.__class__.__name__ == "TritonBackend")
        )
    ):
        _warn(
            f"warmup={args.warmup} may be insufficient for Triton kernels (JIT compilation takes ~100-500ms). "
            "Consider --warmup 100 or higher.",
            aggregate_warnings,
        )

    show_progress = (
        ((args.format == "term") or bool(getattr(args, "export_json", None)))
        and not _SUPPRESS_STDERR
        and sys.stderr.isatty()
    )

    result_a = _profile_variant(
        label="v1",
        mock_prefix="v1",
        binary=binary_a,
        kernel_name=kernel_name,
        args=args,
        physical_gpu_id=physical_gpu_id,
        hardware=hardware,
        binary_env=binary_env,
        backend=backend_a,
        pipeline=pipeline,
        dump_output=_want_dump,
        show_progress=show_progress,
    )

    result_b = _profile_variant(
        label="v2",
        mock_prefix="v2",
        binary=binary_b,
        kernel_name=kernel_name,
        args=args,
        physical_gpu_id=physical_gpu_id,
        hardware=hardware,
        binary_env=binary_env,
        backend=backend_b,
        pipeline=pipeline,
        dump_output=_want_dump,
        show_progress=show_progress,
    )

    if do_correctness:
        _run_correctness_check(
            args, binary_a, binary_b, backend_a, binary_env,
            result_a, result_b, aggregate_warnings, backend_b=backend_b,
        )

    for warning in result_a.warnings + result_b.warnings:
        _warn(warning, aggregate_warnings)

    if _auto_check and not do_correctness:
        v1_vals = result_a.output_vals
        v2_vals = result_b.output_vals
        if v1_vals and v2_vals:
            auto_diffs = [_safe_diff(a, b) for a, b in zip(v1_vals, v2_vals)]
            auto_max_diff = max(auto_diffs)
            import math
            if math.isnan(auto_max_diff) or auto_max_diff > 1e-2:
                _emit_status("auto-correctness...", "warning",
                             f"max_diff={auto_max_diff:.3g} (outputs differ — verify kernels compute same function)")
            else:
                _emit_status("auto-correctness...", "ok", f"max_diff={auto_max_diff:.1e}")

    if args.mock:
        ptx_a = ptx.load_fixture("v1")
        ptx_b = ptx.load_fixture("v2")
    else:
        try:
            ptx_a = backend_a.extract_ptx(file_a, arch=args.arch)
        except Exception:
            ptx_a = {}
        try:
            ptx_b = backend_b.extract_ptx(file_b, arch=args.arch)
        except Exception:
            ptx_b = {}
    result_a.ptx_instructions = ptx_a
    result_b.ptx_instructions = ptx_b

    deltas = sort_deltas(compute_all_deltas(result_a.metrics, result_b.metrics, noise_floor=noise_floor))
    verdict = compute_verdict(result_a, result_b, noise_floor=noise_floor)
    if verdict.get("speedup_uncertainty_x", 0.0) >= 0.5:
        _warn("high uncertainty — consider --noise-threshold or clock locking", aggregate_warnings)
    roofline_v1 = compute_roofline(
        hardware.gpu_name,
        result_a.metrics.get("dram_bw_gbs", 0.0),
        result_a.metrics.get("sm_throughput", 0.0),
        nvml_peak_bw=nvml_peak_bw,
    )
    roofline = compute_roofline(
        hardware.gpu_name,
        result_b.metrics.get("dram_bw_gbs", 0.0),
        result_b.metrics.get("sm_throughput", 0.0),
        nvml_peak_bw=nvml_peak_bw,
    )
    ptx_diff = ptx.diff_ptx(ptx_a, ptx_b)

    # Derive total_hbm_gb for pipeline mode
    total_hbm = None
    if pipeline > 1:
        bw_a = result_a.metrics.get("dram_bw_gbs", 0.0)
        bw_b = result_b.metrics.get("dram_bw_gbs", 0.0)
        lat_a = result_a.min_latency_us
        lat_b = result_b.min_latency_us
        if bw_a > 0 and bw_b > 0:
            gb_a = (bw_a * lat_a) / 1e6
            gb_b = (bw_b * lat_b) / 1e6
            total_hbm = (gb_a, gb_b)

    # Compute noise ceiling for ? annotations
    clocks_locked = hardware.clocks_locked
    noise_ceiling = max(result_a.cv_pct, result_b.cv_pct) * 2.0 if not clocks_locked else 0.0

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
            buf_elems=elems,
            l2_flush=result_a.l2_flush or result_b.l2_flush,
            verdict=verdict,
            deltas=deltas,
            roofline=roofline,
            roofline_v1_bw=roofline_v1.bw_utilization,
            ptx_diff=ptx_diff,
            warnings=aggregate_warnings,
            total_hbm=total_hbm,
            pipeline=pipeline,
            v1_profile=result_a,
            v2_profile=result_b,
        )
        return render_json(payload)
    else:
        sections = [
            render_verdict(verdict, use_color=use_color, clocks_locked=clocks_locked),
            render_metric_table(
                deltas,
                result_a,
                result_b,
                roofline=roofline,
                roofline_v1=roofline_v1,
                roofline_v1_bw=roofline_v1.bw_utilization,
                roofline_v1_compute=roofline_v1.compute_utilization,
                use_color=use_color,
                total_hbm=total_hbm,
                noise_ceiling=noise_ceiling,
            ),
        ]
        ptx_section = render_ptx_diff(ptx_diff)
        if ptx_section:
            sections.append(ptx_section)
        return "\n".join(sections)


def _run_validate(
    args,
    kernel_name: str,
    file_a: str,
    file_b: str,
    hardware,
    physical_gpu_id: int,
    binary_env: dict | None,
    noise_floor: float,
    aggregate_warnings: list[str],
) -> str:
    """Run forward and reverse profiling, check consistency."""
    pipeline = getattr(args, "pipeline", 1)

    if not args.mock:
        binary_a = compile_kernel(file_a, kernel_name, arch=args.arch, mock=False, kernel_call=args.call_expr, dtype=args.dtype, buf_elems=args.elems)
        binary_b = compile_kernel(file_b, kernel_name, arch=args.arch, mock=False, kernel_call=args.call_expr, dtype=args.dtype, buf_elems=args.elems)
    else:
        binary_a = file_a
        binary_b = file_b

    # Forward: v1 then v2
    _emit_status("validate forward...", "")
    fwd_a = profile(binary_a, kernel_name, max_runs=args.max_runs, min_runs=args.min_runs,
                     noise_threshold=args.noise_threshold, warmup=args.warmup,
                     gpu_id=physical_gpu_id, hardware=hardware,
                     mock=args.mock, mock_prefix="v1", env=binary_env, pipeline=pipeline)
    fwd_b = profile(binary_b, kernel_name, max_runs=args.max_runs, min_runs=args.min_runs,
                     noise_threshold=args.noise_threshold, warmup=args.warmup,
                     gpu_id=physical_gpu_id, hardware=hardware,
                     mock=args.mock, mock_prefix="v2", env=binary_env, pipeline=pipeline)

    # Reverse: v2 then v1
    _emit_status("validate reverse...", "")
    rev_b = profile(binary_b, kernel_name, max_runs=args.max_runs, min_runs=args.min_runs,
                     noise_threshold=args.noise_threshold, warmup=args.warmup,
                     gpu_id=physical_gpu_id, hardware=hardware,
                     mock=args.mock, mock_prefix="v2", env=binary_env, pipeline=pipeline)
    rev_a = profile(binary_a, kernel_name, max_runs=args.max_runs, min_runs=args.min_runs,
                     noise_threshold=args.noise_threshold, warmup=args.warmup,
                     gpu_id=physical_gpu_id, hardware=hardware,
                     mock=args.mock, mock_prefix="v1", env=binary_env, pipeline=pipeline)

    for w in fwd_a.warnings + fwd_b.warnings + rev_a.warnings + rev_b.warnings:
        _warn(w, aggregate_warnings)

    fwd_speedup = fwd_a.min_latency_us / fwd_b.min_latency_us if fwd_b.min_latency_us else 0
    rev_speedup = rev_a.min_latency_us / rev_b.min_latency_us if rev_b.min_latency_us else 0

    delta = abs(fwd_speedup - rev_speedup)
    delta_pct = delta / max(fwd_speedup, 0.001) * 100

    if delta_pct > 5.0:
        return f"  validate: WARN  (forward: {fwd_speedup:.2f}x, reverse: {rev_speedup:.2f}x — inconsistent, rerun with locked clocks)"
    else:
        return f"  validate: ok  (forward: {fwd_speedup:.2f}x, reverse: {rev_speedup:.2f}x, delta: {delta_pct:.1f}%)"


def _run_shape_sweep(
    args,
    kernel_name: str,
    file_a: str,
    file_b: str,
    hardware,
    physical_gpu_id: int,
    binary_env: dict | None,
    noise_floor: float,
    aggregate_warnings: list[str],
    shapes: list[int],
    nvml_peak_bw: float | None = None,
) -> str:
    """Run the diff at each shape and produce a summary table."""
    rows = []
    pipeline = getattr(args, "pipeline", 1)
    for shape in shapes:
        if not _SUPPRESS_STDERR:
            print(f"\n  --- shape={shape} ---", file=sys.stderr)
        if not args.mock:
            binary_a = compile_kernel(file_a, kernel_name, arch=args.arch, mock=False, kernel_call=args.call_expr, dtype=args.dtype, buf_elems=shape)
            binary_b = compile_kernel(file_b, kernel_name, arch=args.arch, mock=False, kernel_call=args.call_expr, dtype=args.dtype, buf_elems=shape)
        else:
            binary_a = file_a
            binary_b = file_b

        result_a = _profile_variant(
            label="v1",
            mock_prefix="v1",
            binary=binary_a,
            kernel_name=kernel_name,
            args=args,
            physical_gpu_id=physical_gpu_id,
            hardware=hardware,
            binary_env=binary_env,
            backend=None,
            pipeline=pipeline,
        )
        result_b = _profile_variant(
            label="v2",
            mock_prefix="v2",
            binary=binary_b,
            kernel_name=kernel_name,
            args=args,
            physical_gpu_id=physical_gpu_id,
            hardware=hardware,
            binary_env=binary_env,
            backend=None,
            pipeline=pipeline,
        )
        for w in result_a.warnings + result_b.warnings:
            _warn(w, aggregate_warnings)

        speedup = result_a.min_latency_us / result_b.min_latency_us if result_b.min_latency_us else 0
        dram_a = result_a.metrics.get("dram_bw_gbs", 0.0)
        dram_b = result_b.metrics.get("dram_bw_gbs", 0.0)
        dram_delta = ((dram_b - dram_a) / dram_a * 100) if dram_a > 0 else 0.0
        roofline_a = compute_roofline(hardware.gpu_name, dram_a, result_a.metrics.get("sm_throughput", 0.0), nvml_peak_bw=nvml_peak_bw)
        roofline_b = compute_roofline(hardware.gpu_name, dram_b, result_b.metrics.get("sm_throughput", 0.0), nvml_peak_bw=nvml_peak_bw)
        bound_a = roofline_a.bound[:3] if roofline_a.gpu_matched else "?"
        bound_b = roofline_b.bound[:3] if roofline_b.gpu_matched else "?"
        bound_str = f"{bound_a}->{bound_b}" if bound_a != bound_b else bound_b

        rows.append({
            "shape": shape,
            "v1_us": result_a.min_latency_us,
            "v2_us": result_b.min_latency_us,
            "speedup": speedup,
            "dram_delta": dram_delta,
            "bound": bound_str,
        })

    return render_shape_table(rows)


def main(argv: list[str] | None = None) -> int:
    global _SUPPRESS_STDERR
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.export_json:
        args.format = "json"
        args.output = args.export_json
    prev_suppress = _SUPPRESS_STDERR
    _SUPPRESS_STDERR = args.format == "json" and not bool(args.export_json)
    try:
        config_path = find_config()
        config = {}
        if config_path:
            config = load_config(config_path)
            if config and not _SUPPRESS_STDERR:
                print(f"  config: {config_path}", file=sys.stderr)

        # Apply config defaults (before kernel-specific, which needs fn resolution)
        if config:
            args = apply_config(args, config)

        if args.fn_name and args.all_kernels:
            raise SystemExit("error: --fn and --all are mutually exclusive")

        if args.pipeline > 1 and not args.call_expr:
            raise SystemExit("error: --pipeline requires --call")

        if len(args.files) not in {1, 2}:
            parser.error("expected one or two .cu or .py files")

        if args.elems <= 0:
            raise SystemExit("error: --elems must be > 0")

        if args.min_runs > args.max_runs:
            raise SystemExit(f"error: --min-runs ({args.min_runs}) cannot exceed --max-runs ({args.max_runs})")

        file_a = args.files[0]
        file_b = args.files[1] if len(args.files) == 2 else None

        if args.at_ref is not None and file_b is not None:
            raise SystemExit("error: --at only applies to single-file (git) mode")

        use_color = not args.no_color and args.format == "term" and sys.stdout.isatty()
        aggregate_warnings: list[str] = []

        shapes: list[int] | None = None
        if args.shape:
            try:
                shapes = [int(s.strip()) for s in args.shape.split(",")]
            except ValueError:
                raise SystemExit("error: --shape values must be positive integers (e.g. 1024,2048,4096)")
            if any(s <= 0 for s in shapes):
                raise SystemExit("error: --shape values must be > 0")

        if args.watch:
            return _watch_loop(args, argv)

        return _run_main(args, file_a, file_b, use_color, aggregate_warnings, shapes, config)
    finally:
        _SUPPRESS_STDERR = prev_suppress


def _run_main(args, file_a, file_b, use_color, aggregate_warnings, shapes, config=None):
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.mock:
        hardware = MOCK_HARDWARE
        if not _SUPPRESS_STDERR:
            print(f"  gpu: {hardware.gpu_name} (mock)", file=sys.stderr)
        _warn("mock mode -- no GPU required.", aggregate_warnings)
        physical_gpu_id = args.gpu
        binary_env: dict | None = None
    else:
        if cuda_visible is not None:
            if args.gpu != 0:
                _warn(
                    f"CUDA_VISIBLE_DEVICES={cuda_visible} is set; --gpu {args.gpu} ignored.",
                    aggregate_warnings,
                )
            try:
                physical_gpu_id = int(cuda_visible.split(",")[0])
            except (ValueError, IndexError):
                physical_gpu_id = 0
            binary_env = None
        else:
            physical_gpu_id = args.gpu
            binary_env = {"CUDA_VISIBLE_DEVICES": str(args.gpu)}

        hardware = query_hardware(gpu_id=physical_gpu_id)
        if args.arch == "sm_90":
            detected_arch = detect_sm_arch(hardware.gpu_name)
            if detected_arch:
                args.arch = detected_arch
        state = "clocks locked" if hardware.clocks_locked else "clocks unlocked"
        if not _SUPPRESS_STDERR:
            print(f"  gpu: {hardware.gpu_name} (device {physical_gpu_id}, {args.arch}, {state})", file=sys.stderr)

    nvml_peak_bw = None
    if not args.mock:
        nvml_peak_bw = query_peak_bandwidth_nvml(physical_gpu_id)

    git_display_label = None
    if file_b is None:
        if args.mock:
            file_b = file_a
            file_a = "fixtures/v1_ncu.csv"
        else:
            at_ref = args.at_ref or "HEAD"
            file_a, git_display_label = resolve_git_baseline(args.files[0], at_ref=at_ref)
            file_b = args.files[0]

    if git_display_label:
        if not _SUPPRESS_STDERR:
            print(f"  comparing: {git_display_label}  vs  {args.files[0]} (working copy)", file=sys.stderr)

    if not args.mock:
        for path in (file_a, file_b):
            if not Path(path).exists():
                raise SystemExit(f"error: file not found: {path}")

    if args.all_kernels:
        if not Path(file_a).exists() or not Path(file_b).exists():
            raise SystemExit("error: --all requires both files to exist on disk")
        kernel_names = resolve_all_kernels(file_a, file_b)
    else:
        kernel_name = args.fn_name
        if kernel_name is None and Path(file_a).exists() and Path(file_b).exists():
            kernel_name = resolve_kernel_name(file_a, file_b, None)
        elif kernel_name is None:
            raise SystemExit("error: could not auto-detect kernel — please specify --fn")
        kernel_names = [kernel_name]

    # Apply kernel-specific config overrides now that we know the kernel name
    if config and kernel_names:
        args = apply_config(args, config, kernel_name=kernel_names[0])

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

    if shapes is not None:
        try:
            final_output = _run_shape_sweep(
                args, kernel_names[0], file_a, file_b,
                hardware, physical_gpu_id, binary_env,
                noise_floor, aggregate_warnings, shapes,
                nvml_peak_bw=nvml_peak_bw,
            )
        finally:
            if not args.mock:
                unlock_clocks(physical_gpu_id)
        if args.output:
            write_output(final_output, args.output)
        else:
            print(final_output)
        return 0

    try:
        outputs = []
        for kname in kernel_names:
            output = _run_single_kernel(
                args, kname, file_a, file_b,
                hardware, physical_gpu_id, binary_env,
                noise_floor, aggregate_warnings, use_color,
                nvml_peak_bw=nvml_peak_bw,
                git_display_label=git_display_label,
            )
            if len(kernel_names) > 1:
                outputs.append(f"\n=== {kname} ===\n{output}")
            else:
                outputs.append(output)

        if getattr(args, "validate", False):
            for kname in kernel_names:
                validate_line = _run_validate(
                    args, kname, file_a, file_b,
                    hardware, physical_gpu_id, binary_env,
                    noise_floor, aggregate_warnings,
                )
                outputs.append(validate_line)

        final_output = "\n".join(outputs)
    finally:
        if not args.mock:
            unlock_clocks(physical_gpu_id)

    if args.output:
        write_output(final_output, args.output)
    else:
        print(final_output)
    return 0


def _watch_loop(args, argv: list[str] | None) -> int:
    """Watch .cu files for changes and re-run the diff."""
    files = args.files[:2] if len(args.files) >= 2 else args.files[:1]
    paths = [os.path.abspath(f) for f in files]

    def get_mtimes():
        return tuple(os.path.getmtime(p) if os.path.exists(p) else 0 for p in paths)

    # Build argv without --watch for re-runs
    rerun_argv = [a for a in (argv or sys.argv[1:]) if a != "--watch"]

    last_mtimes = (0,) * len(paths)
    print("  watching for changes... (Ctrl-C to stop)", file=sys.stderr)

    try:
        while True:
            mtimes = get_mtimes()
            if mtimes != last_mtimes:
                last_mtimes = mtimes
                ts = time.strftime("%H:%M:%S")
                print("\033[2J\033[H", end="", flush=True)
                print(f"[{ts}] re-profiling...", file=sys.stderr)
                try:
                    _run_from_argv(rerun_argv)
                except SystemExit:
                    pass
                except Exception as e:
                    print(f"error: {e}", file=sys.stderr)
                print(f"\n  watching for changes... (Ctrl-C to stop)", file=sys.stderr)
            time.sleep(0.5)
    except KeyboardInterrupt:
        print("\n  stopped.", file=sys.stderr)
        return 0


def _run_from_argv(argv: list[str]) -> int:
    """Re-run main with fresh args (for watch mode)."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    file_a = args.files[0]
    file_b = args.files[1] if len(args.files) == 2 else None
    use_color = not args.no_color and args.format == "term" and sys.stdout.isatty()
    shapes = None
    if args.shape:
        shapes = [int(s.strip()) for s in args.shape.split(",")]
    config_path = find_config()
    config = load_config(config_path) if config_path else {}
    if config:
        args = apply_config(args, config)
    return _run_main(args, file_a, file_b, use_color, [], shapes, config)


if __name__ == "__main__":
    raise SystemExit(main())
