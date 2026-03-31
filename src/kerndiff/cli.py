from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

from kerndiff import cli_state
from kerndiff.cli_commands import run_shape_sweep, run_single_kernel, run_validate
from kerndiff.cli_support import (
    _scan_kernels,
    detect_sm_arch,
    resolve_all_kernels,
    resolve_git_baseline,
    resolve_kernel_name,
    resolve_kernel_selection,
)
from kerndiff.compiler import verify_correctness
from kerndiff.config import apply_config, find_config, load_config
from kerndiff.diff import NOISE_FLOOR_LOCKED, NOISE_FLOOR_UNLOCKED
from kerndiff.profiler import MOCK_HARDWARE, lock_clocks, query_hardware, query_peak_bandwidth_nvml, unlock_clocks
from kerndiff.renderer import write_output

_emit_status = cli_state.emit_status
_warn = cli_state.warn
_color_warn = cli_state.color_warn
_run_single_kernel = run_single_kernel
_run_validate = run_validate
_run_shape_sweep = run_shape_sweep


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
    parser.add_argument(
        "--export-perfetto",
        dest="export_perfetto",
        metavar="FILE",
        help="write a Perfetto/Chrome trace JSON file for host-side profiling phases and timing samples",
    )
    parser.add_argument("--ptx", action="store_true", help="show PTX instruction diff (hidden by default)")
    parser.add_argument("--mock", action="store_true")
    parser.add_argument("--no-color", action="store_true")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--arch", default="sm_90")
    parser.add_argument(
        "--timeout", type=int, default=0, metavar="SEC",
        help="per-run timeout in seconds; 0 = no timeout (default: 0)",
    )
    parser.add_argument(
        "--determinism", action="store_true",
        help="run each kernel 3× and warn if outputs are non-deterministic",
    )
    parser.add_argument(
        "--suggestions", action="store_true", default=True,
        help="show actionable optimization suggestions (default: on)",
    )
    parser.add_argument(
        "--no-suggestions", dest="suggestions", action="store_false",
        help="suppress optimization suggestions",
    )
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
    backend_a=None,
    binary_env: dict | None = None,
    result_a=None,
    result_b=None,
    aggregate_warnings: list[str] | None = None,
    backend_b=None,
    # legacy aliases kept for internal callers
    runtime_a=None,
    runtime_b=None,
) -> None:
    """Run correctness check — routes through dump path for Triton, verify_correctness for CUDA."""
    if aggregate_warnings is None:
        aggregate_warnings = []
    # Accept either backend_a or runtime_a (legacy name)
    effective_a = backend_a if backend_a is not None else runtime_a
    effective_b = backend_b if backend_b is not None else runtime_b

    def _is_persistent(b):
        return b is not None and hasattr(b, "is_persistent") and b.is_persistent()

    persistent_a = _is_persistent(effective_a)
    persistent_b = _is_persistent(effective_b)

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


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.export_json:
        args.format = "json"
        args.output = args.export_json
    prev_suppress = cli_state.SUPPRESS_STDERR
    cli_state.SUPPRESS_STDERR = args.format == "json" and not bool(args.export_json)
    try:
        config_path = find_config()
        config = {}
        if config_path:
            config = load_config(config_path)
            if config and not cli_state.SUPPRESS_STDERR:
                print(f"  config: {config_path}", file=sys.stderr)

        # Apply config defaults (before kernel-specific, which needs fn resolution)
        if config:
            args = apply_config(args, config)

        if args.fn_name and args.all_kernels:
            raise SystemExit("error: --fn and --all are mutually exclusive")
        if args.export_perfetto and args.all_kernels:
            raise SystemExit("error: --export-perfetto does not support --all")
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
            if args.export_perfetto:
                raise SystemExit("error: --export-perfetto does not support --shape")
        if args.watch:
            return _watch_loop(args, argv)

        return _run_main(args, file_a, file_b, use_color, aggregate_warnings, shapes, config)
    finally:
        cli_state.SUPPRESS_STDERR = prev_suppress


def _run_main(args, file_a, file_b, use_color, aggregate_warnings, shapes, config=None):
    cli_state.DEFER_WARNINGS = args.format == "term"
    cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if args.mock:
        hardware = MOCK_HARDWARE
        if not cli_state.SUPPRESS_STDERR:
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
        if not cli_state.SUPPRESS_STDERR:
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
        if not cli_state.SUPPRESS_STDERR:
            print(f"  comparing: {git_display_label}  vs  {args.files[0]} (working copy)", file=sys.stderr)

    if not args.mock:
        for path in (file_a, file_b):
            if not Path(path).exists():
                raise SystemExit(f"error: file not found: {path}")

    if args.all_kernels and (not Path(file_a).exists() or not Path(file_b).exists()):
        raise SystemExit("error: --all requires both files to exist on disk")

    if args.fn_name is None and not args.all_kernels and not (Path(file_a).exists() and Path(file_b).exists()):
        raise SystemExit("error: could not auto-detect kernel — please specify --fn")

    selection = resolve_kernel_selection(file_a, file_b, args.fn_name, args.all_kernels)
    kernel_names = selection.names
    if selection.mode == "auto":
        _warn(f"auto-selected kernel '{kernel_names[0]}' — pass --fn to pin it explicitly.", aggregate_warnings)

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
            refreshed = query_hardware(gpu_id=physical_gpu_id)
            if refreshed.gpu_name != "unknown":
                refreshed.clocks_locked = True
                refreshed.mock = hardware.mock
                hardware = refreshed
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
                aggregate_warnings, shapes,
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
                correctness_checker=_run_correctness_check,
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
                    aggregate_warnings,
                )
                outputs.append(validate_line)

        final_output = "\n".join(outputs)
    finally:
        if not args.mock:
            unlock_clocks(physical_gpu_id)

    cli_state.DEFER_WARNINGS = False
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

                # F: three-run trend header — shows trajectory across iterations
                if len(cli_state.WATCH_HISTORY) >= 2:
                    print("  recent runs:", file=sys.stderr)
                    for h in cli_state.WATCH_HISTORY[-3:]:
                        arrow = "↑" if h["speedup"] > 1.05 else ("↓" if h["speedup"] < 0.95 else "~")
                        print(
                            f"    [{h['ts']}] {arrow} {h['speedup']:.2f}x  "
                            f"{h['v1_us']:.0f}→{h['v2_us']:.0f}µs",
                            file=sys.stderr,
                        )
                    print(file=sys.stderr)

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
