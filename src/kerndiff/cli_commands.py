from __future__ import annotations

import math
import sys
import time

from kerndiff import cli_state, ptx
from kerndiff.compiler import check_determinism, compile_kernel, infer_kernel_call
from kerndiff.diff import compute_all_deltas, compute_verdict, sort_deltas
from kerndiff.profiler import (
    interleave_timing,
    interleave_timing_persistent,
    profile,
    query_l2_size,
)
from kerndiff.renderer import (
    build_json_payload,
    render_json,
    render_metric_table,
    render_perfetto_trace,
    render_ptx_diff,
    render_verdict,
    render_shape_table,
    write_output,
)
from kerndiff.roofline import compute_roofline
from kerndiff.suggestions import generate_suggestions


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
    runtime,
    pipeline: int,
    dump_output: bool = False,
    show_progress: bool = False,
    pre_collected_latencies: list[float] | None = None,
):
    result = profile(
        binary,
        kernel_name,
        max_runs=args.max_runs,
        min_runs=args.min_runs,
        noise_threshold=args.noise_threshold,
        warmup=args.warmup,
        gpu_id=physical_gpu_id,
        hardware=hardware,
        mock=args.mock,
        mock_prefix=mock_prefix,
        env=binary_env,
        pipeline=pipeline,
        backend=runtime,
        dump_output=dump_output,
        show_progress=show_progress,
        progress_label=label,
        pre_collected_latencies=pre_collected_latencies,
        run_timeout_sec=getattr(args, "timeout", 0),
    )
    cli_state.emit_status(
        f"profiling {label} {kernel_name}...",
        "ok",
        f"{result.actual_runs} runs  {result.min_latency_us:.0f}us  cv={result.cv_pct:.1f}%",
    )
    return result


def _runtime_name(runtime) -> str:
    return runtime.__class__.__name__ if runtime is not None else ""


def _warn_inferred_call(
    runtime,
    source_path: str,
    kernel_name: str,
    elems: int,
    aggregate_warnings: list[str],
) -> None:
    runtime_name = _runtime_name(runtime)
    if runtime_name == "CUDABackend":
        inferred_call, inference_warnings, inferred_mode = infer_kernel_call(source_path, kernel_name)
        detail = f"using {inferred_mode} CUDA launch for {kernel_name}: {inferred_call}"
        if inference_warnings:
            detail += f" ({'; '.join(inference_warnings)})"
        cli_state.warn(f"{detail} — pass --call to make the launch explicit.", aggregate_warnings)
    elif runtime_name == "TritonBackend" and hasattr(runtime, "default_call_expr"):
        inferred_call = runtime.default_call_expr(kernel_name, elems)
        cli_state.warn(
            f"using default Triton launch for {kernel_name}: {inferred_call} — pass --call to make it explicit.",
            aggregate_warnings,
        )


def run_single_kernel(
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
    correctness_checker,
    buf_elems: int | None = None,
    nvml_peak_bw: float | None = None,
    git_display_label: str | None = None,
) -> str:
    elems = buf_elems if buf_elems is not None else args.elems
    pipeline = getattr(args, "pipeline", 1)

    runtime_a = None
    runtime_b = None
    if not args.mock:
        from kerndiff.runtimes import dispatch as dispatch_runtime

        runtime_a = dispatch_runtime(file_a)
        runtime_b = dispatch_runtime(file_b)
        if args.call_expr is None:
            _warn_inferred_call(runtime_a, file_a, kernel_name, elems, aggregate_warnings)

    cli_state.emit_status(f"compiling {kernel_name}...", "ok")
    if not args.mock:
        binary_a = runtime_a.compile(file_a, kernel_name, arch=args.arch, dtype=args.dtype, buf_elems=elems, call_expr=args.call_expr)
        binary_b = runtime_b.compile(file_b, kernel_name, arch=args.arch, dtype=args.dtype, buf_elems=elems, call_expr=args.call_expr)
    else:
        binary_a = file_a
        binary_b = file_b

    do_correctness = getattr(args, "correctness", False)
    if do_correctness and args.mock:
        cli_state.emit_status("correctness check...", "skipped", "(mock mode)")
        do_correctness = False

    auto_check = file_a != file_b and not args.mock and git_display_label is None
    want_dump = do_correctness or auto_check
    if (
        args.warmup < 25
        and (
            _runtime_name(runtime_a) == "TritonBackend"
            or _runtime_name(runtime_b) == "TritonBackend"
        )
    ):
        cli_state.warn(
            f"warmup={args.warmup} may be insufficient for Triton kernels (JIT compilation takes ~100-500ms). "
            "Consider --warmup 100 or higher.",
            aggregate_warnings,
        )

    show_progress = (
        ((args.format == "term") or bool(getattr(args, "export_json", None)))
        and not cli_state.SUPPRESS_STDERR
        and sys.stderr.isatty()
    )

    both_cuda = (
        runtime_a is not None
        and runtime_b is not None
        and _runtime_name(runtime_a) == "CUDABackend"
        and _runtime_name(runtime_b) == "CUDABackend"
    )
    both_persistent = (
        runtime_a is not None
        and runtime_b is not None
        and getattr(runtime_a, "is_persistent", lambda: False)()
        and getattr(runtime_b, "is_persistent", lambda: False)()
        and hasattr(runtime_a, "_last_compile_args")
        and hasattr(runtime_b, "_last_compile_args")
    )
    use_interleaved = not args.mock and (both_cuda or both_persistent)

    pre_latencies_a: list[float] | None = None
    pre_latencies_b: list[float] | None = None
    run_timeout = getattr(args, "timeout", 0)
    if use_interleaved:
        if _runtime_name(runtime_a) == "CUDABackend":
            pre_latencies_a, pre_latencies_b, timing_warnings = interleave_timing(
                binary_a,
                binary_b,
                kernel_name,
                min_runs=args.min_runs,
                max_runs=args.max_runs,
                noise_threshold=args.noise_threshold,
                warmup=args.warmup,
                gpu_id=physical_gpu_id,
                hardware=hardware,
                env=binary_env,
                show_progress=show_progress,
                run_timeout_sec=run_timeout,
            )
        else:
            l2_size = query_l2_size(physical_gpu_id, gpu_name=hardware.gpu_name)
            a_args = runtime_a._last_compile_args
            b_args = runtime_b._last_compile_args
            timed_a = runtime_a.compile_timed(
                a_args["source_path"],
                a_args["kernel_name"],
                a_args["arch"],
                a_args["dtype"],
                a_args["buf_elems"],
                a_args["call_expr"],
                iters=1,
                l2_flush_bytes=l2_size,
                warmup=args.warmup,
            )
            timed_b = runtime_b.compile_timed(
                b_args["source_path"],
                b_args["kernel_name"],
                b_args["arch"],
                b_args["dtype"],
                b_args["buf_elems"],
                b_args["call_expr"],
                iters=1,
                l2_flush_bytes=l2_size,
                warmup=args.warmup,
            )
            pre_latencies_a, pre_latencies_b, timing_warnings = interleave_timing_persistent(
                runtime_a,
                timed_a,
                runtime_b,
                timed_b,
                kernel_name,
                min_runs=args.min_runs,
                max_runs=args.max_runs,
                noise_threshold=args.noise_threshold,
                gpu_id=physical_gpu_id,
                hardware=hardware,
                env=binary_env,
                show_progress=show_progress,
            )

        cli_state.emit_status("timing (interleaved)...", "ok", f"{len(pre_latencies_a)} pairs")
        for warning in timing_warnings:
            cli_state.warn(warning, aggregate_warnings)

    result_a = _profile_variant(
        label="v1",
        mock_prefix="v1",
        binary=binary_a,
        kernel_name=kernel_name,
        args=args,
        physical_gpu_id=physical_gpu_id,
        hardware=hardware,
        binary_env=binary_env,
        runtime=runtime_a,
        pipeline=pipeline,
        dump_output=want_dump,
        show_progress=show_progress,
        pre_collected_latencies=pre_latencies_a,
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
        runtime=runtime_b,
        pipeline=pipeline,
        dump_output=want_dump,
        show_progress=show_progress,
        pre_collected_latencies=pre_latencies_b,
    )

    if do_correctness:
        correctness_checker(
            args,
            binary_a,
            binary_b,
            backend_a=runtime_a,
            binary_env=binary_env,
            result_a=result_a,
            result_b=result_b,
            aggregate_warnings=aggregate_warnings,
            backend_b=runtime_b,
        )

    for warning in result_a.warnings + result_b.warnings:
        cli_state.warn(warning, aggregate_warnings)

    mhz_a = result_a.metrics.get("actual_sm_mhz", 0.0)
    mhz_b = result_b.metrics.get("actual_sm_mhz", 0.0)
    if mhz_a > 0 and mhz_b > 0:
        clock_diff_pct = abs(mhz_a - mhz_b) / max(mhz_a, mhz_b) * 100
        if clock_diff_pct > 10:
            cli_state.warn(
                f"GPU SM clock differed between v1 ({mhz_a:.0f} MHz) and "
                f"v2 ({mhz_b:.0f} MHz) NCU runs ({clock_diff_pct:.0f}% gap) — "
                f"rate metrics may not be directly comparable; lock clocks for accuracy.",
                aggregate_warnings,
            )

    if getattr(args, "determinism", False) and not args.mock:
        for label, binary in (("v1", binary_a), ("v2", binary_b)):
            is_det, max_diff = check_determinism(binary, n_runs=3, env=binary_env)
            if not is_det:
                cli_state.emit_status(
                    f"determinism ({label})...",
                    "warning",
                    f"max_diff={max_diff:.3g} across 3 runs — kernel output is non-deterministic",
                )

    if auto_check and not do_correctness:
        v1_vals = result_a.output_vals
        v2_vals = result_b.output_vals
        if v1_vals and v2_vals:
            auto_diffs = [abs(a - b) for a, b in zip(v1_vals, v2_vals)]
            auto_max_diff = max(auto_diffs)
            if math.isnan(auto_max_diff) or auto_max_diff > 1e-2:
                cli_state.emit_status(
                    "auto-correctness...",
                    "warning",
                    f"max_diff={auto_max_diff:.3g} (outputs differ — verify kernels compute same function)",
                )
            else:
                cli_state.emit_status("auto-correctness...", "ok", f"max_diff={auto_max_diff:.1e}")

    if args.mock:
        ptx_a = ptx.load_fixture("v1")
        ptx_b = ptx.load_fixture("v2")
    else:
        try:
            ptx_a = runtime_a.extract_ptx(file_a, arch=args.arch)
        except Exception as exc:
            ptx_a = {}
            cli_state.warn(f"PTX extraction failed for v1 ({exc}). Static PTX diff will be incomplete.", aggregate_warnings)
        try:
            ptx_b = runtime_b.extract_ptx(file_b, arch=args.arch)
        except Exception as exc:
            ptx_b = {}
            cli_state.warn(f"PTX extraction failed for v2 ({exc}). Static PTX diff will be incomplete.", aggregate_warnings)
    result_a.ptx_instructions = ptx_a
    result_b.ptx_instructions = ptx_b

    deltas = sort_deltas(compute_all_deltas(result_a.metrics, result_b.metrics, noise_floor=noise_floor))
    verdict = compute_verdict(
        result_a,
        result_b,
        noise_floor=noise_floor,
        paired_latencies_a=pre_latencies_a,
        paired_latencies_b=pre_latencies_b,
    )
    uncertainty = verdict.get("speedup_uncertainty_x", 0.0)
    speedup = abs(verdict.get("speedup", 1.0))
    if uncertainty >= 0.5 and (speedup < 0.01 or uncertainty / speedup >= 0.1):
        cli_state.warn("high uncertainty — consider --noise-threshold or clock locking", aggregate_warnings)

    roofline_v1 = compute_roofline(
        hardware.gpu_name,
        result_a.metrics.get("dram_bw_gbs", 0.0),
        result_a.metrics.get("sm_throughput", 0.0),
        nvml_peak_bw=nvml_peak_bw,
        arith_intensity=result_a.metrics.get("arith_intensity", 0.0),
        tensor_core_util=result_a.metrics.get("tensor_core_util", 0.0),
    )
    roofline = compute_roofline(
        hardware.gpu_name,
        result_b.metrics.get("dram_bw_gbs", 0.0),
        result_b.metrics.get("sm_throughput", 0.0),
        nvml_peak_bw=nvml_peak_bw,
        arith_intensity=result_b.metrics.get("arith_intensity", 0.0),
        tensor_core_util=result_b.metrics.get("tensor_core_util", 0.0),
    )
    ptx_diff = ptx.diff_ptx(ptx_a, ptx_b)

    total_hbm = None
    if pipeline > 1:
        bw_a = result_a.metrics.get("dram_bw_gbs", 0.0)
        bw_b = result_b.metrics.get("dram_bw_gbs", 0.0)
        if bw_a > 0 and bw_b > 0:
            total_hbm = ((bw_a * result_a.min_latency_us) / 1e6, (bw_b * result_b.min_latency_us) / 1e6)

    clocks_locked = hardware.clocks_locked
    noise_ceiling = max(result_a.cv_pct, result_b.cv_pct) * 2.0 if not clocks_locked else 0.0

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

    if getattr(args, "export_perfetto", None):
        write_output(render_perfetto_trace(payload), args.export_perfetto)

    if args.format == "json":
        return render_json(payload)

    sections = [
        render_verdict(verdict, use_color=use_color, clocks_locked=clocks_locked),
        "",
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
    if getattr(args, "ptx", False):
        ptx_section = render_ptx_diff(ptx_diff)
        if ptx_section:
            sections.append(ptx_section)

    if getattr(args, "suggestions", True):
        hints = generate_suggestions(deltas, result_b.metrics)
        if hints:
            sections.append("")
            sections.append("  suggestions")
            for hint in hints:
                sections.append(f"    · {hint}")

    if aggregate_warnings:
        sections.append("")
        sections.extend(cli_state.color_warn(warning, use_color) for warning in aggregate_warnings)

    output = "\n".join(sections)

    cli_state.WATCH_HISTORY.append(
        {
            "ts": time.strftime("%H:%M:%S"),
            "speedup": verdict.get("speedup", 1.0),
            "v1_us": verdict.get("v1_latency_us", 0.0),
            "v2_us": verdict.get("v2_latency_us", 0.0),
        }
    )
    if len(cli_state.WATCH_HISTORY) > 5:
        cli_state.WATCH_HISTORY.pop(0)

    return output


def run_validate(
    args,
    kernel_name: str,
    file_a: str,
    file_b: str,
    hardware,
    physical_gpu_id: int,
    binary_env: dict | None,
    aggregate_warnings: list[str],
) -> str:
    pipeline = getattr(args, "pipeline", 1)

    if not args.mock:
        binary_a = compile_kernel(file_a, kernel_name, arch=args.arch, mock=False, kernel_call=args.call_expr, dtype=args.dtype, buf_elems=args.elems)
        binary_b = compile_kernel(file_b, kernel_name, arch=args.arch, mock=False, kernel_call=args.call_expr, dtype=args.dtype, buf_elems=args.elems)
    else:
        binary_a = file_a
        binary_b = file_b

    cli_state.emit_status("validate forward...", "")
    fwd_a = profile(binary_a, kernel_name, max_runs=args.max_runs, min_runs=args.min_runs, noise_threshold=args.noise_threshold, warmup=args.warmup, gpu_id=physical_gpu_id, hardware=hardware, mock=args.mock, mock_prefix="v1", env=binary_env, pipeline=pipeline)
    fwd_b = profile(binary_b, kernel_name, max_runs=args.max_runs, min_runs=args.min_runs, noise_threshold=args.noise_threshold, warmup=args.warmup, gpu_id=physical_gpu_id, hardware=hardware, mock=args.mock, mock_prefix="v2", env=binary_env, pipeline=pipeline)

    cli_state.emit_status("validate reverse...", "")
    rev_b = profile(binary_b, kernel_name, max_runs=args.max_runs, min_runs=args.min_runs, noise_threshold=args.noise_threshold, warmup=args.warmup, gpu_id=physical_gpu_id, hardware=hardware, mock=args.mock, mock_prefix="v2", env=binary_env, pipeline=pipeline)
    rev_a = profile(binary_a, kernel_name, max_runs=args.max_runs, min_runs=args.min_runs, noise_threshold=args.noise_threshold, warmup=args.warmup, gpu_id=physical_gpu_id, hardware=hardware, mock=args.mock, mock_prefix="v1", env=binary_env, pipeline=pipeline)

    for warning in fwd_a.warnings + fwd_b.warnings + rev_a.warnings + rev_b.warnings:
        cli_state.warn(warning, aggregate_warnings)

    fwd_speedup = fwd_a.min_latency_us / fwd_b.min_latency_us if fwd_b.min_latency_us else 0
    rev_speedup = rev_a.min_latency_us / rev_b.min_latency_us if rev_b.min_latency_us else 0
    delta_pct = abs(fwd_speedup - rev_speedup) / max(fwd_speedup, 0.001) * 100

    if delta_pct > 5.0:
        return f"  validate: WARN  (forward: {fwd_speedup:.2f}x, reverse: {rev_speedup:.2f}x — inconsistent, rerun with locked clocks)"
    return f"  validate: ok  (forward: {fwd_speedup:.2f}x, reverse: {rev_speedup:.2f}x, delta: {delta_pct:.1f}%)"


def run_shape_sweep(
    args,
    kernel_name: str,
    file_a: str,
    file_b: str,
    hardware,
    physical_gpu_id: int,
    binary_env: dict | None,
    aggregate_warnings: list[str],
    shapes: list[int],
    nvml_peak_bw: float | None = None,
) -> str:
    from kerndiff.diff import _pairwise_stats

    rows = []
    pipeline = getattr(args, "pipeline", 1)
    run_timeout = getattr(args, "timeout", 0)
    for shape in shapes:
        if not cli_state.SUPPRESS_STDERR:
            print(f"\n  --- shape={shape} ---", file=sys.stderr)
        if not args.mock:
            binary_a = compile_kernel(file_a, kernel_name, arch=args.arch, mock=False, kernel_call=args.call_expr, dtype=args.dtype, buf_elems=shape)
            binary_b = compile_kernel(file_b, kernel_name, arch=args.arch, mock=False, kernel_call=args.call_expr, dtype=args.dtype, buf_elems=shape)
        else:
            binary_a = file_a
            binary_b = file_b

        sweep_lats_a: list[float] | None = None
        sweep_lats_b: list[float] | None = None
        if not args.mock:
            sweep_lats_a, sweep_lats_b, _ = interleave_timing(
                binary_a,
                binary_b,
                kernel_name,
                min_runs=args.min_runs,
                max_runs=args.max_runs,
                noise_threshold=args.noise_threshold,
                warmup=args.warmup,
                gpu_id=physical_gpu_id,
                hardware=hardware,
                env=binary_env,
                run_timeout_sec=run_timeout,
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
            runtime=None,
            pipeline=pipeline,
            pre_collected_latencies=sweep_lats_a,
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
            runtime=None,
            pipeline=pipeline,
            pre_collected_latencies=sweep_lats_b,
        )
        for warning in result_a.warnings + result_b.warnings:
            cli_state.warn(warning, aggregate_warnings)

        pairwise = _pairwise_stats(sweep_lats_a, sweep_lats_b) if sweep_lats_a and sweep_lats_b else None
        speedup = pairwise[0] if pairwise else (result_a.min_latency_us / result_b.min_latency_us if result_b.min_latency_us else 0)
        dram_a = result_a.metrics.get("dram_bw_gbs", 0.0)
        dram_b = result_b.metrics.get("dram_bw_gbs", 0.0)
        dram_delta = ((dram_b - dram_a) / dram_a * 100) if dram_a > 0 else 0.0
        roofline_a = compute_roofline(hardware.gpu_name, dram_a, result_a.metrics.get("sm_throughput", 0.0), nvml_peak_bw=nvml_peak_bw, arith_intensity=result_a.metrics.get("arith_intensity", 0.0), tensor_core_util=result_a.metrics.get("tensor_core_util", 0.0))
        roofline_b = compute_roofline(hardware.gpu_name, dram_b, result_b.metrics.get("sm_throughput", 0.0), nvml_peak_bw=nvml_peak_bw, arith_intensity=result_b.metrics.get("arith_intensity", 0.0), tensor_core_util=result_b.metrics.get("tensor_core_util", 0.0))
        bound_a = roofline_a.bound[:3] if roofline_a.gpu_matched else "?"
        bound_b = roofline_b.bound[:3] if roofline_b.gpu_matched else "?"
        rows.append(
            {
                "shape": shape,
                "v1_us": result_a.min_latency_us,
                "v2_us": result_b.min_latency_us,
                "speedup": speedup,
                "dram_delta": dram_delta,
                "bound": f"{bound_a}->{bound_b}" if bound_a != bound_b else bound_b,
            }
        )

    return render_shape_table(rows)
