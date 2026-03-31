from __future__ import annotations

import json
from statistics import mean
from pathlib import Path

from kerndiff.diff import MetricDelta
from kerndiff.metrics import fmt_int, fmt_kb
from kerndiff.profiler import ProfileResult
from kerndiff.roofline import RooflineResult

ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "green": "\033[32m",
    "bright_green": "\033[92m",
    "yellow": "\033[33m",
    "red": "\033[31m",
    "bright_red": "\033[91m",
    "cyan": "\033[36m",
}


def _color(text: str, code: str, use_color: bool) -> str:
    if not use_color or not code:
        return text
    return f"{ANSI[code]}{text}{ANSI['reset']}"


def _dim(text: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{ANSI['dim']}{text}{ANSI['reset']}"


def _bold(text: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{ANSI['bold']}{text}{ANSI['reset']}"


def _style_symbol(symbol: str, use_color: bool) -> str:
    if not use_color:
        return symbol
    if symbol == "++":
        return f"{ANSI['bold']}{ANSI['green']}{symbol}{ANSI['reset']}"
    if symbol == "+":
        return f"{ANSI['green']}{symbol}{ANSI['reset']}"
    if symbol == "~":
        return symbol
    if symbol == "-":
        return f"{ANSI['red']}{symbol}{ANSI['reset']}"
    if symbol == "--":
        return f"{ANSI['bold']}{ANSI['red']}{symbol}{ANSI['reset']}"
    return symbol


def format_delta(delta: MetricDelta) -> str:
    metric = delta.metric
    if metric.unit == "%":
        return f"{delta.v2 - delta.v1:+.1f}pp"
    if metric.unit in {"us", "GB/s", "inst", "count", "F/B", "TF"}:
        return f"{delta.delta_pct:+.1f}%"
    # unit == "int" or "B": show raw integer diff
    raw_diff = delta.v2 - delta.v1
    if metric.key == "shared_mem_kb":
        # NCU reports shared memory in bytes; table displays KB, so delta should align.
        raw_diff /= 1024.0
    if metric.unit == "B":
        value = fmt_kb(abs(raw_diff))
    else:
        value = fmt_int(abs(raw_diff))
    sign = "+" if raw_diff >= 0 else "-"
    return f"{sign}{value}"


def render_verdict(verdict: dict, use_color: bool = True, clocks_locked: bool = True) -> str:
    v1_us = verdict["v1_latency_us"]
    v2_us = verdict["v2_latency_us"]
    v1_cv = verdict["v1_cv_pct"]
    v2_cv = verdict["v2_cv_pct"]
    if verdict["direction"] == "unchanged":
        noise_note = f"  (±{max(v1_cv, v2_cv):.0f}% noise)" if not clocks_locked else ""
        text = f"  no significant change  {v1_us:.1f}us vs {v2_us:.1f}us{noise_note}"
        return _color(text, "dim", use_color)
    if verdict["direction"] == "improvement":
        label = f"{ANSI['bold']}{ANSI['bright_green']}{verdict['label']}{ANSI['reset']}" if use_color else verdict["label"]
    else:
        label = f"{ANSI['bold']}{ANSI['bright_red']}{verdict['label']}{ANSI['reset']}" if use_color else verdict["label"]
    cv_note = _dim(f"  (v1 ±{v1_cv:.0f}%  v2 ±{v2_cv:.0f}%)", use_color)
    arrow = _dim("→", use_color)
    timing = f"  {v1_us:.1f}us {arrow} {v2_us:.1f}us"
    line = f"  {label}{timing}{cv_note}"
    unc = verdict.get("speedup_uncertainty_x", 0.0)
    if unc >= 0.02:
        unc_note = f"  ±{unc:.2f}x"
        if verdict.get("paired_uncertainty"):
            unc_note += " (paired)"
        line += _dim(unc_note, use_color)
    if not clocks_locked:
        noise_ceil = max(v1_cv, v2_cv) * 2.0
        line += f"\n  note: clocks not locked — deltas below {noise_ceil:.0f}% may not be reliable"
    return line


def render_metric_table(
    deltas: list[MetricDelta],
    v1: ProfileResult,
    v2: ProfileResult,
    roofline: RooflineResult | None = None,
    roofline_v1: RooflineResult | None = None,
    roofline_v1_bw: float | None = None,
    roofline_v1_compute: float | None = None,
    use_color: bool = True,
    total_hbm: tuple[float, float] | None = None,
    noise_ceiling: float = 0.0,
) -> str:
    rows = []
    for delta in deltas:
        left = delta.metric.display_name
        v1_text = delta.metric.format_fn(delta.v1)
        v2_text = delta.metric.format_fn(delta.v2)
        if delta.metric.key == "latency_us":
            v1_text = f"{v1_text} ±{v1.cv_pct:.0f}%"
            v2_text = f"{v2_text} ±{v2.cv_pct:.0f}%"

        symbol = delta.symbol
        styled_symbol = _style_symbol(symbol, use_color)

        # Add ? for uncertain deltas (not latency — it has ±cv% already)
        uncertain = False
        if noise_ceiling > 0 and delta.metric.key != "latency_us":
            if abs(delta.delta_pct) < noise_ceiling and symbol not in ("~",):
                uncertain = True

        if uncertain:
            question = _color("?", "dim", use_color)
            styled_symbol = f"{styled_symbol} {question}"

        # Color v2 value: green if favorable improvement, red if unfavorable
        v2_colored = v2_text
        if use_color and symbol in ("+", "++"):
            v2_colored = _color(v2_text, "bright_green", use_color)
        elif use_color and symbol in ("-", "--"):
            v2_colored = _color(v2_text, "bright_red", use_color)

        rows.append((left, v1_text, v2_colored, v2_text, format_delta(delta), styled_symbol, delta.metric.group))

    if not rows:
        return ""

    name_w = max(18, *(len(row[0]) for row in rows))
    v1_w = max(12, *(len(row[1]) for row in rows))
    v2_w = max(12, *(len(row[3]) for row in rows))  # raw v2 for width
    delta_w = max(8, *(len(row[4]) for row in rows))
    # symbol column is 2 wide (++/--) plus 2 padding = 4; but we reserve 4 after delta
    total_w = name_w + v1_w + v2_w + delta_w + 6
    sep = f"  {'─' * total_w}"

    # ANSI-aware padding: raw is the printable string for width calc, styled may have codes
    def lpad(styled: str, raw: str, w: int) -> str:
        return styled + " " * (w - len(raw))

    def rpad(styled: str, raw: str, w: int) -> str:
        return " " * (w - len(raw)) + styled

    lines: list[str] = []
    current_group = None
    group_labels = {
        "sol": "throughput",
        "arithmetic": "arithmetic",
        "cache": "cache",
        "warp_state": "warp state",
        "launch": "launch config",
    }
    for left, v1_text, v2_colored, v2_text, delta_text, symbol, group in rows:
        if group != current_group:
            if current_group is not None:
                lines.append("")
            label = group_labels.get(group, group)
            dashes = "─" * max(0, total_w - len(label) - 4)
            lines.append(_dim(f"  ── {label} {dashes}", use_color))
            current_group = group
        lines.append(
            f"  {lpad(_dim(left, use_color), left, name_w)}"
            f"  {rpad(v1_text, v1_text, v1_w)}"
            f"  {rpad(v2_colored, v2_text, v2_w)}"
            f"  {rpad(delta_text, delta_text, delta_w)}"
            f"  {symbol}"
        )
    # Roofline row — skip if both bw and compute are 0 (no data)
    if roofline and roofline.gpu_matched:
        v1_bw = roofline_v1_bw if roofline_v1_bw is not None else 0.0
        v1_compute = roofline_v1_compute if roofline_v1_compute is not None else 0.0
        v2_bw = roofline.bw_utilization
        v2_compute = roofline.compute_utilization
        has_data = (v1_bw > 0 or v1_compute > 0 or v2_bw > 0 or v2_compute > 0)
        if has_data:
            lines.append(_dim(sep, use_color))
            v1_bound = roofline_v1.bound if (roofline_v1 and roofline_v1.gpu_matched) else (
                "memory" if v1_bw > v1_compute else "compute"
            )
            v2_bound = roofline.bound
            if v1_bound != v2_bound:
                bound_text = f"bound: {v1_bound[:3]}->{v2_bound[:3]}"
            else:
                bound_text = f"bound: {v2_bound}"
            v1_text = f"v1: {v1_bw * 100:.0f}%bw" if v1_bound == "memory" else f"v1: {v1_compute * 100:.0f}%sm"
            v2_text = f"v2: {v2_bw * 100:.0f}%bw" if v2_bound == "memory" else f"v2: {v2_compute * 100:.0f}%sm"
            tail = f"{bound_text}  {roofline.headroom_pct:.0f}% headroom"
            if getattr(roofline, "used_tensor_core_peak", False):
                tail += "  [fp16]"
            elif getattr(roofline, "bw_source", "unknown") == "table":
                tail += "  [spec]"
            lines.append(
                f"  {lpad(_dim('roofline', use_color), 'roofline', name_w)}"
                f"  {rpad(v1_text, v1_text, v1_w)}"
                f"  {rpad(v2_text, v2_text, v2_w)}"
                f"  {rpad(tail, tail, delta_w)}"
            )
    # Total HBM row (pipeline mode only)
    if total_hbm is not None:
        gb_a, gb_b = total_hbm
        delta_pct = ((gb_b - gb_a) / gb_a * 100) if gb_a > 0 else 0.0
        s_a, s_b, s_d = f"{gb_a:.3f}GB", f"{gb_b:.3f}GB", f"{delta_pct:+.1f}%"
        lines.append(
            f"  {lpad(_dim('total_hbm', use_color), 'total_hbm', name_w)}"
            f"  {rpad(s_a, s_a, v1_w)}"
            f"  {rpad(s_b, s_b, v2_w)}"
            f"  {rpad(s_d, s_d, delta_w)}"
        )
    return "\n".join(lines)


def render_ptx_diff(rows: list[dict]) -> str:
    if not rows:
        return ""
    name_w = max(18, *(len(r["instruction"]) for r in rows))
    v1_w = max(6, *(len(str(r["v1"])) for r in rows))
    v2_w = max(6, *(len(str(r["v2"])) for r in rows))
    delta_w = max(10, *(len(f"{r['delta_pct']:+.1f}%") for r in rows))
    lines = [
        "  ptx diff  (static instruction count — not dynamic execution count)",
        f"  {'─' * (name_w + v1_w + v2_w + delta_w + 6)}",
        f"  {'instruction':<{name_w}}  {'v1':>{v1_w}}  {'v2':>{v2_w}}  {'delta':>{delta_w}}",
    ]
    for row in rows:
        lines.append(
            f"  {row['instruction']:<{name_w}}  {row['v1']:>{v1_w}}  {row['v2']:>{v2_w}}  {row['delta_pct']:+{delta_w}.1f}%"
        )
    return "\n".join(lines)


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


def _build_histogram(values: list[float], bins: int = 12) -> list[dict[str, float | int]]:
    if not values:
        return []
    lo = min(values)
    hi = max(values)
    if hi <= lo:
        return [{"start_us": lo, "end_us": hi, "count": len(values)}]

    bucket_count = max(1, min(bins, len(values)))
    width = (hi - lo) / bucket_count
    counts = [0 for _ in range(bucket_count)]
    for value in values:
        idx = min(int((value - lo) / width), bucket_count - 1)
        counts[idx] += 1

    histogram: list[dict[str, float | int]] = []
    for idx, count in enumerate(counts):
        start = lo + idx * width
        end = hi if idx == bucket_count - 1 else start + width
        histogram.append({"start_us": start, "end_us": end, "count": count})
    return histogram


def _build_latency_summary(profile: ProfileResult | None) -> dict:
    if profile is None:
        return {
            "runs": 0,
            "clean_runs": 0,
            "mean_us": 0.0,
            "min_us": 0.0,
            "p20_us": 0.0,
            "p50_us": 0.0,
            "p80_us": 0.0,
            "p95_us": 0.0,
            "max_us": 0.0,
            "cv_pct": 0.0,
            "outliers": 0,
        }

    values = profile.clean_latencies_us or profile.all_latencies_us
    return {
        "runs": len(profile.all_latencies_us),
        "clean_runs": len(values),
        "mean_us": mean(values) if values else 0.0,
        "min_us": min(values) if values else 0.0,
        "p20_us": _percentile(values, 0.2),
        "p50_us": _percentile(values, 0.5),
        "p80_us": _percentile(values, 0.8),
        "p95_us": _percentile(values, 0.95),
        "max_us": max(values) if values else 0.0,
        "cv_pct": profile.cv_pct,
        "outliers": profile.n_outliers,
    }


def _has_hardware_counters(profile: ProfileResult | None) -> bool:
    if profile is None:
        return False
    return any(key != "latency_us" for key in profile.metrics)


def _build_instrumentation_status(profile: ProfileResult | None) -> dict:
    if profile is None:
        return {
            "ncu_metrics_available": False,
            "ptx_available": False,
            "clock_telemetry_available": False,
            "clock_telemetry_source": "missing",
            "latency_only": True,
        }
    telemetry = profile.clock_telemetry or {}
    has_counters = _has_hardware_counters(profile)
    return {
        "ncu_metrics_available": has_counters,
        "ptx_available": bool(profile.ptx_instructions),
        "clock_telemetry_available": bool(telemetry.get("available", False)),
        "clock_telemetry_source": telemetry.get("source", "missing"),
        "latency_only": not has_counters,
    }


def _profile_trace_duration(profile: ProfileResult | None) -> float:
    if profile is None or not profile.trace_events:
        return 0.0
    return max((event["ts_us"] + event["dur_us"]) for event in profile.trace_events)


def _build_trace_artifact(
    v1_profile: ProfileResult | None,
    v2_profile: ProfileResult | None,
    gap_us: float = 1000.0,
) -> dict:
    tracks: list[dict] = []
    events: list[dict] = []
    phase_summary: list[dict] = []
    offset_us = 0.0

    for variant, profile in (("v1", v1_profile), ("v2", v2_profile)):
        if profile is None:
            continue
        track_names: dict[str, str] = {}
        for event in profile.trace_events:
            lane = event["lane"]
            if lane not in track_names:
                track_name = f"{variant} {lane}"
                track_id = f"{variant}:{lane}"
                track_names[lane] = track_id
                tracks.append({
                    "id": track_id,
                    "name": track_name,
                    "variant": variant,
                    "lane": lane,
                })

            ts_us = offset_us + event["ts_us"]
            dur_us = event["dur_us"]
            events.append({
                "name": event["name"],
                "category": event["category"],
                "variant": variant,
                "lane": lane,
                "track_id": track_names[lane],
                "track_name": f"{variant} {lane}",
                "ts_us": ts_us,
                "dur_us": dur_us,
                "args": event.get("args", {}),
            })
            if lane == "phases":
                phase_summary.append({
                    "variant": variant,
                    "name": event["name"],
                    "dur_us": dur_us,
                    "args": event.get("args", {}),
                })

        offset_us += _profile_trace_duration(profile) + gap_us

    return {
        "schema": "kerndiff_host_trace_v1",
        "clock": "relative_us",
        "tracks": tracks,
        "events": events,
        "phase_summary": phase_summary,
    }


def build_json_payload(
    *,
    hardware,
    kernel_name: str,
    file_a: str,
    file_b: str,
    actual_runs: int,
    max_runs: int,
    min_runs: int,
    noise_threshold: float,
    warmup: int,
    buf_elems: int = 1 << 22,
    l2_flush: bool,
    verdict: dict,
    deltas: list[MetricDelta],
    roofline: RooflineResult | None,
    roofline_v1_bw: float | None = None,
    ptx_diff: list[dict],
    warnings: list[str],
    total_hbm: tuple[float, float] | None = None,
    pipeline: int = 1,
    v1_profile: ProfileResult | None = None,
    v2_profile: ProfileResult | None = None,
) -> dict:
    payload = {
        "hardware": {
            "gpu": hardware.gpu_name,
            "sm_clock_mhz": hardware.sm_clock_mhz,
            "mem_clock_mhz": hardware.mem_clock_mhz,
            "driver": hardware.driver_version,
            "clocks_locked": hardware.clocks_locked,
            "mock": hardware.mock,
        },
        "kernel": kernel_name,
        "v1": {
            "file": file_a,
            "latencies_us": v1_profile.all_latencies_us if v1_profile else [],
            "clean_latencies_us": v1_profile.clean_latencies_us if v1_profile else [],
            "n_outliers": v1_profile.n_outliers if v1_profile else 0,
            "summary": _build_latency_summary(v1_profile),
            "histogram": _build_histogram(v1_profile.clean_latencies_us if v1_profile else []),
            "instrumentation": _build_instrumentation_status(v1_profile),
            "clock_telemetry": v1_profile.clock_telemetry if v1_profile else {},
        },
        "v2": {
            "file": file_b,
            "latencies_us": v2_profile.all_latencies_us if v2_profile else [],
            "clean_latencies_us": v2_profile.clean_latencies_us if v2_profile else [],
            "n_outliers": v2_profile.n_outliers if v2_profile else 0,
            "summary": _build_latency_summary(v2_profile),
            "histogram": _build_histogram(v2_profile.clean_latencies_us if v2_profile else []),
            "instrumentation": _build_instrumentation_status(v2_profile),
            "clock_telemetry": v2_profile.clock_telemetry if v2_profile else {},
        },
        "actual_runs": actual_runs,
        "max_runs": max_runs,
        "min_runs": min_runs,
        "noise_threshold": noise_threshold,
        "warmup": warmup,
        "config": {"buf_elems": buf_elems, "pipeline": pipeline},
        "l2_flush": l2_flush,
        "verdict": {k: verdict[k] for k in [
            "speedup", "direction", "label", "v1_latency_us", "v2_latency_us",
            "v1_min_us", "v1_max_us", "v1_cv_pct",
            "v1_p20_us", "v1_p50_us", "v1_p80_us", "v1_n_outliers",
            "v2_min_us", "v2_max_us", "v2_cv_pct",
            "v2_p20_us", "v2_p50_us", "v2_p80_us", "v2_n_outliers",
            "speedup_uncertainty_x",
            "paired_uncertainty",
        ]},
        "deltas": [
            {
                "metric": delta.metric.key,
                "ncu_metric": delta.metric.ncu_metric,
                "group": delta.metric.group,
                "lower_is_better": delta.metric.lower_is_better,
                "v1": delta.v1,
                "v2": delta.v2,
                "delta_pct": delta.delta_pct,
                "symbol": delta.symbol,
            }
            for delta in deltas
        ],
        "ptx_diff": {
            "note": "static instruction count — not dynamic execution count",
            "instructions": ptx_diff,
        },
        "trace": _build_trace_artifact(v1_profile, v2_profile),
        "warnings": warnings,
    }
    if roofline and roofline.gpu_matched:
        payload["roofline"] = {
            "bound": roofline.bound,
            "v1_bw_utilization": roofline_v1_bw if roofline_v1_bw is not None else roofline.bw_utilization,
            "v2_bw_utilization": roofline.bw_utilization,
            "headroom_pct": roofline.headroom_pct,
            "gpu_matched": roofline.gpu_matched,
            "ridge_point": roofline.ridge_point,
            "used_tensor_core_peak": roofline.used_tensor_core_peak,
        }
    if total_hbm is not None:
        payload["total_hbm"] = {"v1_gb": total_hbm[0], "v2_gb": total_hbm[1]}
    return payload


def write_output(text: str, output_file: str | None) -> None:
    if output_file:
        Path(output_file).write_text(text)
    else:
        print(text)


def render_shape_table(rows: list[dict]) -> str:
    """Render a shape sweep summary table.

    Each row has: shape, v1_us, v2_us, speedup, dram_delta, bound.
    """
    header = f"  {'shape':>10}  {'v1 (us)':>10}  {'v2 (us)':>10}  {'speedup':>8}  {'dram':>8}  {'bound':>10}"
    sep = f"  {'-' * 64}"
    lines = [header, sep]
    for r in rows:
        sp = f"{r['speedup']:.2f}x"
        dram = f"{r['dram_delta']:+.1f}%"
        lines.append(
            f"  {r['shape']:>10}  {r['v1_us']:>10.1f}  {r['v2_us']:>10.1f}  {sp:>8}  {dram:>8}  {r['bound']:>10}"
        )
    return "\n".join(lines)


def render_json(payload: dict) -> str:
    return json.dumps(payload, indent=2)


def render_perfetto_trace(payload: dict) -> str:
    trace = payload.get("trace", {})
    tracks = trace.get("tracks", [])
    events = trace.get("events", [])

    trace_events: list[dict] = []
    pid = 1
    track_tid: dict[str, int] = {}

    for idx, track in enumerate(tracks, start=1):
        track_tid[track["id"]] = idx
        trace_events.append({
            "name": "thread_name",
            "ph": "M",
            "pid": pid,
            "tid": idx,
            "args": {"name": track["name"]},
        })

    for event in events:
        trace_events.append({
            "name": event["name"],
            "cat": event["category"],
            "ph": "X",
            "pid": pid,
            "tid": track_tid.get(event["track_id"], 0),
            "ts": round(event["ts_us"], 3),
            "dur": round(event["dur_us"], 3),
            "args": {
                "variant": event.get("variant"),
                **event.get("args", {}),
            },
        })

    return json.dumps({
        "displayTimeUnit": "us",
        "traceEvents": trace_events,
    }, indent=2)
