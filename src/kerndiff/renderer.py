from __future__ import annotations

import json
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
    "yellow": "\033[33m",
    "red": "\033[31m",
}


def _color(text: str, code: str, use_color: bool) -> str:
    if not use_color or not code:
        return text
    return f"{ANSI[code]}{text}{ANSI['reset']}"


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
        label = _color(verdict["label"], "green", use_color)
        label = f"{ANSI['bold']}{label}{ANSI['reset']}" if use_color else label
    else:
        label = _color(verdict["label"], "red", use_color)
        label = f"{ANSI['bold']}{label}{ANSI['reset']}" if use_color else label
    cv_note = f"  (v1 ±{v1_cv:.0f}%  v2 ±{v2_cv:.0f}%)"
    line = f"  {label}  {v1_us:.1f}us → {v2_us:.1f}us{cv_note}"
    unc = verdict.get("speedup_uncertainty_x", 0.0)
    if unc >= 0.02:
        line += f"  ±{unc:.2f}x"
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

        rows.append((left, v1_text, v2_text, format_delta(delta), styled_symbol, delta.metric.group))

    name_w = max(22, len("metric"), *(len(row[0]) for row in rows))
    v1_w = max(14, len("v1"), *(len(row[1]) for row in rows))
    v2_w = max(14, len("v2"), *(len(row[2]) for row in rows))
    delta_w = max(10, len("delta"), *(len(row[3]) for row in rows))
    lines = [
        f"  {'metric':<{name_w}}  {'v1':>{v1_w}}  {'v2':>{v2_w}}  {'delta':>{delta_w}}",
        f"  {'─' * (name_w + v1_w + v2_w + delta_w + 6)}",
    ]
    current_group = None
    for left, v1_text, v2_text, delta_text, symbol, group in rows:
        if group != current_group and current_group is not None:
            lines.append("")  # blank line between groups
        current_group = group
        lines.append(f"  {left:<{name_w}}  {v1_text:>{v1_w}}  {v2_text:>{v2_w}}  {delta_text:>{delta_w}}  {symbol}")
    # Roofline row — skip if both bw and compute are 0 (no data)
    if roofline and roofline.gpu_matched:
        v1_bw = roofline_v1_bw if roofline_v1_bw is not None else 0.0
        v1_compute = roofline_v1_compute if roofline_v1_compute is not None else 0.0
        v2_bw = roofline.bw_utilization
        v2_compute = roofline.compute_utilization
        has_data = (v1_bw > 0 or v1_compute > 0 or v2_bw > 0 or v2_compute > 0)
        if has_data:
            lines.append(f"  {'─' * (name_w + v1_w + v2_w + delta_w + 6)}")
            v1_bound = "memory" if v1_bw > v1_compute else "compute"
            v2_bound = roofline.bound
            if v1_bound != v2_bound:
                bound_text = f"bound: {v1_bound[:3]}->{v2_bound[:3]}"
            else:
                bound_text = f"bound: {v2_bound}"
            v1_text = f"v1: {v1_bw * 100:.0f}%bw" if v1_bound == "memory" else f"v1: {v1_compute * 100:.0f}%sm"
            v2_text = f"v2: {v2_bw * 100:.0f}%bw" if v2_bound == "memory" else f"v2: {v2_compute * 100:.0f}%sm"
            tail = f"{bound_text}  {roofline.headroom_pct:.0f}% headroom"
            if getattr(roofline, "bw_source", "unknown") == "table":
                tail += "  [spec]"
            lines.append(
                f"  {'roofline':<{name_w}}  "
                f"{v1_text:>{v1_w}}  "
                f"{v2_text:>{v2_w}}  "
                f"{tail:>{delta_w}}"
            )
    # Total HBM row (pipeline mode only)
    if total_hbm is not None:
        gb_a, gb_b = total_hbm
        delta_pct = ((gb_b - gb_a) / gb_a * 100) if gb_a > 0 else 0.0
        lines.append(
            f"  {'total_hbm':<{name_w}}  "
            f"{f'{gb_a:.3f}GB':>{v1_w}}  "
            f"{f'{gb_b:.3f}GB':>{v2_w}}  "
            f"{f'{delta_pct:+.1f}%':>{delta_w}}"
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
        },
        "v2": {
            "file": file_b,
            "latencies_us": v2_profile.all_latencies_us if v2_profile else [],
            "clean_latencies_us": v2_profile.clean_latencies_us if v2_profile else [],
            "n_outliers": v2_profile.n_outliers if v2_profile else 0,
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
        "warnings": warnings,
    }
    if roofline and roofline.gpu_matched:
        payload["roofline"] = {
            "bound": roofline.bound,
            "v1_bw_utilization": roofline_v1_bw if roofline_v1_bw is not None else roofline.bw_utilization,
            "v2_bw_utilization": roofline.bw_utilization,
            "headroom_pct": roofline.headroom_pct,
            "gpu_matched": roofline.gpu_matched,
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
