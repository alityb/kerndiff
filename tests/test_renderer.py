import io
import json
import re
from contextlib import redirect_stderr, redirect_stdout

from kerndiff.cli import main
from kerndiff.diff import compute_all_deltas, compute_verdict, sort_deltas
from kerndiff.metrics import METRICS_BY_KEY
from kerndiff.renderer import (
    build_json_payload,
    format_delta,
    render_perfetto_trace,
    render_metric_table,
    render_ptx_diff,
    render_verdict,
)
from kerndiff.roofline import RooflineResult


def test_no_color_strips_ansi(v1_result, v2_result):
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    table = render_metric_table(deltas, v1_result, v2_result, use_color=False)
    assert "\033[" not in table


def test_column_widths_expand(v1_result, v2_result):
    v2_result.metrics["shared_mem_kb"] = 9999999
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    table = render_metric_table(deltas, v1_result, v2_result, use_color=False)
    assert "9765KB" in table


def test_latency_row_has_cv(v1_result, v2_result):
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    table = render_metric_table(deltas, v1_result, v2_result, use_color=False)
    assert re.search(r"latency.*±\d+%.*±\d+%", table)


def test_verdict_line_has_range_for_improvement(v1_result, v2_result):
    verdict = compute_verdict(v1_result, v2_result)
    line = render_verdict(verdict, use_color=False)
    assert "±" in line


def test_verdict_line_omits_range_for_unchanged(v1_result):
    verdict = compute_verdict(v1_result, v1_result)
    line = render_verdict(verdict, use_color=False)
    assert "[v1:" not in line


def test_register_delta_is_raw_diff(v1_result, v2_result):
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    table = render_metric_table(deltas, v1_result, v2_result, use_color=False)
    assert "+8" in table
    assert "+12.5%" not in table


def test_format_delta_int_unit_is_raw_integer():
    from kerndiff.diff import MetricDelta
    delta = MetricDelta(
        metric=METRICS_BY_KEY["registers_per_thread"],
        v1=64,
        v2=72,
        delta_pct=12.5,
        favorable=False,
        symbol="-",
    )
    assert format_delta(delta) == "+8"


def test_shared_mem_delta_uses_kb_units(v1_result, v2_result):
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    table = render_metric_table(deltas, v1_result, v2_result, use_color=False)
    assert "+16" in table
    assert "+16384" not in table


def test_l2_delta_uses_pp(v1_result, v2_result):
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    table = render_metric_table(deltas, v1_result, v2_result, use_color=False)
    assert "+26.2pp" in table


def test_json_has_required_keys(v1_result, v2_result):
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    verdict = compute_verdict(v1_result, v2_result)
    payload = build_json_payload(
        hardware=v1_result.hardware,
        kernel_name=v1_result.kernel_name,
        file_a="v1.cu",
        file_b="v2.cu",
        actual_runs=20,
        max_runs=50,
        min_runs=10,
        noise_threshold=1.0,
        warmup=32,
        l2_flush=False,
        verdict=verdict,
        deltas=deltas,
        roofline=RooflineResult("memory", 0.72, 0.89, 11.0, True),
        ptx_diff=[],
        warnings=[],
    )
    encoded = json.dumps(payload)
    decoded = json.loads(encoded)
    for key in ["hardware", "verdict", "deltas", "ptx_diff", "warnings"]:
        assert key in decoded
    for key in [
        "v1_min_us", "v1_max_us", "v1_cv_pct", "v1_p20_us", "v1_p50_us", "v1_p80_us", "v1_n_outliers",
        "v2_min_us", "v2_max_us", "v2_cv_pct", "v2_p20_us", "v2_p50_us", "v2_p80_us", "v2_n_outliers",
        "speedup_uncertainty_x",
    ]:
        assert key in decoded["verdict"]
    assert "latencies_us" in decoded["v1"]
    assert "clean_latencies_us" in decoded["v1"]
    assert "n_outliers" in decoded["v1"]
    assert "summary" in decoded["v1"]
    assert "histogram" in decoded["v1"]
    assert "clock_telemetry" in decoded["v1"]
    assert "trace" in decoded


def test_json_latency_summary_has_percentiles(v1_result, v2_result):
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    verdict = compute_verdict(v1_result, v2_result)
    payload = build_json_payload(
        hardware=v1_result.hardware,
        kernel_name=v1_result.kernel_name,
        file_a="v1.cu",
        file_b="v2.cu",
        actual_runs=20,
        max_runs=50,
        min_runs=10,
        noise_threshold=1.0,
        warmup=32,
        l2_flush=False,
        verdict=verdict,
        deltas=deltas,
        roofline=RooflineResult("memory", 0.72, 0.89, 11.0, True),
        ptx_diff=[],
        warnings=[],
        v1_profile=v1_result,
        v2_profile=v2_result,
    )
    summary = payload["v1"]["summary"]
    assert summary["runs"] == len(v1_result.all_latencies_us)
    assert summary["clean_runs"] == len(v1_result.clean_latencies_us)
    assert summary["p95_us"] >= summary["p80_us"]
    assert payload["v1"]["histogram"]


def test_json_trace_has_tracks_and_phase_summary(v1_result, v2_result):
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    verdict = compute_verdict(v1_result, v2_result)
    payload = build_json_payload(
        hardware=v1_result.hardware,
        kernel_name=v1_result.kernel_name,
        file_a="v1.cu",
        file_b="v2.cu",
        actual_runs=20,
        max_runs=50,
        min_runs=10,
        noise_threshold=1.0,
        warmup=32,
        l2_flush=False,
        verdict=verdict,
        deltas=deltas,
        roofline=RooflineResult("memory", 0.72, 0.89, 11.0, True),
        ptx_diff=[],
        warnings=[],
        v1_profile=v1_result,
        v2_profile=v2_result,
    )
    trace = payload["trace"]
    assert trace["schema"] == "kerndiff_host_trace_v1"
    assert trace["tracks"]
    assert any(event["name"] == "timed_runs" for event in trace["events"])
    assert any(item["name"] == "warmup" for item in trace["phase_summary"])


def test_render_perfetto_trace_includes_metadata_and_events(v1_result, v2_result):
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    verdict = compute_verdict(v1_result, v2_result)
    payload = build_json_payload(
        hardware=v1_result.hardware,
        kernel_name=v1_result.kernel_name,
        file_a="v1.cu",
        file_b="v2.cu",
        actual_runs=20,
        max_runs=50,
        min_runs=10,
        noise_threshold=1.0,
        warmup=32,
        l2_flush=False,
        verdict=verdict,
        deltas=deltas,
        roofline=RooflineResult("memory", 0.72, 0.89, 11.0, True),
        ptx_diff=[],
        warnings=[],
        v1_profile=v1_result,
        v2_profile=v2_result,
    )
    trace = json.loads(render_perfetto_trace(payload))
    names = [event["name"] for event in trace["traceEvents"]]
    assert "thread_name" in names
    assert "timed_runs" in names
    assert "run_001" in names


def test_output_file_writes_and_stdout_empty(tmp_path):
    out_file = tmp_path / "out.txt"
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel", "--output", str(out_file)])
    assert rc == 0
    assert stdout.getvalue() == ""
    assert out_file.read_text()


def test_json_output_file_writes_and_stdout_empty(tmp_path):
    out_file = tmp_path / "out.json"
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main([
            "--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel",
            "--format", "json", "--output", str(out_file),
        ])
    assert rc == 0
    assert stdout.getvalue() == ""
    payload = json.loads(out_file.read_text())
    assert "verdict" in payload


def test_ptx_diff_absent_when_unchanged(v1_result):
    assert render_ptx_diff([]) == ""


def test_roofline_omitted_when_both_zero(v1_result, v2_result):
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    table = render_metric_table(
        deltas, v1_result, v2_result,
        roofline=RooflineResult("memory", 0.0, 0.0, 100.0, True),
        roofline_v1_bw=0.0,
        roofline_v1_compute=0.0,
        use_color=False,
    )
    assert "roofline" not in table


def test_roofline_shows_bound_transition(v1_result, v2_result):
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    table = render_metric_table(
        deltas, v1_result, v2_result,
        roofline=RooflineResult("compute", 0.3, 0.8, 20.0, True),
        roofline_v1_bw=0.7,
        roofline_v1_compute=0.2,
        use_color=False,
    )
    assert "mem->com" in table


def test_roofline_omitted_when_unmatched(v1_result, v2_result):
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    verdict = compute_verdict(v1_result, v2_result)
    payload = build_json_payload(
        hardware=v1_result.hardware,
        kernel_name=v1_result.kernel_name,
        file_a="v1.cu",
        file_b="v2.cu",
        actual_runs=20,
        max_runs=50,
        min_runs=10,
        noise_threshold=1.0,
        warmup=32,
        l2_flush=False,
        verdict=verdict,
        deltas=deltas,
        roofline=RooflineResult("unknown", 0.0, 0.0, 0.0, False),
        ptx_diff=[],
        warnings=[],
    )
    assert "roofline" not in payload


def test_total_hbm_row_in_metric_table(v1_result, v2_result):
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    table = render_metric_table(
        deltas, v1_result, v2_result,
        use_color=False,
        total_hbm=(0.123, 0.456),
    )
    assert "total_hbm" in table
    assert "0.123GB" in table
    assert "0.456GB" in table


def test_total_hbm_omitted_when_none(v1_result, v2_result):
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    table = render_metric_table(deltas, v1_result, v2_result, use_color=False)
    assert "total_hbm" not in table


def test_json_total_hbm_included(v1_result, v2_result):
    v1 = v1_result
    v2 = v2_result
    deltas = sort_deltas(compute_all_deltas(v1.metrics, v2.metrics))
    verdict = compute_verdict(v1, v2)
    payload = build_json_payload(
        hardware=v1.hardware,
        kernel_name=v1.kernel_name,
        file_a="v1.cu",
        file_b="v2.cu",
        actual_runs=20,
        max_runs=50,
        min_runs=10,
        noise_threshold=1.0,
        warmup=32,
        l2_flush=False,
        verdict=verdict,
        deltas=deltas,
        roofline=RooflineResult("memory", 0.72, 0.89, 11.0, True),
        ptx_diff=[],
        warnings=[],
        total_hbm=(0.5, 0.3),
        pipeline=3,
    )
    assert payload["total_hbm"]["v1_gb"] == 0.5
    assert payload["total_hbm"]["v2_gb"] == 0.3
    assert payload["config"]["pipeline"] == 3


def test_shape_table_renders():
    from kerndiff.renderer import render_shape_table
    rows = [
        {"shape": 1024, "v1_us": 10.0, "v2_us": 8.0, "speedup": 1.25, "dram_delta": 5.0, "bound": "mem"},
        {"shape": 2048, "v1_us": 20.0, "v2_us": 15.0, "speedup": 1.33, "dram_delta": -3.0, "bound": "com"},
    ]
    table = render_shape_table(rows)
    assert "1024" in table
    assert "2048" in table
    assert "1.25x" in table
    assert "1.33x" in table


def test_verdict_clocks_unlocked_note(v1_result, v2_result):
    verdict = compute_verdict(v1_result, v2_result)
    line = render_verdict(verdict, use_color=False, clocks_locked=False)
    assert "note: clocks not locked" in line


def test_verdict_clocks_locked_no_note(v1_result, v2_result):
    verdict = compute_verdict(v1_result, v2_result)
    line = render_verdict(verdict, use_color=False, clocks_locked=True)
    assert "note:" not in line


def test_verdict_unchanged_shows_noise_info(v1_result):
    verdict = compute_verdict(v1_result, v1_result)
    line = render_verdict(verdict, use_color=False, clocks_locked=False)
    assert "noise" in line


def test_verdict_shows_uncertainty_for_nontrivial_error(v1_result, v2_result):
    verdict = compute_verdict(v1_result, v2_result)
    line = render_verdict(verdict, use_color=False, clocks_locked=True)
    assert "±" in line
    assert "x" in line


def test_noise_ceiling_question_mark(v1_result, v2_result):
    """Metrics with small deltas should get ? when noise_ceiling is high."""
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    table = render_metric_table(
        deltas, v1_result, v2_result,
        use_color=False,
        noise_ceiling=200.0,  # Very high — everything is uncertain
    )
    assert "?" in table


def test_no_question_mark_when_locked(v1_result, v2_result):
    """No ? when noise_ceiling is 0 (clocks locked)."""
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    table = render_metric_table(
        deltas, v1_result, v2_result,
        use_color=False,
        noise_ceiling=0.0,
    )
    assert "?" not in table


def test_startup_hardware_line_in_stderr():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel", "--no-color"])
    assert rc == 0
    assert "gpu: NVIDIA H100 SXM5 80GB (mock)" in stderr.getvalue()
