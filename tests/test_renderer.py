import io
import json
import re
from contextlib import redirect_stderr, redirect_stdout

from kerndiff.cli import main
from kerndiff.diff import compute_all_deltas, compute_verdict, sort_deltas
from kerndiff.renderer import build_json_payload, render_metric_table, render_ptx_diff, render_verdict
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
    assert "[v1:" in line


def test_verdict_line_omits_range_for_unchanged(v1_result):
    verdict = compute_verdict(v1_result, v1_result)
    line = render_verdict(verdict, use_color=False)
    assert "[v1:" not in line


def test_register_delta_is_raw_diff(v1_result, v2_result):
    deltas = sort_deltas(compute_all_deltas(v1_result.metrics, v2_result.metrics))
    table = render_metric_table(deltas, v1_result, v2_result, use_color=False)
    assert "+8" in table
    assert "+12.5%" not in table


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
    for key in ["v1_min_us", "v1_max_us", "v1_cv_pct", "v2_min_us", "v2_max_us", "v2_cv_pct"]:
        assert key in decoded["verdict"]


def test_output_file_writes_and_stdout_empty(tmp_path):
    out_file = tmp_path / "out.txt"
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel", "--output", str(out_file)])
    assert rc == 0
    assert stdout.getvalue() == ""
    assert out_file.read_text()


def test_ptx_diff_absent_when_unchanged(v1_result):
    assert render_ptx_diff([]) == ""


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


def test_startup_hardware_line_in_stderr():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel", "--no-color"])
    assert rc == 0
    assert "gpu: NVIDIA H100 SXM5 80GB (mock)" in stderr.getvalue()
