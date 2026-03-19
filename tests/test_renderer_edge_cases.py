"""Test renderer with degenerate inputs."""
from kerndiff.renderer import render_verdict, render_metric_table, render_ptx_diff
from kerndiff.diff import compute_verdict, sort_deltas, compute_all_deltas
from conftest import make_result


def test_verdict_render_no_color():
    r1 = make_result(200.0)
    r2 = make_result(100.0)
    v = compute_verdict(r1, r2, noise_floor=0.02)
    out = render_verdict(v, use_color=False)
    assert "2.00x faster" in out
    assert "\033[" not in out  # no ANSI codes


def test_verdict_render_slower():
    r1 = make_result(100.0)
    r2 = make_result(200.0)
    v = compute_verdict(r1, r2, noise_floor=0.02)
    out = render_verdict(v, use_color=False)
    assert "slower" in out
    assert "faster" not in out


def test_verdict_render_unchanged():
    r1 = make_result(100.0)
    r2 = make_result(100.0)
    v = compute_verdict(r1, r2, noise_floor=0.02)
    out = render_verdict(v, use_color=False)
    assert "no significant" in out


def test_metric_table_all_unchanged():
    """All metrics identical — table should have all ~ symbols (no ++/-- judgments)."""
    r = make_result()
    deltas = sort_deltas(compute_all_deltas(r.metrics, r.metrics, noise_floor=0.02))
    table = render_metric_table(deltas, r, r, use_color=False)
    # Check that no delta symbols (++ or --) appear, ignoring separator lines
    for line in table.splitlines():
        if line.strip().startswith("-"):
            continue  # separator line
        assert "  ++\n" not in line + "\n"
        assert "  --\n" not in line + "\n"


def test_metric_table_empty_metrics():
    """No NCU metrics collected — table should still render latency row."""
    r = make_result()
    r.metrics = {"latency_us": 200.0}
    deltas = sort_deltas(compute_all_deltas(r.metrics, r.metrics, noise_floor=0.02))
    table = render_metric_table(deltas, r, r, use_color=False)
    assert "latency" in table


def test_ptx_diff_empty():
    assert render_ptx_diff([]) == ""


def test_ptx_diff_only_additions():
    diff = [{"instruction": "ld.shared", "v1": 0, "v2": 28, "delta_pct": 2800.0}]
    out = render_ptx_diff(diff)
    assert "ld.shared" in out
    assert "28" in out


def test_ptx_diff_only_removals():
    diff = [{"instruction": "ld.global", "v1": 6, "v2": 0, "delta_pct": -100.0}]
    out = render_ptx_diff(diff)
    assert "ld.global" in out
    assert "-100.0%" in out


def test_render_with_unicode_kernel_name():
    """Kernel names should be ASCII but render shouldn't crash on non-ASCII."""
    r = make_result()
    r.kernel_name = "kérnel_tëst"
    v = compute_verdict(r, r, noise_floor=0.02)
    out = render_verdict(v, use_color=False)
    assert out is not None  # didn't crash
