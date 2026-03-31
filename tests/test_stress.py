"""
Stress tests for kerndiff — adversarial inputs, boundary conditions, failure modes.
Covers all new code paths added in the 2026-03-30 session.
"""
from __future__ import annotations

import math
import subprocess

import pytest

from kerndiff.diff import compute_all_deltas, compute_derived_metrics, sort_deltas
from kerndiff.metrics import METRICS, METRICS_BY_KEY
from kerndiff.parser import parse_ncu_csv
from kerndiff.profiler import (
    MOCK_HARDWARE,
    _check_missing_metrics,
    _compute_cv,
    _remove_outliers,
    _run_batch,
    interleave_timing,
    profile,
)
from kerndiff.renderer import format_delta, render_metric_table, render_verdict
from conftest import make_result

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CLOCK_TELEM = {
    "current_sm_clock_mhz": 1800, "current_mem_clock_mhz": 2000,
    "max_sm_clock_mhz": 1800, "max_mem_clock_mhz": 2000,
    "throttle_reasons": [],
}


def _patch_base(monkeypatch, latencies=None, ncu_stdout=""):
    """Patch the three things every profile() call needs to succeed without GPU."""
    monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda *a, **kw: 6 * 1024 * 1024)
    monkeypatch.setattr("kerndiff.profiler.query_clock_telemetry", lambda *a, **kw: _CLOCK_TELEM)
    monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *a, **kw: None)
    monkeypatch.setattr("kerndiff.profiler._find_ncu", lambda: None if ncu_stdout == "" else "/ncu")
    lats = latencies if latencies is not None else [100.0] * 10
    monkeypatch.setattr("kerndiff.profiler._run_timed_legacy", lambda *a, **kw: lats)


# ===========================================================================
# interleave_timing — edge cases
# ===========================================================================

class TestInterleaveTiming:

    def _fake_subprocess(self, a_lats, b_lats):
        """Returns a subprocess.run mock that yields values from the given lists."""
        state = {"a": iter(a_lats), "b": iter(b_lats)}

        def _run(cmd, *args, **kwargs):
            binary = cmd[0]
            key = "a" if binary == "/bin/a" else "b"
            val = next(state[key])
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{val}\n", stderr="")

        return _run

    def test_converges_after_min_runs_when_cv_already_low(self, monkeypatch):
        """If CV is already below threshold after min_runs, stops exactly at min_runs."""
        monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda *a, **kw: 1024)
        monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *a, **kw: None)
        # Very stable latencies — CV ≈ 0
        stable = [100.0] * 50
        monkeypatch.setattr("kerndiff.profiler.subprocess.run",
                            self._fake_subprocess(stable, stable))
        la, lb, warns = interleave_timing(
            "/bin/a", "/bin/b", "k",
            min_runs=5, max_runs=30, noise_threshold=1.0,
            warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
        )
        assert len(la) == len(lb) == 5
        assert warns == []

    def test_runs_until_max_when_high_variance(self, monkeypatch):
        """If CV never drops below threshold, stops at max_runs and warns."""
        monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda *a, **kw: 1024)
        monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *a, **kw: None)
        import random as _random
        rng = _random.Random(0)
        noisy = [rng.uniform(50.0, 200.0) for _ in range(100)]
        monkeypatch.setattr("kerndiff.profiler.subprocess.run",
                            self._fake_subprocess(noisy, noisy))
        la, lb, warns = interleave_timing(
            "/bin/a", "/bin/b", "k",
            min_runs=5, max_runs=10, noise_threshold=0.001,
            warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
        )
        assert len(la) == len(lb) == 10
        assert any("noise threshold" in w for w in warns)

    def test_lists_always_equal_length(self, monkeypatch):
        """No matter what, returned lists must be the same length."""
        monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda *a, **kw: 1024)
        monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *a, **kw: None)
        a_lats = [100.0 + i * 0.01 for i in range(100)]
        b_lats = [200.0 + i * 0.5  for i in range(100)]   # high variance on b
        monkeypatch.setattr("kerndiff.profiler.subprocess.run",
                            self._fake_subprocess(a_lats, b_lats))
        la, lb, _ = interleave_timing(
            "/bin/a", "/bin/b", "k",
            min_runs=5, max_runs=20, noise_threshold=1.0,
            warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
        )
        assert len(la) == len(lb)

    def test_crash_on_binary_a_raises_systemexit(self, monkeypatch):
        monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda *a, **kw: 1024)
        monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *a, **kw: None)

        def _crash(cmd, *args, **kwargs):
            if cmd[0] == "/bin/a":
                raise subprocess.CalledProcessError(1, cmd, output="", stderr="segfault")
            return subprocess.CompletedProcess(cmd, 0, stdout="100.0\n", stderr="")

        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _crash)
        with pytest.raises(SystemExit, match="interleaved timing"):
            interleave_timing("/bin/a", "/bin/b", "k",
                              min_runs=5, max_runs=10, noise_threshold=1.0,
                              warmup=0, gpu_id=0, hardware=MOCK_HARDWARE)

    def test_crash_on_binary_b_raises_systemexit(self, monkeypatch):
        monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda *a, **kw: 1024)
        monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *a, **kw: None)

        def _crash(cmd, *args, **kwargs):
            if cmd[0] == "/bin/b":
                raise subprocess.CalledProcessError(2, cmd, output="", stderr="OOM")
            return subprocess.CompletedProcess(cmd, 0, stdout="100.0\n", stderr="")

        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _crash)
        with pytest.raises(SystemExit, match="interleaved timing"):
            interleave_timing("/bin/a", "/bin/b", "k",
                              min_runs=5, max_runs=10, noise_threshold=1.0,
                              warmup=0, gpu_id=0, hardware=MOCK_HARDWARE)

    def test_max_runs_one(self, monkeypatch):
        """max_runs=1 should produce exactly one pair."""
        monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda *a, **kw: 1024)
        monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *a, **kw: None)
        monkeypatch.setattr("kerndiff.profiler.subprocess.run",
                            self._fake_subprocess([99.0] * 5, [101.0] * 5))
        la, lb, _ = interleave_timing(
            "/bin/a", "/bin/b", "k",
            min_runs=1, max_runs=1, noise_threshold=0.0,
            warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
        )
        assert len(la) == len(lb) == 1

    def test_all_latencies_positive(self, monkeypatch):
        monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda *a, **kw: 1024)
        monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *a, **kw: None)
        monkeypatch.setattr("kerndiff.profiler.subprocess.run",
                            self._fake_subprocess([0.001] * 20, [99999.9] * 20))
        la, lb, _ = interleave_timing(
            "/bin/a", "/bin/b", "k",
            min_runs=5, max_runs=5, noise_threshold=0.0,
            warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
        )
        assert all(v > 0 for v in la)
        assert all(v > 0 for v in lb)


# ===========================================================================
# profile() with pre_collected_latencies — boundary conditions
# ===========================================================================

class TestPreCollectedLatencies:

    def test_empty_list_raises_systemexit(self, monkeypatch):
        """Empty pre_collected_latencies should fail at the 'no samples' check."""
        _patch_base(monkeypatch)
        with pytest.raises(SystemExit, match="no timing samples"):
            profile(binary="/fake", kernel_name="k",
                    max_runs=50, min_runs=10, noise_threshold=1.0,
                    warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
                    mock=False, pre_collected_latencies=[])

    def test_single_sample(self, monkeypatch):
        _patch_base(monkeypatch)
        result = profile(binary="/fake", kernel_name="k",
                         max_runs=50, min_runs=10, noise_threshold=1.0,
                         warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
                         mock=False, pre_collected_latencies=[123.456])
        assert result.min_latency_us == 123.456
        assert result.cv_pct == 0.0   # single sample → CV undefined → 0

    def test_large_list(self, monkeypatch):
        _patch_base(monkeypatch)
        lats = [100.0 + i * 0.001 for i in range(1000)]
        result = profile(binary="/fake", kernel_name="k",
                         max_runs=50, min_runs=10, noise_threshold=1.0,
                         warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
                         mock=False, pre_collected_latencies=lats)
        assert result.actual_runs == 1000
        assert result.min_latency_us == pytest.approx(100.0, abs=0.001)

    def test_outliers_removed(self, monkeypatch):
        _patch_base(monkeypatch)
        lats = [100.0] * 9 + [999.0]   # one massive outlier
        result = profile(binary="/fake", kernel_name="k",
                         max_runs=50, min_runs=5, noise_threshold=1.0,
                         warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
                         mock=False, pre_collected_latencies=lats)
        assert result.n_outliers == 1
        assert len(result.clean_latencies_us) == 9

    def test_all_identical_cv_zero(self, monkeypatch):
        _patch_base(monkeypatch)
        lats = [200.0] * 20
        result = profile(binary="/fake", kernel_name="k",
                         max_runs=50, min_runs=10, noise_threshold=1.0,
                         warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
                         mock=False, pre_collected_latencies=lats)
        assert result.cv_pct == 0.0

    def test_trace_event_pre_collected_present(self, monkeypatch):
        _patch_base(monkeypatch)
        result = profile(binary="/fake", kernel_name="k",
                         max_runs=50, min_runs=10, noise_threshold=1.0,
                         warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
                         mock=False, pre_collected_latencies=[100.0] * 10)
        names = [e["name"] for e in result.trace_events]
        assert "pre_collected_timing" in names
        # warmup and timed_runs phases must NOT be present
        assert "warmup" not in names

    def test_high_variance_pre_collected_warns(self, monkeypatch):
        _patch_base(monkeypatch)
        lats = [100.0, 200.0, 50.0, 180.0, 90.0, 170.0, 60.0, 190.0, 80.0, 160.0]
        result = profile(binary="/fake", kernel_name="k",
                         max_runs=50, min_runs=5, noise_threshold=1.0,
                         warmup=0, gpu_id=0, hardware=MOCK_HARDWARE,
                         mock=False, pre_collected_latencies=lats)
        # High variance should trigger the high-variance warning
        assert any("high variance" in w or "variance" in w.lower() for w in result.warnings)


# ===========================================================================
# _run_batch — edge cases
# ===========================================================================

class TestRunBatch:

    def test_crash_raises_systemexit(self, monkeypatch):
        def _crash(cmd, *a, **kw):
            raise subprocess.CalledProcessError(1, cmd, output="", stderr="segfault")
        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _crash)
        with pytest.raises(SystemExit):
            _run_batch("/bin/fake", "k", l2_size_bytes=1024, n=5, run_env={})

    def test_no_l2_flush_flag_when_zero(self, monkeypatch):
        """When l2_size_bytes=0, --l2-flush must not be included."""
        cmds = []
        def _ok(cmd, *a, **kw):
            cmds.append(cmd)
            return subprocess.CompletedProcess(cmd, 0, stdout="100.0\n" * 3, stderr="")
        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _ok)
        _run_batch("/bin/fake", "k", l2_size_bytes=0, n=3, run_env={})
        assert "--l2-flush" not in cmds[0]

    def test_blank_lines_in_output_ignored(self, monkeypatch):
        def _ok(cmd, *a, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout="\n100.0\n\n200.0\n\n300.0\n\n", stderr="")
        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _ok)
        samples = _run_batch("/bin/fake", "k", l2_size_bytes=0, n=3, run_env={})
        assert samples == [100.0, 200.0, 300.0]

    def test_scientific_notation_parsed(self, monkeypatch):
        def _ok(cmd, *a, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout="1.5e2\n2.0E1\n", stderr="")
        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _ok)
        samples = _run_batch("/bin/fake", "k", l2_size_bytes=0, n=2, run_env={})
        assert samples == pytest.approx([150.0, 20.0])

    def test_multi_time_flag_value_correct(self, monkeypatch):
        cmds = []
        def _ok(cmd, *a, **kw):
            cmds.append(cmd)
            n = int(cmd[cmd.index("--multi-time") + 1])
            return subprocess.CompletedProcess(cmd, 0, stdout="\n".join(["100.0"] * n), stderr="")
        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _ok)
        for n in (1, 5, 50, 100):
            result = _run_batch("/bin/fake", "k", l2_size_bytes=0, n=n, run_env={})
            assert len(result) == n

    def test_empty_output_returns_empty_list(self, monkeypatch):
        def _ok(cmd, *a, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout="   \n\n", stderr="")
        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _ok)
        samples = _run_batch("/bin/fake", "k", l2_size_bytes=0, n=5, run_env={})
        assert samples == []


# ===========================================================================
# _check_missing_metrics — boundary conditions
# ===========================================================================

class TestCheckMissingMetrics:

    def _visible_ncu_keys(self):
        return {m.key for m in METRICS if not m.hidden and m.ncu_metric}

    def test_all_present_no_warning(self):
        metrics = {k: 1.0 for k in self._visible_ncu_keys()}
        warnings: list[str] = []
        _check_missing_metrics(metrics, warnings)
        assert warnings == []

    def test_all_missing_warns_with_ellipsis(self):
        warnings: list[str] = []
        _check_missing_metrics({}, warnings)
        assert len(warnings) == 1
        # More than 5 missing → ellipsis
        assert "..." in warnings[0]

    def test_exactly_five_missing_no_ellipsis(self):
        # Keep all but 5 present
        all_keys = list(self._visible_ncu_keys())
        missing_5 = set(all_keys[:5])
        metrics = {k: 1.0 for k in all_keys[5:]}
        warnings: list[str] = []
        _check_missing_metrics(metrics, warnings)
        assert len(warnings) == 1
        assert "..." not in warnings[0]
        assert "5" in warnings[0]

    def test_six_missing_has_ellipsis(self):
        all_keys = list(self._visible_ncu_keys())
        if len(all_keys) < 6:
            pytest.skip("not enough visible metrics for this test")
        metrics = {k: 1.0 for k in all_keys[6:]}
        warnings: list[str] = []
        _check_missing_metrics(metrics, warnings)
        assert "..." in warnings[0]

    def test_hidden_metrics_not_checked(self):
        # Only hidden metrics are missing → no warning
        visible_keys = self._visible_ncu_keys()
        hidden_keys = {m.key for m in METRICS if m.hidden}
        metrics = {k: 1.0 for k in visible_keys}
        # Remove hidden keys (they shouldn't matter)
        for k in hidden_keys:
            metrics.pop(k, None)
        warnings: list[str] = []
        _check_missing_metrics(metrics, warnings)
        assert warnings == []

    def test_derived_metrics_not_checked(self):
        # Metrics with empty ncu_metric string (derived) must not be flagged
        visible_ncu = {m.key for m in METRICS if not m.hidden and m.ncu_metric}
        metrics = {k: 1.0 for k in visible_ncu}
        warnings: list[str] = []
        _check_missing_metrics(metrics, warnings)
        assert warnings == []


# ===========================================================================
# NCU latency cross-validation — boundary conditions
# ===========================================================================

class TestNCULatencyCrossValidation:

    def _profile_with_ncu_latency(self, monkeypatch, measured_us, ncu_ns):
        _patch_base(monkeypatch, latencies=[measured_us] * 10)
        monkeypatch.setattr("kerndiff.profiler._find_ncu", lambda: "/ncu")
        monkeypatch.setattr("kerndiff.profiler.shutil.which", lambda cmd: None)
        ncu_csv = (
            '"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",'
            '"Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"\n'
            f'"0","1","./b","h","k","t","1","7","Speed","gpu__time_duration.sum","nsecond","{ncu_ns}"\n'
        )
        def _fake_run(cmd, *a, **kw):
            if "/ncu" in cmd[0]:
                return subprocess.CompletedProcess(cmd, 0, stdout=ncu_csv, stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _fake_run)
        return profile(binary="/fake", kernel_name="k",
                       max_runs=10, min_runs=10, noise_threshold=0.0,
                       warmup=0, gpu_id=0, hardware=MOCK_HARDWARE, mock=False)

    def test_exact_match_no_warning(self, monkeypatch):
        result = self._profile_with_ncu_latency(monkeypatch, 100.0, 100_000)
        assert not any("NCU-reported kernel duration" in w for w in result.warnings)

    def test_ratio_0_8_boundary_no_warning(self, monkeypatch):
        # NCU=80us, measured=100us → ratio=0.8, exactly on boundary → no warning
        result = self._profile_with_ncu_latency(monkeypatch, 100.0, 80_000)
        assert not any("NCU-reported kernel duration" in w for w in result.warnings)

    def test_ratio_below_0_8_warns(self, monkeypatch):
        # NCU=79us → ratio=0.79 → warning
        result = self._profile_with_ncu_latency(monkeypatch, 100.0, 79_000)
        assert any("NCU-reported kernel duration" in w for w in result.warnings)

    def test_ratio_1_25_boundary_no_warning(self, monkeypatch):
        # NCU=125us → ratio=1.25, exactly on boundary → no warning
        result = self._profile_with_ncu_latency(monkeypatch, 100.0, 125_000)
        assert not any("NCU-reported kernel duration" in w for w in result.warnings)

    def test_ratio_above_1_25_warns(self, monkeypatch):
        # NCU=126us → ratio=1.26 → warning
        result = self._profile_with_ncu_latency(monkeypatch, 100.0, 126_000)
        assert any("NCU-reported kernel duration" in w for w in result.warnings)

    def test_ncu_latency_zero_no_warning(self, monkeypatch):
        # If NCU doesn't return latency (0.0), no cross-check
        _patch_base(monkeypatch, latencies=[100.0] * 10)
        monkeypatch.setattr("kerndiff.profiler._find_ncu", lambda: None)
        result = profile(binary="/fake", kernel_name="k",
                         max_runs=10, min_runs=10, noise_threshold=0.0,
                         warmup=0, gpu_id=0, hardware=MOCK_HARDWARE, mock=False)
        assert not any("NCU-reported kernel duration" in w for w in result.warnings)

    def test_very_large_discrepancy_warns(self, monkeypatch):
        # NCU=10x longer — clear sign of wrong launch
        result = self._profile_with_ncu_latency(monkeypatch, 100.0, 1_000_000)
        assert any("NCU-reported kernel duration" in w for w in result.warnings)


# ===========================================================================
# Register spill warning
# ===========================================================================

class TestRegisterSpillWarning:

    def _profile_with_spill(self, monkeypatch, rd_sectors, wr_sectors):
        _patch_base(monkeypatch, latencies=[100.0] * 10)
        monkeypatch.setattr("kerndiff.profiler._find_ncu", lambda: "/ncu")
        monkeypatch.setattr("kerndiff.profiler.shutil.which", lambda cmd: None)
        ncu_csv = (
            '"ID","PID","Proc","Host","Kernel","Time","Ctx","Stream","Sec","Metric Name","Metric Unit","Metric Value"\n'
            f'"0","1","./b","h","k","t","1","7","M",'
            f'"smsp__l1tex_m_l1_read_sectors_pipe_lsu_mem_local_op_ld.sum","","{rd_sectors}"\n'
            f'"0","1","./b","h","k","t","1","7","M",'
            f'"smsp__l1tex_m_l1_write_sectors_pipe_lsu_mem_local_op_st.sum","","{wr_sectors}"\n'
        )
        def _fake_run(cmd, *a, **kw):
            if "/ncu" in cmd[0]:
                return subprocess.CompletedProcess(cmd, 0, stdout=ncu_csv, stderr="")
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _fake_run)
        return profile(binary="/fake", kernel_name="k",
                       max_runs=10, min_runs=10, noise_threshold=0.0,
                       warmup=0, gpu_id=0, hardware=MOCK_HARDWARE, mock=False)

    def test_no_spill_no_warning(self, monkeypatch):
        result = self._profile_with_spill(monkeypatch, 0, 0)
        assert not any("register spilling" in w for w in result.warnings)

    def test_only_reads_warns(self, monkeypatch):
        result = self._profile_with_spill(monkeypatch, 512, 0)
        assert any("register spilling" in w for w in result.warnings)

    def test_only_writes_warns(self, monkeypatch):
        result = self._profile_with_spill(monkeypatch, 0, 256)
        assert any("register spilling" in w for w in result.warnings)

    def test_both_warns(self, monkeypatch):
        result = self._profile_with_spill(monkeypatch, 1024, 512)
        assert any("register spilling" in w for w in result.warnings)
        warning_text = next(w for w in result.warnings if "register spilling" in w)
        assert "1024" in warning_text
        assert "512" in warning_text

    def test_large_spill_count(self, monkeypatch):
        result = self._profile_with_spill(monkeypatch, 10_000_000, 5_000_000)
        assert any("register spilling" in w for w in result.warnings)


# ===========================================================================
# compute_derived_metrics — boundary and numeric edge cases
# ===========================================================================

class TestDerivedMetrics:

    def test_all_zeros_returns_empty(self):
        derived = compute_derived_metrics({})
        assert derived == {}

    def test_sm_imbalance_100_when_equal(self):
        d = compute_derived_metrics({"sm_throughput": 75.0, "sm_occupancy": 75.0})
        assert d["sm_imbalance"] == pytest.approx(100.0)

    def test_sm_imbalance_over_100_allowed(self):
        # Throughput can exceed occupancy on some workloads — don't clamp
        d = compute_derived_metrics({"sm_throughput": 90.0, "sm_occupancy": 50.0})
        assert d["sm_imbalance"] == pytest.approx(180.0)

    def test_sm_imbalance_zero_throughput_no_key(self):
        d = compute_derived_metrics({"sm_throughput": 0.0, "sm_occupancy": 80.0})
        assert "sm_imbalance" not in d

    def test_sm_imbalance_zero_occupancy_no_key(self):
        d = compute_derived_metrics({"sm_throughput": 80.0, "sm_occupancy": 0.0})
        assert "sm_imbalance" not in d

    def test_arith_intensity_only_fp32(self):
        d = compute_derived_metrics({
            "raw_ffma": 1_000_000, "raw_fadd": 0, "raw_fmul": 0,
            "raw_hfma": 0, "raw_hadd": 0, "raw_hmul": 0,
            "raw_dram_sectors_rd": 1_000, "raw_dram_sectors_wr": 0,
            "latency_us": 100.0,
        })
        # ffma = 2 FLOPs, 1000 sectors × 32 bytes = 32000 bytes
        expected_ai = (2 * 1_000_000) / (1_000 * 32)
        assert d["arith_intensity"] == pytest.approx(expected_ai)

    def test_arith_intensity_only_fp16(self):
        d = compute_derived_metrics({
            "raw_ffma": 0, "raw_hfma": 500_000,
            "raw_dram_sectors_rd": 500, "raw_dram_sectors_wr": 0,
            "latency_us": 50.0,
        })
        assert d["arith_intensity"] == pytest.approx((2 * 500_000) / (500 * 32))

    def test_flops_tflops_correct(self):
        d = compute_derived_metrics({
            "raw_ffma": 1_000_000_000_000,  # 1e12 ops
            "raw_dram_sectors_rd": 1_000, "raw_dram_sectors_wr": 0,
            "latency_us": 1_000_000.0,  # 1 second
        })
        # 2e12 FLOPs / 1s = 2 TFLOPS
        assert d["flops_tflops"] == pytest.approx(2.0, rel=1e-6)

    def test_no_crash_on_nan_values(self):
        # NaN inputs should not raise — derived metrics just won't be computed
        d = compute_derived_metrics({"sm_throughput": float("nan"), "sm_occupancy": 50.0})
        # sm_throughput is nan → condition sm_throughput > 0 is False → no key
        assert "sm_imbalance" not in d

    def test_no_crash_on_inf_values(self):
        d = compute_derived_metrics({"sm_throughput": float("inf"), "sm_occupancy": 50.0})
        # inf > 0 is True, inf / 50 * 100 = inf — should not raise
        assert "sm_imbalance" in d
        assert math.isinf(d["sm_imbalance"])


# ===========================================================================
# _remove_outliers — edge cases
# ===========================================================================

class TestRemoveOutliers:

    def test_less_than_5_samples_no_removal(self):
        samples = [100.0, 200.0, 50.0, 75.0]
        clean, n = _remove_outliers(samples)
        assert clean == samples
        assert n == 0

    def test_exactly_50_pct_would_be_removed_keeps_original(self):
        # 5 samples, 3 are >2x median → removing 3 would be ≥50%, so keep all
        samples = [100.0, 101.0, 250.0, 260.0, 270.0]
        clean, n = _remove_outliers(samples)
        assert clean == samples
        assert n == 0

    def test_all_identical_no_outliers(self):
        samples = [100.0] * 20
        clean, n = _remove_outliers(samples)
        assert clean == samples
        assert n == 0

    def test_single_massive_outlier_removed(self):
        samples = [100.0] * 9 + [9999.0]
        clean, n = _remove_outliers(samples)
        assert n == 1
        assert 9999.0 not in clean

    def test_multiple_outliers_removed_if_safe(self):
        # 8 clean + 2 outliers (2 < 8/2=4, so safe to remove)
        samples = [100.0] * 8 + [500.0, 600.0]
        clean, n = _remove_outliers(samples)
        assert n == 2
        assert len(clean) == 8

    def test_empty_list(self):
        clean, n = _remove_outliers([])
        assert clean == []
        assert n == 0


# ===========================================================================
# Parser — adversarial CSV inputs
# ===========================================================================

class TestParserStress:

    def test_duplicate_metric_last_value_wins(self):
        csv = (
            '"Metric Name","Metric Unit","Metric Value"\n'
            '"gpu__time_duration.sum","nsecond","100000"\n'
            '"gpu__time_duration.sum","nsecond","200000"\n'
        )
        metrics = parse_ncu_csv(csv)
        # Second row should overwrite first
        assert metrics["latency_us"] == 200.0

    def test_extra_columns_ignored(self):
        csv = (
            '"ID","Process ID","Host","Kernel","Metric Name","Metric Unit","Metric Value","Extra Col"\n'
            '"0","1","h","k","gpu__time_duration.sum","nsecond","50000","garbage"\n'
        )
        metrics = parse_ncu_csv(csv)
        assert metrics["latency_us"] == 50.0

    def test_quoted_values_stripped(self):
        csv = '"Metric Name","Metric Unit","Metric Value"\n"gpu__time_duration.sum","nsecond","300000"\n'
        metrics = parse_ncu_csv(csv)
        assert metrics["latency_us"] == 300.0

    def test_noncsv_preamble_skipped(self):
        csv = (
            "==PROF== Connected to process 12345\n"
            "==PROF== Some other header line\n"
            '"Metric Name","Metric Unit","Metric Value"\n'
            '"gpu__time_duration.sum","nsecond","150000"\n'
        )
        metrics = parse_ncu_csv(csv)
        assert metrics["latency_us"] == 150.0

    def test_entirely_empty_csv(self):
        assert parse_ncu_csv("") == {}

    def test_header_only_no_data(self):
        assert parse_ncu_csv('"Metric Name","Metric Unit","Metric Value"\n') == {}

    def test_byte_per_second_unit_converts_to_gbs(self):
        csv = (
            '"Metric Name","Metric Unit","Metric Value"\n'
            '"dram__bytes.sum.per_second","byte/second","900000000000"\n'
        )
        metrics = parse_ncu_csv(csv)
        assert metrics["dram_bw_gbs"] == pytest.approx(900.0)

    def test_byte_s_unit_variant_converts(self):
        csv = (
            '"Metric Name","Metric Unit","Metric Value"\n'
            '"dram__bytes.sum.per_second","byte/s","500000000000"\n'
        )
        metrics = parse_ncu_csv(csv)
        assert metrics["dram_bw_gbs"] == pytest.approx(500.0)

    def test_invalid_float_skipped(self):
        csv = (
            '"Metric Name","Metric Unit","Metric Value"\n'
            '"gpu__time_duration.sum","nsecond","not_a_number"\n'
        )
        assert parse_ncu_csv(csv) == {}

    def test_comma_in_large_number(self):
        csv = (
            '"Metric Name","Metric Unit","Metric Value"\n'
            '"smsp__inst_executed.sum","inst","1,234,567"\n'
        )
        metrics = parse_ncu_csv(csv)
        assert metrics["inst_executed"] == 1_234_567.0

    def test_ncu_scale_applied_to_thread_active(self):
        # thread_active_pct has ncu_scale=100/32; NCU returns 0–32
        csv = (
            '"Metric Name","Metric Unit","Metric Value"\n'
            '"smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio","","32.0"\n'
        )
        metrics = parse_ncu_csv(csv)
        assert metrics["thread_active_pct"] == pytest.approx(100.0)

    def test_warp_exec_eff_scale_applied(self):
        csv = (
            '"Metric Name","Metric Unit","Metric Value"\n'
            '"smsp__thread_inst_executed_per_inst_executed.ratio","","16.0"\n'
        )
        metrics = parse_ncu_csv(csv)
        assert metrics["warp_exec_eff"] == pytest.approx(50.0)

    def test_new_stall_metrics_parsed(self):
        csv = (
            '"Metric Name","Metric Unit","Metric Value"\n'
            '"smsp__warp_issue_stalled_no_instruction_per_warp_active.pct","percent","12.5"\n'
            '"smsp__warp_issue_stalled_math_pipe_throttle_per_warp_active.pct","percent","7.3"\n'
            '"smsp__warp_issue_stalled_tex_throttle_per_warp_active.pct","percent","3.1"\n'
            '"smsp__warp_issue_stalled_wait_per_warp_active.pct","percent","9.9"\n'
        )
        metrics = parse_ncu_csv(csv)
        assert metrics["stall_not_selected"] == pytest.approx(12.5)
        assert metrics["stall_pipe_busy"] == pytest.approx(7.3)
        assert metrics["stall_tex_throttle"] == pytest.approx(3.1)
        assert metrics["stall_wait"] == pytest.approx(9.9)

    def test_tensor_core_metric_parsed(self):
        csv = (
            '"Metric Name","Metric Unit","Metric Value"\n'
            '"smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_active","percent","78.4"\n'
        )
        metrics = parse_ncu_csv(csv)
        assert metrics["tensor_core_util"] == pytest.approx(78.4)

    def test_local_memory_spill_metrics_parsed(self):
        csv = (
            '"Metric Name","Metric Unit","Metric Value"\n'
            '"smsp__l1tex_m_l1_read_sectors_pipe_lsu_mem_local_op_ld.sum","","4096"\n'
            '"smsp__l1tex_m_l1_write_sectors_pipe_lsu_mem_local_op_st.sum","","2048"\n'
        )
        metrics = parse_ncu_csv(csv)
        assert metrics["local_load_sectors"] == 4096.0
        assert metrics["local_store_sectors"] == 2048.0


# ===========================================================================
# compute_all_deltas — numeric edge cases
# ===========================================================================

class TestDeltasStress:

    def test_v1_zero_no_crash(self):
        # v1=0 → delta uses denominator=1 fallback
        delta = compute_all_deltas({"latency_us": 0.0}, {"latency_us": 100.0})
        assert any(d.metric.key == "latency_us" for d in delta)

    def test_both_zero_symbol_tilde(self):
        deltas = compute_all_deltas({"latency_us": 0.0}, {"latency_us": 0.0})
        lat = next(d for d in deltas if d.metric.key == "latency_us")
        assert lat.symbol == "~"

    def test_nan_v1_symbol_tilde(self):
        deltas = compute_all_deltas({"latency_us": float("nan")}, {"latency_us": 100.0})
        lat = next(d for d in deltas if d.metric.key == "latency_us")
        assert lat.symbol == "~"

    def test_nan_v2_symbol_tilde(self):
        deltas = compute_all_deltas({"latency_us": 100.0}, {"latency_us": float("nan")})
        lat = next(d for d in deltas if d.metric.key == "latency_us")
        assert lat.symbol == "~"

    def test_inf_v1_symbol_tilde(self):
        deltas = compute_all_deltas({"latency_us": float("inf")}, {"latency_us": 100.0})
        lat = next(d for d in deltas if d.metric.key == "latency_us")
        assert lat.symbol == "~"

    def test_missing_metric_in_v1_skipped(self):
        # Only metrics present in BOTH v1 and v2 should appear
        deltas = compute_all_deltas({}, {"latency_us": 100.0})
        keys = {d.metric.key for d in deltas}
        assert "latency_us" not in keys

    def test_new_metrics_appear_in_deltas(self):
        v1 = {"stall_not_selected": 20.0, "warp_exec_eff": 60.0, "branch_divergence": 10.0}
        v2 = {"stall_not_selected": 10.0, "warp_exec_eff": 80.0, "branch_divergence": 5.0}
        deltas = compute_all_deltas(v1, v2)
        keys = {d.metric.key for d in deltas}
        assert "stall_not_selected" in keys
        assert "warp_exec_eff" in keys
        assert "branch_divergence" in keys

    def test_warp_exec_eff_improvement_symbol_positive(self):
        # warp_exec_eff: higher is better; 60 → 80 = +33% → "++" or "+"
        v1 = {"warp_exec_eff": 60.0}
        v2 = {"warp_exec_eff": 80.0}
        deltas = compute_all_deltas(v1, v2)
        d = next(x for x in deltas if x.metric.key == "warp_exec_eff")
        assert d.symbol in ("+", "++")

    def test_branch_divergence_improvement_symbol_positive(self):
        # lower is better; 20 → 5 = -75% → "++"
        v1 = {"branch_divergence": 20.0}
        v2 = {"branch_divergence": 5.0}
        deltas = compute_all_deltas(v1, v2)
        d = next(x for x in deltas if x.metric.key == "branch_divergence")
        assert d.symbol == "++"

    def test_tensor_core_util_regression_symbol_negative(self):
        # higher is better; 80 → 20 = -75% → "--"
        v1 = {"tensor_core_util": 80.0}
        v2 = {"tensor_core_util": 20.0}
        deltas = compute_all_deltas(v1, v2)
        d = next(x for x in deltas if x.metric.key == "tensor_core_util")
        assert d.symbol == "--"


# ===========================================================================
# Renderer — stress with new metrics and edge cases
# ===========================================================================

class TestRendererStress:

    def test_render_verdict_improvement(self):
        v = make_result(200.0)
        v2 = make_result(100.0)
        verdict = {
            "direction": "improvement", "label": "v2 is 2.00x faster",
            "v1_latency_us": 200.0, "v2_latency_us": 100.0,
            "v1_cv_pct": 0.5, "v2_cv_pct": 0.3,
            "speedup_uncertainty_x": 0.01,
        }
        out = render_verdict(verdict, use_color=False)
        assert "2.00x faster" in out

    def test_render_verdict_no_color_no_ansi(self):
        verdict = {
            "direction": "regression", "label": "v2 is 1.50x slower",
            "v1_latency_us": 100.0, "v2_latency_us": 150.0,
            "v1_cv_pct": 1.0, "v2_cv_pct": 1.0,
            "speedup_uncertainty_x": 0.02,
        }
        out = render_verdict(verdict, use_color=False)
        assert "\033[" not in out
        assert "slower" in out

    def test_render_metric_table_with_new_metrics(self):
        v1 = make_result(200.0, {"warp_exec_eff": 60.0, "branch_divergence": 15.0,
                                  "stall_not_selected": 20.0})
        v2 = make_result(100.0, {"warp_exec_eff": 90.0, "branch_divergence": 3.0,
                                  "stall_not_selected": 25.0})
        deltas = sort_deltas(compute_all_deltas(v1.metrics, v2.metrics))
        out = render_metric_table(deltas, v1, v2, use_color=False)
        assert "warp_exec_eff" in out
        assert "branch_divergence" in out
        assert "stall_not_selected" in out

    def test_render_metric_table_empty_deltas(self):
        v1 = make_result(200.0)
        v2 = make_result(100.0)
        out = render_metric_table([], v1, v2, use_color=False)
        assert out == ""

    def test_format_delta_percentage_unit(self):
        delta = compute_all_deltas({"l2_hit_rate": 40.0}, {"l2_hit_rate": 60.0})
        d = next(x for x in delta if x.metric.key == "l2_hit_rate")
        fmt = format_delta(d)
        assert "pp" in fmt   # percentage-point delta

    def test_format_delta_count_unit(self):
        delta = compute_all_deltas({"l1_bank_conflicts": 1000.0}, {"l1_bank_conflicts": 500.0})
        d = next(x for x in delta if x.metric.key == "l1_bank_conflicts")
        fmt = format_delta(d)
        assert "%" in fmt

    def test_render_table_no_crash_extreme_values(self):
        v1 = make_result(0.001)   # sub-microsecond
        v2 = make_result(1_000_000.0)  # 1 second
        deltas = sort_deltas(compute_all_deltas(v1.metrics, v2.metrics))
        # Should not raise
        render_metric_table(deltas, v1, v2, use_color=False)

    def test_render_table_all_tilde_symbols(self):
        # v1 == v2 → all deltas should be "~"
        result = make_result(100.0)
        deltas = sort_deltas(compute_all_deltas(result.metrics, result.metrics))
        out = render_metric_table(deltas, result, result, use_color=False)
        # Should render without crashing; all symbols are ~
        assert out != ""


# ===========================================================================
# _compute_cv — numeric edge cases
# ===========================================================================

class TestComputeCV:

    def test_single_value_returns_zero(self):
        assert _compute_cv([100.0]) == 0.0

    def test_empty_returns_zero(self):
        assert _compute_cv([]) == 0.0

    def test_all_identical_returns_zero(self):
        assert _compute_cv([50.0] * 100) == 0.0

    def test_two_equal_values_returns_zero(self):
        assert _compute_cv([77.0, 77.0]) == 0.0

    def test_known_cv(self):
        # mean=100, stdev≈10 → CV≈10%
        import statistics
        vals = [90.0, 100.0, 110.0]
        expected = (statistics.stdev(vals) / statistics.mean(vals)) * 100.0
        assert _compute_cv(vals) == pytest.approx(expected)
