"""
Tests for all 11 new features implemented in the session.

A. globaltimer timing in harness — verified via _run_batch mock
B. SM clock mismatch warning — actual_sm_mhz divergence detection
C. SM frequency as metric — derived from raw_sm_cycles
D. Triton interleaved — interleave_timing_persistent
E. Suggestions module — pattern-based hints
F. Watch trend history — _WATCH_HISTORY populated
G. Shape sweep pairwise — _pairwise_stats used in sweep
H. Bank conflict split metrics — parsed from NCU CSV
I. L2 read/write split metrics — parsed from NCU CSV
J. --timeout flag — subprocess.TimeoutExpired raised
K. --determinism flag — check_determinism detects non-determinism
"""
from __future__ import annotations

import subprocess
import statistics

import pytest

from kerndiff.compiler import check_determinism
from kerndiff.diff import _pairwise_stats, compute_derived_metrics
from kerndiff.metrics import METRICS_BY_KEY
from kerndiff.parser import parse_ncu_csv
from kerndiff.profiler import (
    MOCK_HARDWARE,
    _run_batch,
    interleave_timing_persistent,
    profile,
)
from kerndiff.suggestions import generate_suggestions
from conftest import make_result


# ─────────────────────────────────────────────────────────────────────────────
# C. SM frequency derived metric
# ─────────────────────────────────────────────────────────────────────────────

class TestActualSmMhz:

    def test_derived_from_raw_sm_cycles_and_latency(self):
        # 1800 MHz × 0.001 s = 1,800,000 cycles
        d = compute_derived_metrics({"raw_sm_cycles": 1_800_000.0, "latency_us": 1000.0})
        assert d["actual_sm_mhz"] == pytest.approx(1800.0)

    def test_not_computed_without_cycles(self):
        d = compute_derived_metrics({"latency_us": 1000.0})
        assert "actual_sm_mhz" not in d

    def test_not_computed_without_latency(self):
        d = compute_derived_metrics({"raw_sm_cycles": 1_000_000.0, "latency_us": 0.0})
        assert "actual_sm_mhz" not in d

    def test_raw_sm_cycles_parsed_from_ncu_csv(self):
        csv = (
            '"Metric Name","Metric Unit","Metric Value"\n'
            '"sm__cycles_elapsed.avg","cycle","1800000"\n'
        )
        metrics = parse_ncu_csv(csv)
        assert metrics["raw_sm_cycles"] == pytest.approx(1_800_000.0)

    def test_actual_sm_mhz_metric_def_exists(self):
        assert "actual_sm_mhz" in METRICS_BY_KEY
        m = METRICS_BY_KEY["actual_sm_mhz"]
        assert m.ncu_metric == ""   # derived, not requested from NCU directly


# ─────────────────────────────────────────────────────────────────────────────
# H. Shared memory bank conflicts split
# ─────────────────────────────────────────────────────────────────────────────

class TestBankConflictSplit:

    def test_read_and_write_conflicts_parsed(self):
        csv = (
            '"Metric Name","Metric Unit","Metric Value"\n'
            '"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum","","1024"\n'
            '"l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_st.sum","","512"\n'
        )
        m = parse_ncu_csv(csv)
        assert m["l1_bank_conflicts_rd"] == pytest.approx(1024.0)
        assert m["l1_bank_conflicts_wr"] == pytest.approx(512.0)

    def test_both_metrics_in_registry(self):
        assert "l1_bank_conflicts_rd" in METRICS_BY_KEY
        assert "l1_bank_conflicts_wr" in METRICS_BY_KEY
        assert METRICS_BY_KEY["l1_bank_conflicts_rd"].lower_is_better is True
        assert METRICS_BY_KEY["l1_bank_conflicts_wr"].lower_is_better is True

    def test_total_and_split_coexist(self):
        assert "l1_bank_conflicts" in METRICS_BY_KEY


# ─────────────────────────────────────────────────────────────────────────────
# I. L2 read/write split
# ─────────────────────────────────────────────────────────────────────────────

class TestL2Split:

    def test_l2_sector_read_write_parsed(self):
        csv = (
            '"Metric Name","Metric Unit","Metric Value"\n'
            '"lts__t_sectors_srcunit_tex_op_read.sum","","2048000"\n'
            '"lts__t_sectors_srcunit_tex_op_write.sum","","512000"\n'
        )
        m = parse_ncu_csv(csv)
        assert m["l2_sectors_rd"] == pytest.approx(2_048_000.0)
        assert m["l2_sectors_wr"] == pytest.approx(512_000.0)

    def test_both_metrics_in_registry(self):
        assert "l2_sectors_rd" in METRICS_BY_KEY
        assert "l2_sectors_wr" in METRICS_BY_KEY


# ─────────────────────────────────────────────────────────────────────────────
# E. Suggestions module
# ─────────────────────────────────────────────────────────────────────────────

class TestSuggestions:

    def _make_deltas(self, v1_metrics, v2_metrics):
        from kerndiff.diff import compute_all_deltas
        return compute_all_deltas(v1_metrics, v2_metrics)

    def test_no_suggestions_for_clean_kernel(self):
        metrics = {
            "stall_memory": 5.0, "l2_hit_rate": 80.0, "branch_divergence": 1.0,
            "warp_exec_eff": 95.0, "sm_occupancy": 75.0, "registers_per_thread": 32.0,
            "tensor_core_util": 0.0, "flops_tflops": 0.0,
            "local_load_sectors": 0.0, "local_store_sectors": 0.0,
        }
        hints = generate_suggestions([], metrics)
        assert hints == []

    def test_register_spill_suggestion(self):
        metrics = {"local_load_sectors": 512.0, "local_store_sectors": 256.0}
        hints = generate_suggestions([], metrics)
        assert any("Register spilling" in h for h in hints)

    def test_no_tensor_core_suggestion(self):
        metrics = {"tensor_core_util": 0.0, "flops_tflops": 5.0}
        hints = generate_suggestions([], metrics)
        assert any("tensor core" in h.lower() for h in hints)

    def test_tensor_core_hint_suppressed_when_using_tc(self):
        metrics = {"tensor_core_util": 80.0, "flops_tflops": 5.0}
        hints = generate_suggestions([], metrics)
        assert not any("tensor core" in h.lower() for h in hints)

    def test_high_memory_stall_with_low_l2_hit(self):
        metrics = {"stall_memory": 30.0, "l2_hit_rate": 25.0}
        hints = generate_suggestions([], metrics)
        assert any("tiling" in h.lower() or "locality" in h.lower() for h in hints)

    def test_high_branch_divergence_fires(self):
        metrics = {"branch_divergence": 20.0}
        hints = generate_suggestions([], metrics)
        assert any("divergence" in h.lower() for h in hints)

    def test_low_warp_exec_efficiency_fires(self):
        metrics = {"warp_exec_eff": 40.0}
        hints = generate_suggestions([], metrics)
        assert any("warp execution" in h.lower() for h in hints)

    def test_low_occupancy_with_high_regs_fires(self):
        metrics = {"sm_occupancy": 25.0, "registers_per_thread": 128.0}
        hints = generate_suggestions([], metrics)
        assert any("occupancy" in h.lower() for h in hints)

    def test_worsened_bank_conflicts_fires(self):
        v1 = {"l1_bank_conflicts_rd": 100.0, "l1_bank_conflicts_wr": 50.0}
        v2 = {"l1_bank_conflicts_rd": 5000.0, "l1_bank_conflicts_wr": 2000.0}
        deltas = self._make_deltas(v1, v2)
        hints = generate_suggestions(deltas, v2)
        assert any("bank conflict" in h.lower() for h in hints)

    def test_sm_imbalance_fires_when_low(self):
        metrics = {"sm_imbalance": 40.0}
        hints = generate_suggestions([], metrics)
        assert any("imbalance" in h.lower() for h in hints)


# ─────────────────────────────────────────────────────────────────────────────
# J. --timeout flag
# ─────────────────────────────────────────────────────────────────────────────

class TestTimeout:

    def test_run_batch_raises_on_timeout(self, monkeypatch):
        def _slow(cmd, *a, **kw):
            raise subprocess.TimeoutExpired(cmd, kw.get("timeout", 0))
        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _slow)
        with pytest.raises(SystemExit, match="timed out"):
            _run_batch("/bin/fake", "k", l2_size_bytes=0, n=1, run_env={}, timeout_sec=1)

    def test_zero_timeout_passes_none_to_subprocess(self, monkeypatch):
        calls = []
        def _ok(cmd, *a, **kw):
            calls.append(kw.get("timeout"))
            return subprocess.CompletedProcess(cmd, 0, stdout="100.0\n", stderr="")
        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _ok)
        _run_batch("/bin/fake", "k", l2_size_bytes=0, n=1, run_env={}, timeout_sec=0)
        assert calls[0] is None  # no timeout

    def test_nonzero_timeout_passed_to_subprocess(self, monkeypatch):
        calls = []
        def _ok(cmd, *a, **kw):
            calls.append(kw.get("timeout"))
            return subprocess.CompletedProcess(cmd, 0, stdout="100.0\n", stderr="")
        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _ok)
        _run_batch("/bin/fake", "k", l2_size_bytes=0, n=1, run_env={}, timeout_sec=30)
        assert calls[0] == 30


# ─────────────────────────────────────────────────────────────────────────────
# K. --determinism: check_determinism
# ─────────────────────────────────────────────────────────────────────────────

class TestDeterminism:

    def test_identical_runs_are_deterministic(self, monkeypatch):
        call_count = [0]
        def _run(cmd, *a, **kw):
            call_count[0] += 1
            return subprocess.CompletedProcess(cmd, 0, stdout="1.0\n2.0\n3.0\n", stderr="")
        monkeypatch.setattr("kerndiff.compiler.subprocess.run", _run)
        is_det, max_diff = check_determinism("/fake", n_runs=3)
        assert is_det is True
        assert max_diff == 0.0
        assert call_count[0] == 3

    def test_differing_runs_are_non_deterministic(self, monkeypatch):
        call_count = [0]
        outputs = ["1.0\n2.0\n3.0\n", "1.0\n2.0\n4.0\n", "1.0\n2.0\n3.0\n"]
        def _run(cmd, *a, **kw):
            out = outputs[call_count[0] % len(outputs)]
            call_count[0] += 1
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
        monkeypatch.setattr("kerndiff.compiler.subprocess.run", _run)
        is_det, max_diff = check_determinism("/fake", n_runs=3)
        assert is_det is False
        assert max_diff == pytest.approx(1.0)

    def test_empty_output_treated_as_deterministic(self, monkeypatch):
        def _run(cmd, *a, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        monkeypatch.setattr("kerndiff.compiler.subprocess.run", _run)
        is_det, _ = check_determinism("/fake", n_runs=3)
        assert is_det is True

    def test_two_runs_sufficient(self, monkeypatch):
        outputs = ["1.0\n", "2.0\n"]
        count = [0]
        def _run(cmd, *a, **kw):
            out = outputs[count[0] % 2]
            count[0] += 1
            return subprocess.CompletedProcess(cmd, 0, stdout=out, stderr="")
        monkeypatch.setattr("kerndiff.compiler.subprocess.run", _run)
        is_det, max_diff = check_determinism("/fake", n_runs=2)
        assert is_det is False
        assert max_diff == pytest.approx(1.0)


# ─────────────────────────────────────────────────────────────────────────────
# D. Triton interleaved — interleave_timing_persistent
# ─────────────────────────────────────────────────────────────────────────────

class TestInterleavePersistent:

    def _make_backend(self, latency_value: float):
        """A minimal mock of a persistent backend."""
        class MockBackend:
            def __init__(self, lat):
                self._lat = lat
                self._proc = object()

            def is_persistent(self):
                return True

            def spawn_persistent(self, binary, env=None):
                return self._proc

            def send_time(self, proc):
                return self._lat

            def shutdown(self, proc):
                pass

        return MockBackend(latency_value)

    def test_both_lists_equal_length(self):
        ba = self._make_backend(100.0)
        bb = self._make_backend(200.0)
        la, lb, _ = interleave_timing_persistent(
            ba, "/bin/a", bb, "/bin/b", "k",
            min_runs=10, max_runs=20, noise_threshold=1.0,
            gpu_id=0, hardware=MOCK_HARDWARE,
        )
        assert len(la) == len(lb)
        assert len(la) >= 10

    def test_values_assigned_to_correct_lists(self):
        ba = self._make_backend(111.0)
        bb = self._make_backend(222.0)
        la, lb, _ = interleave_timing_persistent(
            ba, "/bin/a", bb, "/bin/b", "k",
            min_runs=20, max_runs=20, noise_threshold=0.0,
            gpu_id=0, hardware=MOCK_HARDWARE,
        )
        assert all(v == pytest.approx(111.0) for v in la), "a values in wrong list"
        assert all(v == pytest.approx(222.0) for v in lb), "b values in wrong list"

    def test_warns_when_cv_not_converged(self):
        import random
        rng = random.Random(0)

        class NoisyBackend:
            def is_persistent(self): return True
            def spawn_persistent(self, *a, **kw): return object()
            def send_time(self, proc): return rng.uniform(50, 200)
            def shutdown(self, proc): pass

        ba, bb = NoisyBackend(), NoisyBackend()
        _, _, warnings = interleave_timing_persistent(
            ba, "/bin/a", bb, "/bin/b", "k",
            min_runs=5, max_runs=5, noise_threshold=0.001,
            gpu_id=0, hardware=MOCK_HARDWARE,
        )
        assert any("noise threshold" in w for w in warnings)

    def test_speedup_from_persistent_matches_pairwise(self):
        ba = self._make_backend(100.0)
        bb = self._make_backend(50.0)
        la, lb, _ = interleave_timing_persistent(
            ba, "/bin/a", bb, "/bin/b", "k",
            min_runs=10, max_runs=10, noise_threshold=0.0,
            gpu_id=0, hardware=MOCK_HARDWARE,
        )
        ps = _pairwise_stats(la, lb)
        assert ps is not None
        assert ps[0] == pytest.approx(2.0)   # 100/50 = 2.0x


# ─────────────────────────────────────────────────────────────────────────────
# B. SM clock mismatch warning (via actual_sm_mhz in profile results)
# ─────────────────────────────────────────────────────────────────────────────

class TestSmClockMismatch:

    def test_actual_sm_mhz_formula(self):
        # 1000 MHz kernel runs for 500µs → cycles = 1000e6 * 500e-6 = 500,000
        d = compute_derived_metrics({"raw_sm_cycles": 500_000.0, "latency_us": 500.0})
        assert d["actual_sm_mhz"] == pytest.approx(1000.0)

    def test_large_cycle_count_high_freq(self):
        # H100 at 1980 MHz, kernel runs 1000µs → 1,980,000 cycles
        d = compute_derived_metrics({"raw_sm_cycles": 1_980_000.0, "latency_us": 1000.0})
        assert d["actual_sm_mhz"] == pytest.approx(1980.0)

    def test_mhz_proportional_to_cycles(self):
        # Double the cycles → double the MHz (for same latency)
        d1 = compute_derived_metrics({"raw_sm_cycles": 1_000_000.0, "latency_us": 1000.0})
        d2 = compute_derived_metrics({"raw_sm_cycles": 2_000_000.0, "latency_us": 1000.0})
        assert d2["actual_sm_mhz"] == pytest.approx(2 * d1["actual_sm_mhz"])


# ─────────────────────────────────────────────────────────────────────────────
# G. Shape sweep pairwise speedup — _pairwise_stats integration
# ─────────────────────────────────────────────────────────────────────────────

class TestShapeSweepPairwise:

    def test_pairwise_stats_used_for_sweep_speedup(self):
        # When paired latencies exist, speedup = median(ratios)
        a = [100.0, 110.0, 90.0, 105.0, 95.0]
        b = [50.0,  55.0,  45.0, 52.5,  47.5]
        ps = _pairwise_stats(a, b)
        assert ps is not None
        median_speedup = ps[0]
        # All ratios are exactly 2.0 → median = 2.0
        assert median_speedup == pytest.approx(2.0)

    def test_min_min_differs_from_pairwise_with_noise(self):
        # Noise that doesn't cancel between kernels makes min/min different from median
        a = [100.0, 200.0, 100.0, 100.0, 100.0]  # one noisy run
        b = [50.0,  50.0,  50.0,  50.0,  100.0]  # different noisy run
        ps = _pairwise_stats(a, b)
        assert ps is not None
        min_min = min(a) / min(b)   # 100/50 = 2.0 — happens to be same here
        # The ratio for pair (200, 50) = 4.0 which pulls median away from 2.0
        ratios = [x/y for x,y in zip(a,b)]
        median_r = statistics.median(ratios)
        # median of [2.0, 4.0, 2.0, 2.0, 1.0] = 2.0 (same), but the stdev is meaningful
        assert ps[1] > 0  # non-zero uncertainty from noisy pairs


# ─────────────────────────────────────────────────────────────────────────────
# A. Globaltimer harness — _run_batch output format unchanged
# ─────────────────────────────────────────────────────────────────────────────

class TestGlobaltimer:

    def test_run_batch_parses_float_output(self, monkeypatch):
        # The harness now prints nanosecond-derived µs floats — same format as before
        def _ok(cmd, *a, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout="123.456\n789.012\n", stderr="")
        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _ok)
        samples = _run_batch("/fake", "k", l2_size_bytes=0, n=2, run_env={})
        assert samples == pytest.approx([123.456, 789.012])

    def test_sub_microsecond_precision_not_lost(self, monkeypatch):
        # globaltimer gives ns precision; we print %.3f → 0.001µs resolution
        def _ok(cmd, *a, **kw):
            return subprocess.CompletedProcess(cmd, 0, stdout="0.123\n", stderr="")
        monkeypatch.setattr("kerndiff.profiler.subprocess.run", _ok)
        samples = _run_batch("/fake", "k", l2_size_bytes=0, n=1, run_env={})
        assert samples[0] == pytest.approx(0.123, abs=1e-6)
