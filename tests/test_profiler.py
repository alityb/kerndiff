import subprocess

from kerndiff.profiler import MOCK_HARDWARE, _remove_outliers, profile, query_l2_size


def test_l2_size_a10g():
    size = query_l2_size(0, gpu_name="NVIDIA A10G")
    assert size == 6 * 1024 * 1024


def test_l2_size_h100():
    size = query_l2_size(0, gpu_name="NVIDIA H100 SXM5 80GB")
    assert size == 50 * 1024 * 1024


def test_l2_size_unknown_falls_back():
    size = query_l2_size(99, gpu_name="Unknown GPU XYZ")
    # Should return a fallback (6MB) since nvidia-smi will fail for gpu_id=99
    assert size > 0


def test_ncu_permission_error_warning_format(monkeypatch):
    monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda gpu_id, gpu_name="": 6 * 1024 * 1024)
    monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *args, **kwargs: None)
    monkeypatch.setattr("kerndiff.profiler._run_timed_legacy", lambda *args, **kwargs: [100.0] * 10)
    monkeypatch.setattr("kerndiff.profiler._find_ncu", lambda: "/usr/bin/ncu")

    monkeypatch.setattr("kerndiff.profiler.shutil.which", lambda cmd: None)

    ncu_cmds = []

    def _fake_run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "/usr/bin/ncu":
            ncu_cmds.append(cmd)
            return subprocess.CompletedProcess(cmd, 1, stdout="ERR_NVGPUCTRPERM", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("kerndiff.profiler.subprocess.run", _fake_run)

    result = profile(
        binary="/tmp/fake_bin",
        kernel_name="k",
        max_runs=10,
        min_runs=10,
        noise_threshold=0.0,
        warmup=0,
        gpu_id=0,
        hardware=MOCK_HARDWARE,
        mock=False,
    )
    warning_text = "\n".join(result.warnings)
    assert "NCU requires elevated permissions on this system." in warning_text
    assert "perf_event_paranoid" in warning_text
    assert ncu_cmds
    assert "--launch-skip" in ncu_cmds[0]
    skip_idx = ncu_cmds[0].index("--launch-skip")
    assert ncu_cmds[0][skip_idx + 1] == "2"
    assert "--clock-control" in ncu_cmds[0]
    cc_idx = ncu_cmds[0].index("--clock-control")
    assert ncu_cmds[0][cc_idx + 1] == "none"


def test_remove_outliers_filters_large_spike():
    clean, n_removed = _remove_outliers([100.0, 101.0, 99.0, 100.5, 250.0])
    assert n_removed == 1
    assert clean == [100.0, 101.0, 99.0, 100.5]


def test_profile_outlier_warning_and_clean_latencies(monkeypatch):
    monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda gpu_id, gpu_name="": 6 * 1024 * 1024)
    monkeypatch.setattr("kerndiff.profiler.query_clock_telemetry", lambda gpu_id, hardware=None: {
        "current_sm_clock_mhz": 1800,
        "current_mem_clock_mhz": 2000,
        "max_sm_clock_mhz": 1800,
        "max_mem_clock_mhz": 2000,
        "throttle_reasons": [],
    })
    monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *args, **kwargs: None)
    monkeypatch.setattr("kerndiff.profiler._run_timed_legacy", lambda *args, **kwargs: [100.0, 101.0, 99.0, 100.0, 260.0])
    monkeypatch.setattr("kerndiff.profiler._find_ncu", lambda: None)

    result = profile(
        binary="/tmp/fake_bin",
        kernel_name="k",
        max_runs=10,
        min_runs=5,
        noise_threshold=0.0,
        warmup=0,
        gpu_id=0,
        hardware=MOCK_HARDWARE,
        mock=False,
    )
    assert result.n_outliers == 1
    assert len(result.all_latencies_us) == 5
    assert len(result.clean_latencies_us) == 4
    assert any("outlier run(s) detected" in w for w in result.warnings)


def test_profile_warns_when_throttle_reasons_active(monkeypatch):
    monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda gpu_id, gpu_name="": 6 * 1024 * 1024)
    monkeypatch.setattr("kerndiff.profiler.query_clock_telemetry", lambda gpu_id, hardware=None: {
        "current_sm_clock_mhz": 1200,
        "current_mem_clock_mhz": 1800,
        "max_sm_clock_mhz": 1800,
        "max_mem_clock_mhz": 2000,
        "throttle_reasons": ["sw_power_cap"],
    })
    monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *args, **kwargs: None)
    monkeypatch.setattr("kerndiff.profiler._run_timed_legacy", lambda *args, **kwargs: [100.0] * 5)
    monkeypatch.setattr("kerndiff.profiler._find_ncu", lambda: None)

    result = profile(
        binary="/tmp/fake_bin",
        kernel_name="k",
        max_runs=10,
        min_runs=5,
        noise_threshold=0.0,
        warmup=0,
        gpu_id=0,
        hardware=MOCK_HARDWARE,
        mock=False,
    )
    assert result.clock_telemetry["throttle_reasons"] == ["sw_power_cap"]
    assert any("clock-throttle reasons" in w for w in result.warnings)
