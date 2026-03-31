import subprocess

from kerndiff.profiler import MOCK_HARDWARE, _remove_outliers, _run_batch, interleave_timing, profile, query_l2_size


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


def test_profile_uses_pre_collected_latencies(monkeypatch):
    """When pre_collected_latencies is passed, warmup and timing loop must be skipped."""
    monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda gpu_id, gpu_name="": 6 * 1024 * 1024)
    monkeypatch.setattr("kerndiff.profiler.query_clock_telemetry", lambda gpu_id, hardware=None: {
        "current_sm_clock_mhz": 1800,
        "current_mem_clock_mhz": 2000,
        "max_sm_clock_mhz": 1800,
        "max_mem_clock_mhz": 2000,
        "throttle_reasons": [],
    })
    monkeypatch.setattr("kerndiff.profiler._find_ncu", lambda: None)

    def _must_not_be_called(*args, **kwargs):
        raise AssertionError("timing/warmup functions must not be called when pre_collected_latencies is set")

    monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", _must_not_be_called)
    monkeypatch.setattr("kerndiff.profiler._run_timed_legacy", _must_not_be_called)

    pre = [100.0, 101.0, 99.5, 100.2, 100.8, 99.8, 100.1, 100.3, 99.7, 100.5]
    result = profile(
        binary="/tmp/fake_bin",
        kernel_name="k",
        max_runs=50,
        min_runs=10,
        noise_threshold=1.0,
        warmup=32,
        gpu_id=0,
        hardware=MOCK_HARDWARE,
        mock=False,
        pre_collected_latencies=pre,
    )
    assert result.all_latencies_us == pre
    assert result.actual_runs == len(pre)
    assert any(e.get("name") == "pre_collected_timing" for e in result.trace_events)


def test_profile_warns_on_ncu_latency_mismatch(monkeypatch):
    """When NCU reports a kernel duration >25% different from measured, warn."""
    monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda gpu_id, gpu_name="": 6 * 1024 * 1024)
    monkeypatch.setattr("kerndiff.profiler.query_clock_telemetry", lambda gpu_id, hardware=None: {
        "current_sm_clock_mhz": 1800, "current_mem_clock_mhz": 2000,
        "max_sm_clock_mhz": 1800, "max_mem_clock_mhz": 2000,
        "throttle_reasons": [],
    })
    monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *args, **kwargs: None)
    monkeypatch.setattr("kerndiff.profiler._run_timed_legacy", lambda *args, **kwargs: [100.0] * 10)
    monkeypatch.setattr("kerndiff.profiler._find_ncu", lambda: "/usr/bin/ncu")
    monkeypatch.setattr("kerndiff.profiler.shutil.which", lambda cmd: None)

    # NCU reports 200us but measured is 100us — big mismatch.
    ncu_csv = (
        '"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",'
        '"Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","1","./b","h","k","t","1","7","Speed","gpu__time_duration.sum","nsecond","200000"\n'
    )

    def _fake_run(cmd, *args, **kwargs):
        if cmd and "/ncu" in cmd[0]:
            return subprocess.CompletedProcess(cmd, 0, stdout=ncu_csv, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("kerndiff.profiler.subprocess.run", _fake_run)

    result = profile(
        binary="/tmp/fake_bin", kernel_name="k",
        max_runs=10, min_runs=10, noise_threshold=0.0,
        warmup=0, gpu_id=0, hardware=MOCK_HARDWARE, mock=False,
    )
    assert any("NCU-reported kernel duration" in w for w in result.warnings)


def test_profile_warns_on_register_spill(monkeypatch):
    """When NCU reports non-zero local memory sectors, emit a spill warning."""
    monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda gpu_id, gpu_name="": 6 * 1024 * 1024)
    monkeypatch.setattr("kerndiff.profiler.query_clock_telemetry", lambda gpu_id, hardware=None: {
        "current_sm_clock_mhz": 1800, "current_mem_clock_mhz": 2000,
        "max_sm_clock_mhz": 1800, "max_mem_clock_mhz": 2000,
        "throttle_reasons": [],
    })
    monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *args, **kwargs: None)
    monkeypatch.setattr("kerndiff.profiler._run_timed_legacy", lambda *args, **kwargs: [100.0] * 10)
    monkeypatch.setattr("kerndiff.profiler._find_ncu", lambda: "/usr/bin/ncu")
    monkeypatch.setattr("kerndiff.profiler.shutil.which", lambda cmd: None)

    ncu_csv = (
        '"ID","Process ID","Process Name","Host Name","Kernel Name","Kernel Time",'
        '"Context","Stream","Section Name","Metric Name","Metric Unit","Metric Value"\n'
        '"0","1","./b","h","k","t","1","7","Speed","gpu__time_duration.sum","nsecond","100000"\n'
        '"0","1","./b","h","k","t","1","7","Mem","smsp__l1tex_m_l1_read_sectors_pipe_lsu_mem_local_op_ld.sum","","512"\n'
        '"0","1","./b","h","k","t","1","7","Mem","smsp__l1tex_m_l1_write_sectors_pipe_lsu_mem_local_op_st.sum","","256"\n'
    )

    def _fake_run(cmd, *args, **kwargs):
        if cmd and "/ncu" in cmd[0]:
            return subprocess.CompletedProcess(cmd, 0, stdout=ncu_csv, stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("kerndiff.profiler.subprocess.run", _fake_run)

    result = profile(
        binary="/tmp/fake_bin", kernel_name="k",
        max_runs=10, min_runs=10, noise_threshold=0.0,
        warmup=0, gpu_id=0, hardware=MOCK_HARDWARE, mock=False,
    )
    assert any("register spilling" in w for w in result.warnings)


def test_run_batch_returns_n_samples(monkeypatch):
    """_run_batch must invoke --multi-time N and parse N latency lines."""
    calls = []

    def _fake_run(cmd, *args, **kwargs):
        calls.append(cmd)
        # Simulate harness printing N latency values
        n = int(cmd[cmd.index("--multi-time") + 1])
        output = "\n".join(f"{100.0 + i * 0.5:.3f}" for i in range(n))
        return subprocess.CompletedProcess(cmd, 0, stdout=output, stderr="")

    monkeypatch.setattr("kerndiff.profiler.subprocess.run", _fake_run)

    samples = _run_batch("/fake/bin", "k", l2_size_bytes=6 * 1024 * 1024, n=10, run_env={})
    assert len(samples) == 10
    assert "--multi-time" in calls[0]
    assert calls[0][calls[0].index("--multi-time") + 1] == "10"
    assert "--l2-flush" in calls[0]
    assert all(s > 0 for s in samples)


def test_interleave_timing_alternates_kernels(monkeypatch):
    """Both binaries must be called once per pair; lists must be equal length >= min_runs."""
    monkeypatch.setattr("kerndiff.profiler.query_l2_size", lambda gpu_id, gpu_name="": 6 * 1024 * 1024)
    monkeypatch.setattr("kerndiff.profiler._run_warmup_legacy", lambda *args, **kwargs: None)

    call_log: list[str] = []
    counters = {"a": 100.0, "b": 200.0}

    def _fake_run(cmd, *args, **kwargs):
        binary = cmd[0]
        if binary == "/fake/binary_a":
            call_log.append("a")
            counters["a"] += 0.1
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{counters['a']}", stderr="")
        elif binary == "/fake/binary_b":
            call_log.append("b")
            counters["b"] += 0.1
            return subprocess.CompletedProcess(cmd, 0, stdout=f"{counters['b']}", stderr="")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr("kerndiff.profiler.subprocess.run", _fake_run)

    latencies_a, latencies_b, warnings = interleave_timing(
        binary_a="/fake/binary_a",
        binary_b="/fake/binary_b",
        kernel_name="k",
        min_runs=10,
        max_runs=20,
        noise_threshold=5.0,
        warmup=0,
        gpu_id=0,
        hardware=MOCK_HARDWARE,
    )

    assert len(latencies_a) == len(latencies_b)
    assert len(latencies_a) >= 10
    assert call_log.count("a") == len(latencies_a)
    assert call_log.count("b") == len(latencies_b)
    assert all(v > 0 for v in latencies_a)
    assert all(v > 0 for v in latencies_b)
