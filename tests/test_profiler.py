import subprocess

from kerndiff.profiler import MOCK_HARDWARE, profile, query_l2_size


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

    def _fake_run(cmd, *args, **kwargs):
        if cmd and cmd[0] == "/usr/bin/ncu":
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
