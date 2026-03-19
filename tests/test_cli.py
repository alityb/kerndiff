import io
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path

import pytest

from kerndiff.cli import main, _scan_kernels


SINGLE_KERNEL = """
__global__ void chunked_scan_kernel(float* a, float* b, float* c, int n) {}
"""

MULTI_KERNEL = """
__global__ void chunked_scan_kernel(float* a, float* b, float* c, int n) {}
__global__ void prefill_kernel(float* a, float* b, float* c, int n) {}
"""


def test_fn_auto_detect_single_matching_kernel(tmp_path):
    file_a = tmp_path / "v1.cu"
    file_b = tmp_path / "v2.cu"
    file_a.write_text(SINGLE_KERNEL)
    file_b.write_text(SINGLE_KERNEL)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", str(file_a), str(file_b)])
    assert rc == 0
    assert "could not auto-detect" not in stderr.getvalue()


def test_fn_auto_detect_error_lists_kernels(tmp_path):
    file_a = tmp_path / "v1.cu"
    file_b = tmp_path / "v2.cu"
    file_a.write_text(MULTI_KERNEL)
    file_b.write_text(SINGLE_KERNEL)
    with pytest.raises(SystemExit) as exc:
        main(["--mock", str(file_a), str(file_b)])
    message = str(exc.value)
    assert "error: could not auto-detect kernel" in message
    assert "prefill_kernel" in message
    assert "chunked_scan_kernel" in message


def test_single_file_mock_mode_skips_git(tmp_path):
    file_a = tmp_path / "kernel.cu"
    file_a.write_text(SINGLE_KERNEL)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", str(file_a), "--fn", "chunked_scan_kernel"])
    assert rc == 0
    assert "mock mode -- no GPU required." in stderr.getvalue()


def test_all_and_fn_mutually_exclusive():
    with pytest.raises(SystemExit) as exc:
        main(["--mock", "v1.cu", "v2.cu", "--fn", "kernel", "--all"])
    assert "mutually exclusive" in str(exc.value)


def test_all_flag_profiles_common_kernels(tmp_path):
    file_a = tmp_path / "v1.cu"
    file_b = tmp_path / "v2.cu"
    file_a.write_text(MULTI_KERNEL)
    file_b.write_text(MULTI_KERNEL)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", str(file_a), str(file_b), "--all"])
    assert rc == 0
    output = stdout.getvalue()
    assert "chunked_scan_kernel" in output
    assert "prefill_kernel" in output


def test_all_flag_skips_non_common_kernels(tmp_path):
    file_a = tmp_path / "v1.cu"
    file_b = tmp_path / "v2.cu"
    file_a.write_text(MULTI_KERNEL)
    file_b.write_text(SINGLE_KERNEL)
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", str(file_a), str(file_b), "--all"])
    assert rc == 0
    err = stderr.getvalue()
    assert "skipping prefill_kernel" in err
    # Only one common kernel, so output is a diff (no header needed)
    assert "latency" in stdout.getvalue()


def test_all_flag_errors_when_no_kernels_in_v1(tmp_path):
    file_a = tmp_path / "v1.cu"
    file_b = tmp_path / "v2.cu"
    file_a.write_text("// no kernels here\n")
    file_b.write_text(SINGLE_KERNEL)
    with pytest.raises(SystemExit) as exc:
        main(["--mock", str(file_a), str(file_b), "--all"])
    assert f"no kernels found in {file_a.name}" in str(exc.value)


def test_all_flag_errors_when_no_common_kernels(tmp_path):
    file_a = tmp_path / "v1.cu"
    file_b = tmp_path / "v2.cu"
    file_a.write_text("__global__ void only_a(float* a, float* b, float* c, int n) {}\n")
    file_b.write_text("__global__ void only_b(float* a, float* b, float* c, int n) {}\n")
    with pytest.raises(SystemExit) as exc:
        main(["--mock", str(file_a), str(file_b), "--all"])
    assert f"no kernels in common between {file_a.name} and {file_b.name}" in str(exc.value)


def test_dtype_flag_accepted():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel", "--dtype", "half"])
    assert rc == 0


def test_dtype_flag_invalid():
    with pytest.raises(SystemExit):
        main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel", "--dtype", "double"])


def test_elems_flag_accepted():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel", "--elems", "8388608"])
    assert rc == 0


def test_elems_in_json_output():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel", "--elems", "1048576", "--format", "json"])
    assert rc == 0
    import json
    payload = json.loads(stdout.getvalue())
    assert payload["config"]["buf_elems"] == 1048576


def test_pipeline_requires_call():
    with pytest.raises(SystemExit) as exc:
        main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel", "--pipeline", "3"])
    assert "--pipeline requires --call" in str(exc.value)


def test_pipeline_flag_accepted_with_call():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main([
            "--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel",
            "--pipeline", "3", "--call", "chunked_scan_kernel<<<1,128>>>(d_a, d_b, d_c, N)",
        ])
    assert rc == 0


def test_pipeline_json_has_pipeline_field():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main([
            "--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel",
            "--pipeline", "3", "--call", "chunked_scan_kernel<<<1,128>>>(d_a, d_b, d_c, N)",
            "--format", "json",
        ])
    assert rc == 0
    import json
    payload = json.loads(stdout.getvalue())
    assert payload["config"]["pipeline"] == 3


def test_correctness_flag_accepted_mock():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel", "--correctness"])
    assert rc == 0
    assert "correctness check" in stderr.getvalue()


def test_shape_flag_invalid_format():
    with pytest.raises(SystemExit) as exc:
        main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel", "--shape", "abc,def"])
    assert "positive integers" in str(exc.value)


def test_shape_flag_accepted_mock():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel", "--shape", "1024,2048"])
    assert rc == 0
    output = stdout.getvalue()
    assert "1024" in output
    assert "2048" in output


def test_mamba_examples_have_expected_kernels():
    examples = Path(__file__).resolve().parent.parent / "examples"
    unfused = _scan_kernels(str(examples / "mamba_unfused.cu"))
    fused = _scan_kernels(str(examples / "mamba_fused.cu"))
    assert "ssd_pipeline" in unfused
    assert "ssd_pipeline" in fused


def test_at_flag_with_two_files_errors():
    with pytest.raises(SystemExit) as exc:
        main(["--mock", "v1.cu", "v2.cu", "--fn", "k", "--at", "HEAD"])
    assert "--at only applies" in str(exc.value)


def test_validate_flag_accepted_mock():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel", "--validate", "--no-color"])
    assert rc == 0
    output = stdout.getvalue()
    assert "validate:" in output


def test_validate_mock_shows_ok():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "v1.cu", "v2.cu", "--fn", "chunked_scan_kernel", "--validate", "--no-color"])
    assert rc == 0
    assert "validate: ok" in stdout.getvalue()


def test_config_file_applied(tmp_path, monkeypatch):
    """Test that kerndiff.toml in cwd is found and applied."""
    toml = tmp_path / "kerndiff.toml"
    toml.write_text('[defaults]\nfn = "chunked_scan_kernel"\n')
    monkeypatch.chdir(tmp_path)

    file_a = tmp_path / "v1.cu"
    file_b = tmp_path / "v2.cu"
    file_a.write_text(MULTI_KERNEL)
    file_b.write_text(MULTI_KERNEL)

    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", str(file_a), str(file_b)])
    assert rc == 0
    assert "config:" in stderr.getvalue()


def test_resolve_git_baseline(tmp_path, monkeypatch):
    """Test resolve_git_baseline extracts HEAD version of a file."""
    import subprocess
    from kerndiff.cli import resolve_git_baseline
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init"], capture_output=True, cwd=tmp_path)
    subprocess.run(["git", "config", "user.email", "test@test.com"], capture_output=True, cwd=tmp_path)
    subprocess.run(["git", "config", "user.name", "Test"], capture_output=True, cwd=tmp_path)

    kernel = tmp_path / "kernel.cu"
    kernel.write_text("// original version\n" + SINGLE_KERNEL)
    subprocess.run(["git", "add", "kernel.cu"], capture_output=True, cwd=tmp_path)
    subprocess.run(["git", "commit", "-m", "init"], capture_output=True, cwd=tmp_path)

    # Modify working copy
    kernel.write_text("// modified version\n" + SINGLE_KERNEL)

    temp_path, display_label = resolve_git_baseline(str(kernel))
    assert "HEAD:" in display_label
    assert "kernel.cu" in display_label
    content = Path(temp_path).read_text()
    assert "// original version" in content
    assert "// modified version" not in content


def test_resolve_git_baseline_at_ref(tmp_path, monkeypatch):
    """Test resolve_git_baseline with --at ref."""
    import subprocess
    from kerndiff.cli import resolve_git_baseline
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init"], capture_output=True, cwd=tmp_path)
    subprocess.run(["git", "config", "user.email", "test@test.com"], capture_output=True, cwd=tmp_path)
    subprocess.run(["git", "config", "user.name", "Test"], capture_output=True, cwd=tmp_path)

    kernel = tmp_path / "kernel.cu"
    kernel.write_text(SINGLE_KERNEL)
    subprocess.run(["git", "add", "kernel.cu"], capture_output=True, cwd=tmp_path)
    subprocess.run(["git", "commit", "-m", "init"], capture_output=True, cwd=tmp_path)

    temp_path, display_label = resolve_git_baseline(str(kernel), at_ref="HEAD")
    assert Path(temp_path).exists()
    assert "kernel.cu" in display_label


def test_resolve_git_baseline_untracked_errors(tmp_path, monkeypatch):
    """Test that an untracked file gives a clear error."""
    import subprocess
    from kerndiff.cli import resolve_git_baseline
    monkeypatch.chdir(tmp_path)
    subprocess.run(["git", "init"], capture_output=True, cwd=tmp_path)
    subprocess.run(["git", "config", "user.email", "test@test.com"], capture_output=True, cwd=tmp_path)
    subprocess.run(["git", "config", "user.name", "Test"], capture_output=True, cwd=tmp_path)

    readme = tmp_path / "README"
    readme.write_text("init")
    subprocess.run(["git", "add", "README"], capture_output=True, cwd=tmp_path)
    subprocess.run(["git", "commit", "-m", "init"], capture_output=True, cwd=tmp_path)

    kernel = tmp_path / "kernel.cu"
    kernel.write_text(SINGLE_KERNEL)
    # Don't git add

    with pytest.raises(SystemExit) as exc:
        resolve_git_baseline(str(kernel))
    assert "not tracked by git" in str(exc.value)


def test_multi_kernel_non_tty_errors(tmp_path):
    """When not a tty with multiple kernels and no --fn, should error (not hang)."""
    file_a = tmp_path / "v1.cu"
    file_b = tmp_path / "v2.cu"
    file_a.write_text(MULTI_KERNEL)
    file_b.write_text(MULTI_KERNEL)
    with pytest.raises(SystemExit) as exc:
        main(["--mock", str(file_a), str(file_b)])
    assert "could not auto-detect kernel" in str(exc.value)


def test_mock_cli_numeric_display_formats():
    stdout = io.StringIO()
    stderr = io.StringIO()
    with redirect_stdout(stdout), redirect_stderr(stderr):
        rc = main(["--mock", "v1.cu", "v2.cu", "--fn", "k", "--no-color"])
    assert rc == 0
    out = stdout.getvalue() + stderr.getvalue()
    assert "\033[" not in out
    assert "-23.5%" in out          # latency (% delta)
    assert "+26.2pp" in out         # l2_hit_rate (pp)
    assert "+139.9%" in out         # l1_bank_conflicts (% delta)
    assert "+8" in out              # register_count (raw int diff)
    assert "+16" in out             # shared_mem (KB diff)
