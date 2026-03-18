import argparse

from kerndiff.config import apply_config, find_config, load_config


def test_find_config_returns_none_when_absent(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    assert find_config() is None


def test_find_config_finds_file_in_cwd(tmp_path, monkeypatch):
    (tmp_path / "kerndiff.toml").write_text('[defaults]\nfn = "test"\n')
    monkeypatch.chdir(tmp_path)
    result = find_config()
    assert result is not None
    assert result.name == "kerndiff.toml"


def test_find_config_stops_at_git_root(tmp_path, monkeypatch):
    # Create a nested structure with .git at parent
    (tmp_path / ".git").mkdir()
    sub = tmp_path / "sub" / "deep"
    sub.mkdir(parents=True)
    monkeypatch.chdir(sub)
    # No kerndiff.toml anywhere → should return None (stopped at .git)
    assert find_config() is None


def test_find_config_walks_up_to_parent(tmp_path, monkeypatch):
    (tmp_path / ".git").mkdir()
    (tmp_path / "kerndiff.toml").write_text('[defaults]\nfn = "parent"\n')
    sub = tmp_path / "src"
    sub.mkdir()
    monkeypatch.chdir(sub)
    result = find_config()
    assert result is not None


def test_load_config_parses_toml(tmp_path):
    toml_file = tmp_path / "kerndiff.toml"
    toml_file.write_text('[defaults]\nfn = "my_kernel"\nelems = 2048\n')
    config = load_config(toml_file)
    assert config["defaults"]["fn"] == "my_kernel"
    assert config["defaults"]["elems"] == 2048


def test_apply_config_fills_defaults():
    args = argparse.Namespace(
        fn_name=None, dtype="float", elems=1 << 22, noise_threshold=1.0,
        max_runs=50, min_runs=10, warmup=32, arch="sm_90", call_expr=None, pipeline=1,
    )
    config = {"defaults": {"fn": "scan_kernel", "elems": 8192}}
    result = apply_config(args, config)
    assert result.fn_name == "scan_kernel"
    assert result.elems == 8192


def test_apply_config_cli_wins():
    args = argparse.Namespace(
        fn_name="cli_kernel", dtype="half", elems=1 << 22, noise_threshold=1.0,
        max_runs=50, min_runs=10, warmup=32, arch="sm_90", call_expr=None, pipeline=1,
    )
    config = {"defaults": {"fn": "config_kernel", "dtype": "float"}}
    result = apply_config(args, config)
    # CLI-set fn_name should win
    assert result.fn_name == "cli_kernel"
    # CLI-set dtype should win
    assert result.dtype == "half"


def test_apply_config_kernel_section_overrides_defaults():
    args = argparse.Namespace(
        fn_name=None, dtype="float", elems=1 << 22, noise_threshold=1.0,
        max_runs=50, min_runs=10, warmup=32, arch="sm_90", call_expr=None, pipeline=1,
    )
    config = {
        "defaults": {"fn": "scan_kernel", "elems": 4096},
        "kernels": {"scan_kernel": {"elems": 8192, "call": "scan_kernel<<<32,128>>>(d_a,d_b,d_c,N)"}},
    }
    result = apply_config(args, config, kernel_name="scan_kernel")
    assert result.elems == 8192
    assert result.call_expr == "scan_kernel<<<32,128>>>(d_a,d_b,d_c,N)"
