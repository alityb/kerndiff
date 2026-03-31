from kerndiff.compiler import (
    _format_compile_error,
    DTYPE_MAP,
    build_harness,
    generate_call,
    infer_kernel_call,
    parse_kernel_signature,
    verify_correctness,
)


def test_format_compile_error_shows_source_name():
    msg = _format_compile_error(
        "examples/vec_add_v1.cu",
        'vec_add_v1.cu(12): error: identifier "half" is undefined',
        ["nvcc", "-O2", "-arch=sm_86", "-o", "/tmp/bench", "/tmp/bench.cu"],
        "/tmp/bench.cu",
    )
    assert "vec_add_v1.cu" in msg
    assert 'identifier "half" is undefined' in msg
    assert "--dtype half" in msg


def test_format_compile_error_undefined_symbol_hint():
    msg = _format_compile_error(
        "kernel.cu",
        'kernel.cu(5): error: identifier "my_func" is undefined',
        ["nvcc", "-o", "/tmp/bench", "/tmp/bench.cu"],
        "/tmp/bench.cu",
    )
    assert "--fn" in msg


def test_format_compile_error_shows_nvcc_command():
    msg = _format_compile_error(
        "kernel.cu",
        "some error",
        ["nvcc", "-O2", "-arch=sm_86", "-o", "/tmp/bench", "/tmp/bench.cu"],
        "/tmp/bench.cu",
    )
    assert "nvcc -O2 -arch=sm_86" in msg


def test_dtype_map_has_required_types():
    for dtype in ("float", "half", "int", "int4"):
        assert dtype in DTYPE_MAP
        elem_type, include = DTYPE_MAP[dtype]
        assert elem_type
        assert "#include" in include


def test_parse_kernel_signature_standard():
    source = '__global__ void my_kernel(float* a, float* b, float* c, int n) {}'
    params = parse_kernel_signature(source, "my_kernel")
    assert len(params) == 4
    assert params[0] == ("float*", "a")
    assert params[3] == ("int", "n")


def test_parse_kernel_signature_with_restrict():
    source = '__global__ void kern(float* __restrict__ a, const float* b, int stride) {}'
    params = parse_kernel_signature(source, "kern")
    assert len(params) == 3
    assert params[0][0] == "float*"
    assert params[2] == ("int", "stride")


def test_parse_kernel_signature_not_found():
    source = '__global__ void other_kernel(float* a) {}'
    params = parse_kernel_signature(source, "missing_kernel")
    assert params == []


def test_generate_call_standard():
    params = [("float*", "a"), ("float*", "b"), ("float*", "c"), ("int", "n")]
    call, warnings = generate_call("my_kernel", params)
    assert call == "my_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N)"
    assert len(warnings) == 0


def test_generate_call_with_stride():
    params = [("float*", "a"), ("float*", "b"), ("float*", "c"), ("int", "n"), ("int", "stride")]
    call, warnings = generate_call("kern", params)
    assert "d_a" in call
    assert "N" in call
    assert "1" in call  # stride mapped to 1


def test_generate_call_extra_pointers():
    params = [("float*", "a"), ("float*", "b"), ("float*", "c"), ("float*", "d"), ("int", "n")]
    call, warnings = generate_call("kern", params)
    assert "nullptr" in call
    assert len(warnings) > 0


def test_infer_kernel_call_reports_mode_and_warnings(tmp_path):
    source = tmp_path / "kernel.cu"
    source.write_text("__global__ void kern(float* a, float* b, float* c, float alpha) {}\n")
    call, warnings, mode = infer_kernel_call(str(source), "kern")
    assert mode == "inferred"
    assert "GRID_SIZE" in call
    assert any("alpha" in warning for warning in warnings)


def test_infer_kernel_call_falls_back_when_signature_missing(tmp_path):
    source = tmp_path / "kernel.cu"
    source.write_text("// no matching kernel\n")
    call, warnings, mode = infer_kernel_call(str(source), "kern")
    assert mode == "default"
    assert "d_a, d_b, d_c, N" in call
    assert warnings == ["could not parse kernel signature"]


def test_format_compile_error_shows_auto_call():
    msg = _format_compile_error(
        "kernel.cu",
        "some error",
        ["nvcc", "-o", "/tmp/bench", "/tmp/bench.cu"],
        "/tmp/bench.cu",
        auto_call="my_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N)",
    )
    assert "auto-generated call was" in msg
    assert "--call to override" in msg


def test_verify_correctness_parses_matching_output(tmp_path):
    """Test verify_correctness with scripts that output matching values."""
    script_a = tmp_path / "a.sh"
    script_b = tmp_path / "b.sh"
    script_a.write_text("#!/bin/bash\nfor i in $(seq 1 4); do echo \"1.0\"; done\n")
    script_b.write_text("#!/bin/bash\nfor i in $(seq 1 4); do echo \"1.0\"; done\n")
    script_a.chmod(0o755)
    script_b.chmod(0o755)
    max_diff, v1_vals, v2_vals = verify_correctness(str(script_a), str(script_b))
    assert max_diff == 0.0
    assert len(v1_vals) == 4


def test_verify_correctness_detects_mismatch(tmp_path):
    """Test verify_correctness detects when outputs differ."""
    script_a = tmp_path / "a.sh"
    script_b = tmp_path / "b.sh"
    script_a.write_text("#!/bin/bash\necho 1.0\necho 2.0\n")
    script_b.write_text("#!/bin/bash\necho 1.0\necho 3.0\n")
    script_a.chmod(0o755)
    script_b.chmod(0o755)
    max_diff, v1_vals, v2_vals = verify_correctness(str(script_a), str(script_b))
    assert max_diff == 1.0


def test_build_harness_does_not_replace_placeholders_inside_kernel_source(tmp_path):
    src = tmp_path / "k.cu"
    src.write_text(
        "__global__ void my_kernel(float* a, float* b, float* c, int n) {\n"
        "  // literal placeholder {{KERNEL_NAME}} should stay unchanged in source\n"
        "}\n"
    )
    harness_path = build_harness(
        str(src),
        "my_kernel",
        "my_kernel<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N)",
    )
    content = open(harness_path).read()
    assert "// literal placeholder {{KERNEL_NAME}}" in content
