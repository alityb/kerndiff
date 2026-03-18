from kerndiff.compiler import _format_compile_error, DTYPE_MAP


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
