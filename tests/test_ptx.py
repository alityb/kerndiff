from kerndiff.ptx import diff_ptx, parse_ptx_instructions


def test_parse_ptx_skips_comments_directives_and_labels():
    text = """
// comment
.version 8.0
label:
ld.global.f32 %f1, [%rd1];
fma.rn.f32 %f2, %f3, %f4, %f5;
ret;
"""
    counts = parse_ptx_instructions(text)
    assert counts["ld.global"] == 1
    assert counts["fma.rn"] == 1
    assert counts["ret"] == 1


def test_diff_ptx_missing_v2_instruction_included():
    rows = diff_ptx({"ld.global": 10}, {})
    assert rows[0]["v2"] == 0
    assert rows[0]["delta_pct"] == -100.0


def test_diff_ptx_unchanged_excluded():
    rows = diff_ptx({"ld.global": 10}, {"ld.global": 10})
    assert rows == []


def test_diff_ptx_zero_denominator_uses_one():
    rows = diff_ptx({"ld.global": 0}, {"ld.global": 5})
    assert rows[0]["delta_pct"] == 500.0
