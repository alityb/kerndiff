"""Test NCU CSV parser with malformed input."""
from kerndiff.parser import parse_ncu_csv


def test_empty_string():
    assert parse_ncu_csv("") == {}


def test_only_whitespace():
    assert parse_ncu_csv("   \n\n\t  ") == {}


def test_prefix_lines_before_csv():
    csv = (
        '==PROF== Connected to process 1234\n'
        '==PROF== Profiling "my_kernel"\n'
        '"Metric Name","Metric Unit","Metric Value"\n'
        '"gpu__time_duration.sum","nsecond","12345.6"\n'
    )
    result = parse_ncu_csv(csv)
    assert "latency_us" in result


def test_comma_in_value():
    csv = (
        '"Metric Name","Metric Unit","Metric Value"\n'
        '"sm__throughput.avg.pct_of_peak_sustained_elapsed","%","1,234.5"\n'
    )
    result = parse_ncu_csv(csv)
    assert "sm_throughput" in result
    assert result["sm_throughput"] == 1234.5


def test_missing_metric_value_column():
    csv = '"Metric Name","Metric Unit"\n"sm__throughput.avg.pct_of_peak_sustained_elapsed","%"\n'
    result = parse_ncu_csv(csv)
    assert result == {}


def test_non_numeric_value():
    csv = (
        '"Metric Name","Metric Unit","Metric Value"\n'
        '"sm__throughput.avg.pct_of_peak_sustained_elapsed","%","N/A"\n'
    )
    result = parse_ncu_csv(csv)
    assert result == {}


def test_duplicate_metric_names():
    csv = (
        '"Metric Name","Metric Unit","Metric Value"\n'
        '"gpu__time_duration.sum","nsecond","100"\n'
        '"gpu__time_duration.sum","nsecond","200"\n'
    )
    result = parse_ncu_csv(csv)
    # Should not crash — last value wins
    assert "latency_us" in result


def test_very_large_value():
    csv = (
        '"Metric Name","Metric Unit","Metric Value"\n'
        '"gpu__time_duration.sum","nsecond","99999999999999"\n'
    )
    result = parse_ncu_csv(csv)
    assert "latency_us" in result
    # Shouldn't crash on very large numbers


def test_byte_per_second_conversion():
    csv = (
        '"Metric Name","Metric Unit","Metric Value"\n'
        '"dram__bytes.sum.per_second","byte/second","500000000000"\n'
    )
    result = parse_ncu_csv(csv)
    assert "dram_bw_gbs" in result
    assert abs(result["dram_bw_gbs"] - 500.0) < 0.1


def test_nsecond_conversion():
    csv = (
        '"Metric Name","Metric Unit","Metric Value"\n'
        '"gpu__time_duration.sum","nsecond","5000"\n'
    )
    result = parse_ncu_csv(csv)
    assert "latency_us" in result
    assert abs(result["latency_us"] - 5.0) < 0.01
