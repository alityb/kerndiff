from pathlib import Path

import kerndiff

from kerndiff.parser import parse_ncu_csv


FIXTURES = Path(kerndiff.__file__).resolve().parent / "fixtures"


def test_v1_fixture_parses():
    metrics = parse_ncu_csv((FIXTURES / "v1_ncu.csv").read_text())
    assert metrics["latency_us"] == 247.3
    assert metrics["l2_hit_rate"] == 41.2


def test_nsecond_converts_to_us():
    metrics = parse_ncu_csv(
        '"Metric Name","Metric Unit","Metric Value"\n"gpu__time_duration.sum","nsecond","247300"\n'
    )
    assert metrics["latency_us"] == 247.3


def test_commas_are_stripped():
    metrics = parse_ncu_csv(
        '"Metric Name","Metric Unit","Metric Value"\n"smsp__inst_executed.sum","inst","312,847"\n'
    )
    assert metrics["inst_executed"] == 312847.0


def test_unknown_metric_skipped():
    metrics = parse_ncu_csv(
        '"Metric Name","Metric Unit","Metric Value"\n"unknown_metric","inst","7"\n'
    )
    assert metrics == {}


def test_empty_value_skipped():
    metrics = parse_ncu_csv(
        '"Metric Name","Metric Unit","Metric Value"\n"smsp__inst_executed.sum","inst",""\n'
    )
    assert metrics == {}
