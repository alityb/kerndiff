from __future__ import annotations

import csv
from io import StringIO

from kerndiff.metrics import METRICS_BY_NCU


def _find_csv_start(lines: list[str]) -> int:
    for i, line in enumerate(lines):
        if '"Metric Name"' in line or "Metric Name" in line.split(","):
            return i
    return 0


def _parse_row(row: dict) -> tuple[str, float] | None:
    raw_name = (row.get("Metric Name") or "").strip().strip('"')
    metric_def = METRICS_BY_NCU.get(raw_name)
    if metric_def is None:
        return None
    raw_value = (row.get("Metric Value") or "").strip().strip('"')
    if not raw_value:
        return None
    raw_unit = (row.get("Metric Unit") or "").strip().strip('"')
    try:
        value = float(raw_value.replace(",", ""))
    except ValueError:
        return None
    if raw_unit == "nsecond" or raw_unit == "ns":
        value /= 1000.0
    elif raw_unit == "byte/s" or raw_unit == "byte/second":
        value /= 1e9  # convert to GB/s
    if metric_def.ncu_scale != 1.0:
        value *= metric_def.ncu_scale
    # Clamp percentage metrics that can exceed 100 due to NCU replay
    if metric_def.key in {"global_load_eff", "l1_hit_rate", "thread_active_pct"}:
        value = max(0.0, min(100.0, value))
    return metric_def.key, value


def parse_ncu_csv(csv_text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    lines = csv_text.splitlines(keepends=True)
    csv_start = _find_csv_start(lines)
    csv_body = "".join(lines[csv_start:])
    reader = csv.DictReader(StringIO(csv_body))
    for row in reader:
        result = _parse_row(row)
        if result:
            metrics[result[0]] = result[1]
    return metrics


def parse_ncu_csv_pipeline(csv_text: str, launch_count: int) -> dict[str, float]:
    """Parse NCU CSV with multiple launches, summing metrics across all launches."""
    lines = csv_text.splitlines(keepends=True)
    csv_start = _find_csv_start(lines)
    csv_body = "".join(lines[csv_start:])
    reader = csv.DictReader(StringIO(csv_body))

    # Collect all values per metric key across launches
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    for row in reader:
        result = _parse_row(row)
        if result:
            key, value = result
            sums[key] = sums.get(key, 0.0) + value
            counts[key] = counts.get(key, 0) + 1

    # For rate metrics (bandwidth, throughput %), average; for counts, sum
    from kerndiff.metrics import METRICS_BY_KEY
    metrics: dict[str, float] = {}
    for key, total in sums.items():
        mdef = METRICS_BY_KEY.get(key)
        if mdef and mdef.unit in {"%", "GB/s", "int"}:
            # Rates/percentages: average across launches
            metrics[key] = total / max(counts.get(key, 1), 1)
        else:
            # Counts, latency: sum across launches
            metrics[key] = total
    return metrics
