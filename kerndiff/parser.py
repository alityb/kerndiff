from __future__ import annotations

import csv
from io import StringIO

from kerndiff.metrics import METRICS_BY_NCU


def parse_ncu_csv(csv_text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    # NCU output may have non-CSV prefix lines (==PROF==, timing output, etc.)
    # Find the CSV header line and parse from there
    lines = csv_text.splitlines(keepends=True)
    csv_start = 0
    for i, line in enumerate(lines):
        if '"Metric Name"' in line or "Metric Name" in line.split(","):
            csv_start = i
            break
    csv_body = "".join(lines[csv_start:])
    reader = csv.DictReader(StringIO(csv_body))
    for row in reader:
        raw_name = (row.get("Metric Name") or "").strip().strip('"')
        metric_def = METRICS_BY_NCU.get(raw_name)
        if metric_def is None:
            continue
        raw_value = (row.get("Metric Value") or "").strip().strip('"')
        if not raw_value:
            continue
        raw_unit = (row.get("Metric Unit") or "").strip().strip('"')
        try:
            value = float(raw_value.replace(",", ""))
        except ValueError:
            continue
        if raw_unit == "nsecond" or raw_unit == "ns":
            value /= 1000.0
        elif raw_unit == "byte/s" or raw_unit == "byte/second":
            value /= 1e9  # convert to GB/s
        metrics[metric_def.key] = value
    return metrics
