from __future__ import annotations

import csv
from io import StringIO

from kerndiff.metrics import METRICS_BY_NCU


def parse_ncu_csv(csv_text: str) -> dict[str, float]:
    metrics: dict[str, float] = {}
    reader = csv.DictReader(StringIO(csv_text))
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
        if raw_unit == "nsecond":
            value /= 1000.0
        metrics[metric_def.key] = value
    return metrics
