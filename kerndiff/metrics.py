from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class MetricDef:
    key: str
    display_name: str
    ncu_metric: str
    unit: str
    group: str
    lower_is_better: bool | None
    format_fn: Callable[[float], str]


def fmt_us(v: float) -> str:
    return f"{v:.1f}us" if v >= 100 else f"{v:.2f}us"


def fmt_pct(v: float) -> str:
    return f"{v:.1f}%"


def fmt_gbs(v: float) -> str:
    return f"{v:.1f}"


def fmt_k(v: float) -> str:
    return f"{int(v / 1000)}K" if v >= 1000 else str(int(v))


def fmt_int(v: float) -> str:
    return str(int(v))


def fmt_kb(v: float) -> str:
    return f"{int(v / 1024)}KB" if v >= 1024 else f"{int(v)}B"


def fmt_inst(v: float) -> str:
    return f"{v:.1f}"


METRICS: list[MetricDef] = [
    MetricDef("latency_us", "latency", "gpu__time_duration.sum", "us", "latency", True, fmt_us),
    MetricDef(
        "sm_throughput",
        "sm_throughput",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "%",
        "latency",
        False,
        fmt_pct,
    ),
    MetricDef(
        "memory_throughput",
        "memory_throughput",
        "l1tex__t_bytes_pipe_lsu_mem_global_op_ld.sum.pct_of_peak_sustained_elapsed",
        "%",
        "latency",
        False,
        fmt_pct,
    ),
    MetricDef("l2_hit_rate", "l2_hit_rate", "l2cache__hit_rate.pct", "%", "memory", False, fmt_pct),
    MetricDef("l1_hit_rate", "l1_hit_rate", "l1tex__hit_rate.pct", "%", "memory", False, fmt_pct),
    MetricDef("dram_bw_gbs", "dram_bw", "dram__bytes.sum.per_second", "GB/s", "memory", False, fmt_gbs),
    MetricDef(
        "l1_bank_conflicts",
        "l1_bank_conflicts",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
        "count",
        "memory",
        True,
        fmt_k,
    ),
    MetricDef(
        "global_load_eff",
        "global_load_eff",
        "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
        "%",
        "memory",
        False,
        fmt_pct,
    ),
    MetricDef(
        "sm_occupancy",
        "sm_occupancy",
        "smsp__warps_active.avg.pct_of_peak_sustained_active",
        "%",
        "compute",
        False,
        fmt_pct,
    ),
    MetricDef(
        "warp_stall_mio",
        "warp_stall_mio",
        "smsp__average_warp_latency_per_inst_issued.ratio",
        "inst",
        "compute",
        True,
        fmt_inst,
    ),
    MetricDef(
        "warp_stall_lmem",
        "warp_stall_lmem",
        "smsp__warp_issue_stalled_local_mem_throttle_per_warp_active.pct",
        "%",
        "compute",
        True,
        fmt_pct,
    ),
    MetricDef(
        "warp_divergence",
        "warp_divergence",
        "smsp__branch_targets_threads_diverged.avg",
        "%",
        "compute",
        True,
        fmt_pct,
    ),
    MetricDef(
        "registers_per_thread",
        "register_count",
        "launch__registers_per_thread",
        "int",
        "code",
        True,
        fmt_int,
    ),
    MetricDef("ptx_instructions", "ptx_instructions", "inst_executed", "count", "code", True, fmt_int),
    MetricDef(
        "shared_mem_kb",
        "shared_mem",
        "launch__shared_mem_per_block_static",
        "int",
        "code",
        None,
        fmt_kb,
    ),
]

METRICS_BY_KEY = {m.key: m for m in METRICS}
METRICS_BY_NCU = {m.ncu_metric: m for m in METRICS}
