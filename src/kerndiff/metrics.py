from __future__ import annotations

from dataclasses import dataclass, field
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
    ncu_scale: float = 1.0   # multiply raw NCU value by this before storing
    hidden: bool = False     # if True, collected but not shown in table/deltas


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


def fmt_fb(v: float) -> str:
    return f"{v:.1f}"


def fmt_tflops(v: float) -> str:
    return f"{v:.2f}" if v >= 0.01 else f"{v:.4f}"


METRICS: list[MetricDef] = [
    # --- Speed of Light ---
    MetricDef("latency_us", "latency", "gpu__time_duration.sum", "us", "sol", True, fmt_us),
    MetricDef(
        "sm_throughput", "sm_throughput",
        "sm__throughput.avg.pct_of_peak_sustained_elapsed",
        "%", "sol", False, fmt_pct,
    ),
    MetricDef(
        "memory_throughput", "memory_throughput",
        "gpu__compute_memory_throughput.avg.pct_of_peak_sustained_elapsed",
        "%", "sol", False, fmt_pct,
    ),
    MetricDef("dram_bw_gbs", "dram_bw", "dram__bytes.sum.per_second", "GB/s", "sol", False, fmt_gbs),

    # --- Arithmetic ---
    # arith_intensity and flops_tflops are derived (computed from raw counters).
    # ncu_metric="" means they are NOT requested from NCU directly.
    MetricDef("arith_intensity", "arith_intensity", "", "F/B", "arithmetic", False, fmt_fb),
    MetricDef("flops_tflops", "flops", "", "TF", "arithmetic", False, fmt_tflops),
    MetricDef(
        "thread_active_pct", "thread_active",
        "smsp__average_thread_inst_executed_pred_on_per_inst_executed.ratio",
        "%", "arithmetic", False, fmt_pct,
        ncu_scale=100.0 / 32.0,  # NCU returns 0–32 ratio; scale to 0–100%
    ),

    # --- Cache ---
    MetricDef("l2_hit_rate", "l2_hit_rate", "lts__t_sector_hit_rate.pct", "%", "cache", False, fmt_pct),
    MetricDef("l1_hit_rate", "l1_hit_rate", "l1tex__t_sector_hit_rate.pct", "%", "cache", False, fmt_pct),
    MetricDef(
        "l1_bank_conflicts", "l1_bank_conflicts",
        "l1tex__data_bank_conflicts_pipe_lsu_mem_shared.sum",
        "count", "cache", True, fmt_k,
    ),
    MetricDef(
        "global_load_eff", "global_load_eff",
        "smsp__sass_average_data_bytes_per_sector_mem_global_op_ld.pct",
        "%", "cache", False, fmt_pct,
    ),

    # --- Warp State ---
    MetricDef(
        "sm_occupancy", "sm_occupancy",
        "smsp__warps_active.avg.pct_of_peak_sustained_active",
        "%", "warp_state", False, fmt_pct,
    ),
    MetricDef(
        "stall_memory", "stall_memory",
        "smsp__warp_issue_stalled_long_scoreboard_per_warp_active.pct",
        "%", "warp_state", True, fmt_pct,
    ),
    MetricDef(
        "stall_memqueue", "stall_memqueue",
        "smsp__warp_issue_stalled_lg_throttle_per_warp_active.pct",
        "%", "warp_state", True, fmt_pct,
    ),
    MetricDef(
        "stall_compute", "stall_compute",
        "smsp__warp_issue_stalled_short_scoreboard_per_warp_active.pct",
        "%", "warp_state", True, fmt_pct,
    ),
    MetricDef(
        "stall_sync", "stall_sync",
        "smsp__warp_issue_stalled_barrier_per_warp_active.pct",
        "%", "warp_state", True, fmt_pct,
    ),

    # --- Launch ---
    MetricDef(
        "registers_per_thread", "register_count",
        "launch__registers_per_thread",
        "int", "launch", True, fmt_int,
    ),
    MetricDef(
        "shared_mem_kb", "shared_mem",
        "launch__shared_mem_per_block_static",
        "int", "launch", None, fmt_kb,
    ),

    # --- Raw counters (hidden — collected for derived metrics, not shown in table) ---
    MetricDef("raw_ffma", "raw_ffma", "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum", "count", "raw", None, fmt_int, hidden=True),
    MetricDef("raw_fadd", "raw_fadd", "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum", "count", "raw", None, fmt_int, hidden=True),
    MetricDef("raw_fmul", "raw_fmul", "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum", "count", "raw", None, fmt_int, hidden=True),
    MetricDef("raw_hfma", "raw_hfma", "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum", "count", "raw", None, fmt_int, hidden=True),
    MetricDef("raw_hadd", "raw_hadd", "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum", "count", "raw", None, fmt_int, hidden=True),
    MetricDef("raw_hmul", "raw_hmul", "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum", "count", "raw", None, fmt_int, hidden=True),
    MetricDef("raw_dram_sectors_rd", "raw_dram_rd", "dram__sectors_read.sum", "count", "raw", None, fmt_int, hidden=True),
    MetricDef("raw_dram_sectors_wr", "raw_dram_wr", "dram__sectors_write.sum", "count", "raw", None, fmt_int, hidden=True),
    # Dynamic instruction count — hidden; kept for reference
    MetricDef("inst_executed", "inst_executed", "smsp__inst_executed.sum", "count", "raw", True, fmt_k, hidden=True),
]

METRICS_BY_KEY = {m.key: m for m in METRICS}
# Only include metrics that have an actual NCU metric string (exclude derived metrics)
METRICS_BY_NCU = {m.ncu_metric: m for m in METRICS if m.ncu_metric}
