from __future__ import annotations

from kerndiff.diff import MetricDelta


def generate_suggestions(deltas: list[MetricDelta], v2_metrics: dict[str, float]) -> list[str]:
    """Return ranked, actionable suggestions based on v2 metric state and deltas.

    Suggestions fire on absolute thresholds (what the kernel looks like now)
    and relative changes (what got worse).  Only non-empty suggestions are
    returned; the list is empty when nothing notable is detected.
    """
    m = v2_metrics
    by_key = {d.metric.key: d for d in deltas}

    def val(key: float, default: float = 0.0) -> float:
        return m.get(key, default)

    def worsened(key: str) -> bool:
        d = by_key.get(key)
        return d is not None and d.symbol in ("-", "--")

    suggestions: list[str] = []

    # Register spills — always a clear action
    if val("local_load_sectors") > 0 or val("local_store_sectors") > 0:
        suggestions.append(
            "Register spilling to local memory — reduce register pressure "
            "or use --maxrregcount to trade occupancy for register count."
        )

    # Tensor core opportunity
    if val("tensor_core_util") == 0 and val("flops_tflops") > 0:
        suggestions.append(
            "No tensor core usage detected despite FP ops — "
            "consider WMMA / mma.sync for matrix-multiply patterns."
        )

    # Memory latency bottleneck
    if val("stall_memory") > 20 or worsened("stall_memory"):
        if val("l2_hit_rate") < 50:
            suggestions.append(
                "High DRAM stalls + low L2 hit rate — improve data locality: "
                "add tiling, transpose accesses, or prefetching."
            )
        else:
            suggestions.append(
                "High DRAM stalls with decent L2 hit rate — "
                "consider memory-access coalescing or reducing working set."
            )

    # Branch divergence
    if val("branch_divergence") > 10 or worsened("branch_divergence"):
        suggestions.append(
            "Warp divergence detected — restructure conditionals for warp-uniformity "
            "or use predication instead of branches."
        )

    # Low warp execution efficiency
    if val("warp_exec_eff") > 0 and val("warp_exec_eff") < 60:
        suggestions.append(
            f"Low warp execution efficiency ({val('warp_exec_eff'):.0f}%) — "
            "threads are idle within warps; check for irregular work distribution."
        )

    # Low occupancy from register pressure
    if val("sm_occupancy") < 40 and val("registers_per_thread") > 64:
        suggestions.append(
            f"Low occupancy ({val('sm_occupancy'):.0f}%) with high register count "
            f"({int(val('registers_per_thread'))}/thread) — "
            "use --maxrregcount to trade register reuse for occupancy."
        )

    # Shared memory bank conflicts
    total_conflicts = val("l1_bank_conflicts_rd") + val("l1_bank_conflicts_wr")
    if total_conflicts > 0 and (worsened("l1_bank_conflicts_rd") or worsened("l1_bank_conflicts_wr")):
        suggestions.append(
            "Shared memory bank conflicts increased — "
            "pad shared arrays by one element or reorder access patterns."
        )

    # SM load imbalance
    if val("sm_imbalance") > 0 and val("sm_imbalance") < 70:
        suggestions.append(
            f"SM load imbalance proxy is low ({val('sm_imbalance'):.0f}%) — "
            "some SMs may be underutilised; check grid dimensions and work distribution."
        )

    # Math pipe throttle (compute-bound by a specific pipe)
    if val("stall_pipe_busy") > 15 or worsened("stall_pipe_busy"):
        suggestions.append(
            "Math pipe throttle stalls are significant — "
            "consider instruction-level parallelism or mixing operation types."
        )

    return suggestions
