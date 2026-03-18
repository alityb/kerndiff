import pytest

from kerndiff.profiler import MOCK_HARDWARE, ProfileResult


def make_result(latency_us=247.0, overrides=None):
    metrics = {
        "latency_us": latency_us,
        "sm_throughput": 61.3,
        "memory_throughput": 72.1,
        "l2_hit_rate": 41.2,
        "l1_hit_rate": 38.1,
        "dram_bw_gbs": 412.3,
        "l1_bank_conflicts": 124132,
        "global_load_eff": 79.3,
        "sm_occupancy": 62.4,
        "warp_stall_mio": 18.2,
        "warp_stall_lmem": 3.1,
        "warp_divergence": 2.1,
        "registers_per_thread": 64,
        "ptx_instructions": 312847,
        "shared_mem_kb": 16384,
    }
    if overrides:
        metrics.update(overrides)
    lats = [latency_us * (1 + i * 0.01) for i in range(20)]
    from statistics import mean, stdev

    cv = (stdev(lats) / mean(lats)) * 100
    return ProfileResult(
        kernel_name="test_kernel",
        metrics=metrics,
        min_latency_us=latency_us,
        all_latencies_us=lats,
        cv_pct=cv,
        ptx_instructions={"ld.global": 48, "st.shared": 24, "fma.rn": 96},
        hardware=MOCK_HARDWARE,
        warnings=[],
        actual_runs=len(lats),
        max_runs=50,
        min_runs=10,
        noise_threshold=1.0,
        warmup=32,
        l2_flush=False,
    )


@pytest.fixture
def v1_result():
    return make_result(247.0)


@pytest.fixture
def v2_result():
    return make_result(
        189.1,
        {
            "l2_hit_rate": 67.4,
            "l1_bank_conflicts": 297800,
            "sm_occupancy": 51.2,
            "warp_stall_mio": 7.1,
            "registers_per_thread": 72,
            "ptx_instructions": 247300,
        },
    )
