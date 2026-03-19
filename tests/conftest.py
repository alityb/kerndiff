import pytest

from kerndiff.profiler import MOCK_HARDWARE, ProfileResult


def make_result(latency_us=247.0, overrides=None):
    # arith_intensity = 2*134217728 / ((1048576+524288)*32) = 268435456/50331648 = 5.33 F/B
    # flops_tflops = 268435456 / (247.0e-6) / 1e12 = 1.087 TF
    metrics = {
        "latency_us": latency_us,
        "sm_throughput": 61.3,
        "memory_throughput": 72.1,
        "dram_bw_gbs": 412.3,
        "arith_intensity": 5.33,
        "flops_tflops": 1.09,
        "thread_active_pct": 94.1,
        "l2_hit_rate": 41.2,
        "l1_hit_rate": 38.1,
        "l1_bank_conflicts": 124132,
        "global_load_eff": 79.3,
        "sm_occupancy": 62.4,
        "stall_memory": 22.5,
        "stall_memqueue": 18.2,
        "stall_compute": 4.1,
        "stall_sync": 2.3,
        "registers_per_thread": 64,
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
        clean_latencies_us=lats,
        median_latency_us=latency_us * 1.095,
        p20_latency_us=lats[int(len(lats) * 0.2)],
        p80_latency_us=lats[int(len(lats) * 0.8)],
        cv_pct=cv,
        n_outliers=0,
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
            "sm_throughput": 79.4,
            "memory_throughput": 89.3,
            "dram_bw_gbs": 509.1,
            "arith_intensity": 10.67,
            "flops_tflops": 1.42,
            "thread_active_pct": 96.3,
            "l2_hit_rate": 67.4,
            "l1_hit_rate": 71.2,
            "l1_bank_conflicts": 297800,
            "sm_occupancy": 51.2,
            "stall_memory": 7.1,
            "stall_memqueue": 3.2,
            "stall_compute": 8.9,
            "stall_sync": 12.4,
            "registers_per_thread": 72,
            "shared_mem_kb": 32768,
        },
    )
