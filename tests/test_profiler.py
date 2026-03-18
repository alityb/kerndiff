from kerndiff.profiler import GPU_L2_SIZES, query_l2_size


def test_l2_size_a10g():
    size = query_l2_size(0, gpu_name="NVIDIA A10G")
    assert size == 6 * 1024 * 1024


def test_l2_size_h100():
    size = query_l2_size(0, gpu_name="NVIDIA H100 SXM5 80GB")
    assert size == 50 * 1024 * 1024


def test_l2_size_unknown_falls_back():
    size = query_l2_size(99, gpu_name="Unknown GPU XYZ")
    # Should return a fallback (6MB) since nvidia-smi will fail for gpu_id=99
    assert size > 0
