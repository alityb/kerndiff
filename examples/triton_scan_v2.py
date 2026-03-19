# triton_scan_v2.py — tl.cumsum-based scan in Triton
# kerndiff examples/triton_scan_v1.py examples/triton_scan_v2.py --fn prefix_scan
import triton
import triton.language as tl


@triton.jit
def prefix_scan(
    a_ptr, b_ptr, c_ptr,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(a_ptr + offsets, mask=mask, other=0.0)
    x = tl.cumsum(x, axis=0)
    tl.store(c_ptr + offsets, x, mask=mask)
