# triton_scan_v1.py — associative scan using explicit tl.associative_scan API
# kerndiff examples/triton_scan_v1.py examples/triton_scan_v2.py --fn prefix_scan
#
# v1: explicit tl.associative_scan with user-supplied combine function
# v2: tl.cumsum shorthand (same hardware path, slightly less code)
# Both compute an inclusive prefix sum within each BLOCK_SIZE-element tile.
#
# Note: the original version of this file attempted Hillis-Steele by reading
# from a_ptr in each stride pass. That is incorrect — it reads the original
# input rather than running partial sums, producing wrong results.
# tl.associative_scan is the correct primitive for parallel scans in Triton.
import triton
import triton.language as tl


@triton.jit
def _add(a, b):
    return a + b


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
    # Explicit associative scan — same hardware path as tl.cumsum
    x = tl.associative_scan(x, 0, _add)
    tl.store(c_ptr + offsets, x, mask=mask)
