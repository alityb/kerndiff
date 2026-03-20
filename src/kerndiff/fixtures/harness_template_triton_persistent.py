# kerndiff Triton persistent benchmark harness (auto-generated)
# Stays alive across all timed runs. Communicates via stdin/stdout pipes.
# Protocol: kerndiff sends "time" or "quit"; harness responds with latency in us.
import sys, pathlib, torch

{{KERNEL_SOURCE}}

_DTYPE = {{TORCH_DTYPE}}
_N = {{BUF_ELEMS}}
_L2_BYTES = {{L2_FLUSH_BYTES}}
_PTX_OUT = "{{PTX_OUTPUT_PATH}}"
_WARMUP = {{WARMUP}}

# Allocate buffers — non-trivial input so zero-output kernels are detectable
_idx = torch.arange(_N, device="cuda") % 64 + 1
x = _idx.to(_DTYPE)
y = _idx.to(_DTYPE)
z = torch.zeros(_N, dtype=_DTYPE, device="cuda")

# L2 flush buffer — allocated once, reused between runs
_flush_buf = None
if _L2_BYTES > 0:
    _flush_buf = torch.empty(_L2_BYTES, dtype=torch.int8, device="cuda")


def _flush_l2():
    if _flush_buf is not None:
        _flush_buf.zero_()
        torch.cuda.synchronize()


def _time_kernel():
    # Saturate GPU queue before timing so CPU cannot outrun GPU for fast kernels.
    # This matches the approach used by triton.testing.do_bench.
    torch.cuda._sleep(1_000_000)
    _t0 = torch.cuda.Event(enable_timing=True)
    _t1 = torch.cuda.Event(enable_timing=True)
    _t0.record()
    {{KERNEL_CALL}}
    _t1.record()
    torch.cuda.synchronize()
    return _t0.elapsed_time(_t1) * 1000  # ms -> us


# Warmup — triggers Triton JIT compilation
for _ in range(_WARMUP):
    {{KERNEL_CALL}}
torch.cuda.synchronize()

# Extract PTX after warmup (kernel is compiled at this point)
if _PTX_OUT:
    try:
        _fn = {{KERNEL_NAME}}
        _ptx_found = False
        if hasattr(_fn, 'device_caches'):
            for _dev, _binder in _fn.device_caches.items():
                _cdict = _binder[0] if isinstance(_binder, tuple) else _binder
                if hasattr(_cdict, 'items'):
                    for _k, _compiled in _cdict.items():
                        if hasattr(_compiled, 'asm') and 'ptx' in _compiled.asm:
                            pathlib.Path(_PTX_OUT).write_text(_compiled.asm['ptx'])
                            _ptx_found = True
                            break
                if _ptx_found:
                    break
        if not _ptx_found and hasattr(_fn, 'cache') and _fn.cache:
            _key = next(iter(_fn.cache))
            pathlib.Path(_PTX_OUT).write_text(_fn.cache[_key].asm['ptx'])
    except Exception as _e:
        sys.stderr.write(f"warning: PTX extraction failed: {_e}\n")

# Signal ready to kerndiff (warmup and PTX extraction complete)
print("ready", flush=True)

# Main command loop — process stays alive until "quit" or EOF
while True:
    _cmd = sys.stdin.readline()
    if not _cmd:
        break  # EOF — kerndiff disconnected
    _cmd = _cmd.strip()
    if _cmd == "time":
        _flush_l2()
        _us = _time_kernel()
        print(f"{_us:.3f}", flush=True)
    elif _cmd == "dump":
        torch.cuda.synchronize()
        _vals = z[:16].cpu().tolist()
        print(" ".join(f"{_v:.6f}" for _v in _vals), flush=True)
    elif _cmd == "quit":
        break
    elif _cmd:
        sys.stderr.write(f"unknown command: {_cmd}\n")
        print("error", flush=True)
