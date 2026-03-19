# kerndiff Triton benchmark harness (auto-generated)
import os, sys, time, pathlib, torch

{{KERNEL_SOURCE}}

_DTYPE = {{TORCH_DTYPE}}
_N = {{BUF_ELEMS}}
_ITERS = {{ITERS}}
_L2_FLUSH = {{L2_FLUSH_BYTES}}
_PTX_OUT = "{{PTX_OUTPUT_PATH}}"
_KERNEL_NAME = "{{KERNEL_NAME}}"
_DUMP_OUTPUT = {{DUMP_OUTPUT}}

_idx = torch.arange(_N, device="cuda") % 64 + 1
x = _idx.to(_DTYPE)
y = _idx.to(_DTYPE)
z = torch.zeros(_N, dtype=_DTYPE, device="cuda")

# Warmup (triggers Triton JIT compilation)
for _ in range(32):
    {{KERNEL_CALL}}
torch.cuda.synchronize()

# Dump PTX after warmup (kernel is compiled at this point)
if _PTX_OUT:
    try:
        _fn = {{KERNEL_NAME}}
        _ptx_found = False
        # Triton 3.6+: device_caches[device_id] -> (compiled_dict, ...)
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
        # Triton <3.6 fallback: cache[key].asm['ptx']
        if not _ptx_found and hasattr(_fn, 'cache'):
            _cache = _fn.cache
            if _cache:
                _key = next(iter(_cache))
                _inner = _cache[_key]
                if hasattr(_inner, 'asm') and 'ptx' in _inner.asm:
                    pathlib.Path(_PTX_OUT).write_text(_inner.asm['ptx'])
    except Exception as _e:
        sys.stderr.write(f"warning: could not extract PTX: {_e}\n")

# Dump output mode (for --correctness)
if _DUMP_OUTPUT > 0:
    {{KERNEL_CALL}}
    torch.cuda.synchronize()
    _vals = z[:_DUMP_OUTPUT].cpu().tolist()
    for _v in _vals:
        print(f"{_v:.6g}")
    sys.exit(0)

# L2 flush buffer
if _L2_FLUSH > 0:
    _flush_buf = torch.zeros(_L2_FLUSH // 4, dtype=torch.float32, device="cuda")

# Timed runs
_times = []
for _i in range(_ITERS):
    if _L2_FLUSH > 0:
        _flush_buf.fill_(0.0)
        torch.cuda.synchronize()
    _t0 = torch.cuda.Event(enable_timing=True)
    _t1 = torch.cuda.Event(enable_timing=True)
    _t0.record()
    {{KERNEL_CALL}}
    _t1.record()
    torch.cuda.synchronize()
    _times.append(_t0.elapsed_time(_t1) * 1000)  # ms -> us

print(f"{min(_times):.3f}")
