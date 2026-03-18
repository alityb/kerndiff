from __future__ import annotations

import atexit
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_TEMPLATE = FIXTURES_DIR / "harness_template.cu"
_TEMP_DIRS: list[str] = []

DTYPE_MAP = {
    "float":  ("float",  "#include <cuda_runtime.h>"),
    "half":   ("half",   "#include <cuda_fp16.h>"),
    "int":    ("int",    "#include <cuda_runtime.h>"),
    "int4":   ("int4",   "#include <cuda_runtime.h>"),
}


def _cleanup_tempdirs() -> None:
    for path in _TEMP_DIRS:
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_tempdirs)


def _find_nvcc() -> str:
    nvcc = shutil.which("nvcc")
    if nvcc:
        return nvcc
    # Check common pip-installed CUDA locations
    import site
    for sp in site.getsitepackages() + [site.getusersitepackages()]:
        for pattern in ["nvidia/cu*/bin/nvcc", "nvidia/cuda_nvcc/bin/nvcc"]:
            import glob
            matches = glob.glob(os.path.join(sp, pattern))
            if matches:
                return matches[0]
    raise RuntimeError("nvcc not found on PATH or in pip-installed CUDA packages")


def _find_cuda_lib_dir(nvcc_path: str) -> str | None:
    # nvcc is at .../bin/nvcc, lib is at .../lib/
    cuda_root = Path(nvcc_path).resolve().parent.parent
    lib_dir = cuda_root / "lib"
    if lib_dir.is_dir():
        return str(lib_dir)
    lib64_dir = cuda_root / "lib64"
    if lib64_dir.is_dir():
        return str(lib64_dir)
    return None


def build_harness(source_path: str, kernel_name: str, kernel_call: str, dtype: str = "float") -> str:
    temp_dir = tempfile.mkdtemp(prefix="kerndiff_build_")
    _TEMP_DIRS.append(temp_dir)
    harness_path = Path(temp_dir) / "bench.cu"
    template = _TEMPLATE.read_text()
    source = Path(source_path).read_text()
    elem_type, dtype_include = DTYPE_MAP.get(dtype, DTYPE_MAP["float"])
    harness = (
        template
        .replace("{{KERNEL_SOURCE}}", source)
        .replace("{{KERNEL_NAME}}", kernel_name)
        .replace("{{KERNEL_CALL}}", kernel_call)
        .replace("{{ELEM_TYPE}}", elem_type)
        .replace("{{DTYPE_INCLUDE}}", dtype_include)
    )
    harness_path.write_text(harness)
    return str(harness_path)


def _format_compile_error(source_path: str, stderr_text: str, nvcc_cmd: list[str], harness_path: str) -> str:
    source_name = os.path.basename(source_path)
    lines = stderr_text.strip().splitlines()

    # Extract first real error line
    first_error = None
    for line in lines:
        if "error" in line.lower():
            first_error = line.strip()
            break
    if not first_error:
        first_error = lines[0] if lines else "unknown error"

    hint = ""
    lower = stderr_text.lower()
    if "undefined" in lower and ("half" in lower or "int4" in lower or "__half" in lower):
        hint = "\n  (add --dtype half or include cuda_fp16.h in your source)"
    elif "undefined" in lower:
        hint = "\n  (check --fn matches the kernel name in your source)"

    return (
        f"error: compilation failed for {source_name}\n\n"
        f"  {first_error}{hint}\n\n"
        f"  full output: {harness_path}\n"
        f"  run: {' '.join(nvcc_cmd)}"
    )


def compile_kernel(
    source_path: str,
    kernel_name: str,
    arch: str = "sm_90",
    mock: bool = False,
    kernel_call: str | None = None,
    dtype: str = "float",
) -> str:
    if mock:
        return source_path
    nvcc = _find_nvcc()
    call_expr = kernel_call if kernel_call else f"{kernel_name}<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N)"
    harness_path = build_harness(source_path, kernel_name, call_expr, dtype=dtype)
    output_path = str(Path(harness_path).with_suffix(""))

    cmd = [nvcc, "-O2", f"-arch={arch}"]
    lib_dir = _find_cuda_lib_dir(nvcc)
    if lib_dir:
        cmd.extend([f"-L{lib_dir}"])
    cmd.extend(["-o", output_path, harness_path])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(_format_compile_error(source_path, result.stderr or result.stdout or "nvcc failed", cmd, harness_path))
    return output_path
