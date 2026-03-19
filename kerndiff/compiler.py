from __future__ import annotations

import atexit
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_TEMPLATE = FIXTURES_DIR / "harness_template.cu"
_TEMP_DIRS: list[str] = []

# Regex to extract __global__ kernel signature: captures fn_name and params
_SIG_RE = re.compile(
    r"__global__\s+\w+\s+(\w+)\s*\(([^)]*)\)",
    re.DOTALL,
)

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


def parse_kernel_signature(source: str, fn_name: str) -> list[tuple[str, str]]:
    """Extract parameter (type, name) pairs from a __global__ kernel declaration."""
    for match in _SIG_RE.finditer(source):
        if match.group(1) == fn_name:
            raw_params = match.group(2)
            params = []
            for part in raw_params.split(","):
                part = part.strip()
                if not part:
                    continue
                # Remove __restrict__ and const qualifiers
                cleaned = part.replace("__restrict__", "").replace("const ", "").strip()
                # Split into type and name: last token is name, rest is type
                tokens = cleaned.split()
                if len(tokens) >= 2:
                    name = tokens[-1].lstrip("*")
                    ptype = " ".join(tokens[:-1])
                    # Check if pointer star is on the name side (e.g., "float *a")
                    if "*" in cleaned and "*" not in ptype:
                        ptype += "*"
                    # Normalize: "float *" -> "float*"
                    ptype = ptype.replace(" *", "*").replace("* ", "*")
                    params.append((ptype, name))
                elif len(tokens) == 1:
                    params.append((tokens[0], ""))
            return params
    return []


def generate_call(fn_name: str, params: list[tuple[str, str]]) -> tuple[str, list[str]]:
    """Generate a best-effort kernel call expression from parsed parameters.

    Returns (call_expr, warnings).
    """
    if not params:
        return f"{fn_name}<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N)", []

    warnings: list[str] = []
    args = []
    ptr_index = 0
    ptr_names = ["d_a", "d_b", "d_c"]

    for ptype, pname in params:
        ptype_lower = ptype.lower().rstrip("*").strip()
        is_ptr = "*" in ptype

        if is_ptr:
            if ptr_index < len(ptr_names):
                args.append(ptr_names[ptr_index])
                ptr_index += 1
            else:
                args.append("nullptr")
                warnings.append(f"extra pointer param '{pname}' mapped to nullptr — use --call to override")
        elif ptype_lower in ("int", "unsigned", "unsigned int", "size_t", "long", "long long"):
            name_lower = pname.lower()
            if name_lower in ("n", "num", "size", "count", "len", "length", "num_elements", "nelems"):
                args.append("N")
            elif "stride" in name_lower:
                args.append("1")
            elif "batch" in name_lower:
                args.append("1")
            else:
                args.append("N")
                if pname:
                    warnings.append(f"int param '{pname}' mapped to N — use --call to override")
        elif ptype_lower == "float" and not is_ptr:
            args.append("1.0f")
            if pname:
                warnings.append(f"float param '{pname}' mapped to 1.0f — use --call to override")
        else:
            args.append("0")
            warnings.append(f"unknown param type '{ptype} {pname}' mapped to 0 — use --call to override")

    call = f"{fn_name}<<<GRID_SIZE, BLOCK_SIZE>>>({', '.join(args)})"
    return call, warnings


def build_harness(source_path: str, kernel_name: str, kernel_call: str, dtype: str = "float", buf_elems: int = 1 << 22) -> str:
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
        .replace("{{BUF_ELEMS}}", str(buf_elems))
    )
    harness_path.write_text(harness)
    return str(harness_path)


def _format_compile_error(source_path: str, stderr_text: str, nvcc_cmd: list[str], harness_path: str, auto_call: str | None = None) -> str:
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

    auto_hint = ""
    if auto_call:
        auto_hint = f"\n  hint: auto-generated call was: {auto_call} — use --call to override"

    return (
        f"error: compilation failed for {source_name}\n\n"
        f"  {first_error}{hint}{auto_hint}\n\n"
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
    buf_elems: int = 1 << 22,
) -> str:
    if mock:
        return source_path
    nvcc = _find_nvcc()
    auto_call = None
    if kernel_call:
        call_expr = kernel_call
    else:
        call_expr = f"{kernel_name}<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N)"
        # Try auto-generating from signature
        try:
            source = Path(source_path).read_text()
            params = parse_kernel_signature(source, kernel_name)
            if params:
                generated, _ = generate_call(kernel_name, params)
                call_expr = generated
                auto_call = generated
        except Exception:
            pass
    harness_path = build_harness(source_path, kernel_name, call_expr, dtype=dtype, buf_elems=buf_elems)
    output_path = str(Path(harness_path).with_suffix(""))

    cmd = [nvcc, "-O2", f"-arch={arch}"]
    lib_dir = _find_cuda_lib_dir(nvcc)
    if lib_dir:
        cmd.extend([f"-L{lib_dir}"])
    cmd.extend(["-o", output_path, harness_path])

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise SystemExit(_format_compile_error(
            source_path, result.stderr or result.stdout or "nvcc failed", cmd, harness_path,
            auto_call=auto_call,
        ))
    return output_path


def verify_correctness(
    binary_a: str,
    binary_b: str,
    env: dict | None = None,
    dump_count: int = 16,
) -> tuple[float, list[float], list[float]]:
    """Run both binaries with --dump-output and compare first dump_count elements of d_c.

    Returns (max_diff, v1_values, v2_values).
    """
    run_env = os.environ.copy()
    if env:
        run_env.update(env)

    r1 = subprocess.run(
        [binary_a, "--dump-output", str(dump_count)],
        capture_output=True, text=True, env=run_env,
    )
    r2 = subprocess.run(
        [binary_b, "--dump-output", str(dump_count)],
        capture_output=True, text=True, env=run_env,
    )

    def parse_values(output: str) -> list[float]:
        vals = []
        for line in output.strip().splitlines():
            try:
                vals.append(float(line.strip()))
            except ValueError:
                continue
        return vals

    v1_vals = parse_values(r1.stdout or "")
    v2_vals = parse_values(r2.stdout or "")

    if not v1_vals or not v2_vals:
        return float("inf"), v1_vals, v2_vals

    max_diff = max(_safe_diff(a, b) for a, b in zip(v1_vals, v2_vals))
    return max_diff, v1_vals, v2_vals


def _safe_diff(a: float, b: float) -> float:
    """Return |a-b|, treating NaN and inf as special cases."""
    if math.isnan(a) and math.isnan(b):
        return 0.0  # both NaN — consistent (same "wrong" answer)
    if math.isnan(a) or math.isnan(b):
        return float("inf")  # one NaN, one real — always fails
    if math.isinf(a) and math.isinf(b) and (a > 0) == (b > 0):
        return 0.0  # both same-sign inf — consistent
    return abs(a - b)
