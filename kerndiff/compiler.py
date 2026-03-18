from __future__ import annotations

import atexit
import shutil
import subprocess
import tempfile
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"
_TEMPLATE = FIXTURES_DIR / "harness_template.cu"
_TEMP_DIRS: list[str] = []


def _cleanup_tempdirs() -> None:
    for path in _TEMP_DIRS:
        shutil.rmtree(path, ignore_errors=True)


atexit.register(_cleanup_tempdirs)


def build_harness(source_path: str, kernel_name: str, kernel_call: str) -> str:
    temp_dir = tempfile.mkdtemp(prefix="kerndiff_build_")
    _TEMP_DIRS.append(temp_dir)
    harness_path = Path(temp_dir) / "bench.cu"
    template = _TEMPLATE.read_text()
    source = Path(source_path).read_text()
    harness = (
        template
        .replace("{{KERNEL_SOURCE}}", source)
        .replace("{{KERNEL_NAME}}", kernel_name)
        .replace("{{KERNEL_CALL}}", kernel_call)
    )
    harness_path.write_text(harness)
    return str(harness_path)


def compile_kernel(
    source_path: str,
    kernel_name: str,
    arch: str = "sm_90",
    mock: bool = False,
    kernel_call: str | None = None,
) -> str:
    if mock:
        return source_path
    nvcc = shutil.which("nvcc")
    if nvcc is None:
        raise RuntimeError("nvcc not found on PATH")
    call_expr = kernel_call if kernel_call else f"{kernel_name}<<<GRID_SIZE, BLOCK_SIZE>>>(d_a, d_b, d_c, N)"
    harness_path = build_harness(source_path, kernel_name, call_expr)
    output_path = str(Path(harness_path).with_suffix(""))
    result = subprocess.run(
        [nvcc, "-O2", f"-arch={arch}", "-o", output_path, harness_path],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError((result.stderr or result.stdout or "nvcc failed").strip())
    return output_path
