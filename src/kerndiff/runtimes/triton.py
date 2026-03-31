from __future__ import annotations

import atexit
import os
import re
import select
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

from kerndiff.ptx import parse_ptx_instructions

FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures"
_TRITON_TEMPLATE = FIXTURES_DIR / "harness_template_triton.py"
_TRITON_PERSISTENT_TEMPLATE = FIXTURES_DIR / "harness_template_triton_persistent.py"
_TEMP_DIRS: list[str] = []
_PTX_PATHS: dict[str, str] = {}

_TRITON_SIG_RE = re.compile(r"@triton\.jit\s+def\s+(\w+)")


def _cleanup():
    for d in _TEMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)


atexit.register(_cleanup)


TORCH_DTYPE_MAP = {
    "float": "torch.float32",
    "half": "torch.float16",
    "int": "torch.int32",
    "int4": "torch.int32",
}


def _check_triton():
    try:
        import triton  # noqa: F401
    except ImportError:
        raise SystemExit(
            "error: Triton runtime requires triton — install with: pip install triton"
        )


def parse_triton_kernels(source: str) -> list[str]:
    """Scan a .py file for @triton.jit decorated function names."""
    return _TRITON_SIG_RE.findall(source)


class TritonBackend:
    """Backend for .py Triton files — generates a persistent Python harness."""

    def __init__(self):
        # Harness generation and command construction do not require Triton to
        # be importable in the current Python process. Runtime execution will
        # fail later with the harness stderr if Triton is actually unavailable.
        pass

    # ------------------------------------------------------------------
    # Harness generation
    # ------------------------------------------------------------------

    def default_call_expr(self, kernel_name: str, buf_elems: int) -> str:
        return f"{kernel_name}[({buf_elems} + 127) // 128,](x, y, z, {buf_elems}, BLOCK_SIZE=128)"

    def _build_persistent_harness(
        self,
        source_path: str,
        kernel_name: str,
        dtype: str,
        buf_elems: int,
        call_expr: str | None,
        l2_flush_bytes: int,
        warmup: int,
    ) -> str:
        torch_dtype = TORCH_DTYPE_MAP.get(dtype, "torch.float32")

        if call_expr is None:
            call_expr = self.default_call_expr(kernel_name, buf_elems)

        source_text = Path(source_path).read_text()
        temp_dir = tempfile.mkdtemp(prefix="kerndiff_triton_")
        _TEMP_DIRS.append(temp_dir)

        # Reuse PTX path from first compile() call so extract_ptx() finds it
        ptx_output_path = _PTX_PATHS.get(source_path, str(Path(temp_dir) / "kernel.ptx"))
        _PTX_PATHS[source_path] = ptx_output_path

        template = _TRITON_PERSISTENT_TEMPLATE.read_text()
        harness = (
            template
            .replace("{{KERNEL_SOURCE}}", source_text)
            .replace("{{TORCH_DTYPE}}", torch_dtype)
            .replace("{{BUF_ELEMS}}", str(buf_elems))
            .replace("{{L2_FLUSH_BYTES}}", str(l2_flush_bytes))
            .replace("{{KERNEL_CALL}}", call_expr)
            .replace("{{KERNEL_NAME}}", kernel_name)
            .replace("{{PTX_OUTPUT_PATH}}", ptx_output_path)
            .replace("{{WARMUP}}", str(warmup))
        )
        harness_path = Path(temp_dir) / "harness.py"
        harness_path.write_text(harness)
        return str(harness_path)

    def compile(
        self,
        source_path: str,
        kernel_name: str,
        arch: str,
        dtype: str,
        buf_elems: int,
        call_expr: str | None,
    ) -> str:
        """Generate persistent timing harness (no L2 flush, default warmup)."""
        self._last_compile_args = {
            "source_path": source_path,
            "kernel_name": kernel_name,
            "arch": arch,
            "dtype": dtype,
            "buf_elems": buf_elems,
            "call_expr": call_expr,
        }
        return self._build_persistent_harness(
            source_path, kernel_name, dtype, buf_elems, call_expr,
            l2_flush_bytes=0, warmup=32,
        )

    def compile_timed(
        self,
        source_path: str,
        kernel_name: str,
        arch: str,
        dtype: str,
        buf_elems: int,
        call_expr: str | None,
        iters: int = 1,
        l2_flush_bytes: int = 0,
        warmup: int = 32,
    ) -> str:
        """Generate persistent timing harness with L2 flush baked in.

        iters is accepted for API compatibility with CUDABackend but ignored —
        the number of timing runs is controlled by the profiler via the pipe.
        """
        return self._build_persistent_harness(
            source_path, kernel_name, dtype, buf_elems, call_expr,
            l2_flush_bytes=l2_flush_bytes, warmup=warmup,
        )

    def compile_ncu(
        self,
        source_path: str,
        kernel_name: str,
        arch: str,
        dtype: str,
        buf_elems: int,
        call_expr: str | None,
    ) -> str:
        """Generate a single-run harness for NCU profiling (not persistent)."""
        torch_dtype = TORCH_DTYPE_MAP.get(dtype, "torch.float32")

        if call_expr is None:
            call_expr = self.default_call_expr(kernel_name, buf_elems)

        source_text = Path(source_path).read_text()
        temp_dir = tempfile.mkdtemp(prefix="kerndiff_triton_ncu_")
        _TEMP_DIRS.append(temp_dir)

        ptx_output_path = _PTX_PATHS.get(source_path, "")

        template = _TRITON_TEMPLATE.read_text()
        harness = (
            template
            .replace("{{KERNEL_SOURCE}}", source_text)
            .replace("{{TORCH_DTYPE}}", torch_dtype)
            .replace("{{BUF_ELEMS}}", str(buf_elems))
            .replace("{{ITERS}}", "1")
            .replace("{{KERNEL_CALL}}", call_expr)
            .replace("{{KERNEL_NAME}}", kernel_name)
            .replace("{{PTX_OUTPUT_PATH}}", ptx_output_path)
            .replace("{{L2_FLUSH_BYTES}}", "0")
            .replace("{{DUMP_OUTPUT}}", "0")
        )
        harness_path = Path(temp_dir) / "harness_ncu.py"
        harness_path.write_text(harness)
        return str(harness_path)

    # ------------------------------------------------------------------
    # Persistent process protocol
    # ------------------------------------------------------------------

    def is_persistent(self) -> bool:
        return True

    def spawn_persistent(self, artifact: str, env: dict | None = None) -> subprocess.Popen:
        """Spawn the persistent harness and wait for the 'ready' signal.

        The harness performs warmup and PTX extraction before printing 'ready'.
        Raises SystemExit if the harness fails to start.
        """
        proc = subprocess.Popen(
            [sys.executable, artifact],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            env={**os.environ, **(env or {})},
        )
        ready_line = proc.stdout.readline().strip()
        if ready_line != "ready":
            stderr_text = ""
            try:
                stderr_text = proc.stderr.read(500)
            except Exception:
                pass
            proc.kill()
            raise SystemExit(
                f"error: Triton harness failed to start\n"
                f"  output: {ready_line!r}\n"
                f"  stderr: {stderr_text}"
            )
        return proc

    def send_time(self, proc: subprocess.Popen, timeout_sec: int = 30) -> float:
        """Send one 'time' command and return measured latency in microseconds."""
        proc.stdin.write("time\n")
        proc.stdin.flush()
        ready, _, _ = select.select([proc.stdout], [], [], timeout_sec)
        if not ready:
            proc.kill()
            raise SystemExit(
                f"error: Triton harness timed out after {timeout_sec}s\n"
                f"  (kernel may be hanging or deadlocked)"
            )
        response = proc.stdout.readline().strip()
        if not response or response == "error":
            stderr_text = ""
            try:
                stderr_text = proc.stderr.read(300)
            except Exception:
                pass
            proc.kill()
            raise SystemExit(
                f"error: Triton harness crashed during timing\n"
                f"  stderr: {stderr_text}"
            )
        return float(response)

    def dump_output(self, proc: subprocess.Popen, timeout_sec: int = 30) -> list[float]:
        """Send 'dump' command; returns first 16 output elements as floats."""
        proc.stdin.write("dump\n")
        proc.stdin.flush()
        ready, _, _ = select.select([proc.stdout], [], [], timeout_sec)
        if not ready:
            proc.kill()
            raise SystemExit(
                f"error: Triton harness timed out during correctness dump ({timeout_sec}s)"
            )
        response = proc.stdout.readline().strip()
        if not response or response == "error":
            stderr_text = ""
            try:
                stderr_text = proc.stderr.read(300)
            except Exception:
                pass
            proc.kill()
            raise SystemExit(
                f"error: Triton harness failed to dump output\n"
                f"  stderr: {stderr_text}"
            )
        return [float(v) for v in response.split()]

    def shutdown(self, proc: subprocess.Popen) -> None:
        try:
            proc.stdin.write("quit\n")
            proc.stdin.flush()
            proc.wait(timeout=5)
        except Exception:
            proc.kill()

    # ------------------------------------------------------------------
    # PTX extraction
    # ------------------------------------------------------------------

    def extract_ptx(self, source_path: str, arch: str) -> dict[str, int]:
        ptx_path = _PTX_PATHS.get(source_path)
        if not ptx_path or not Path(ptx_path).exists():
            return {}
        try:
            return parse_ptx_instructions(Path(ptx_path).read_text())
        except Exception:
            return {}

    # ------------------------------------------------------------------
    # Legacy interface (used for one-shot runs and NCU)
    # ------------------------------------------------------------------

    def run_cmd(self, artifact: str, kernel_name: str, iters: int, l2_flush: int) -> list[str]:
        return [sys.executable, artifact]

    def ncu_cmd(
        self,
        ncu_path: str,
        artifact: str,
        kernel_name: str,
        metrics: str,
        launch_count: int,
    ) -> list[str]:
        python = sys.executable
        return [
            ncu_path,
            "--target-processes", "all",
            "--kernel-name", f"regex:{kernel_name}.*",
            "--csv",
            "--metrics", metrics,
            "--launch-count", str(launch_count),
            "--cache-control", "all",
            "--clock-control", "none",
            python, artifact,
        ]
