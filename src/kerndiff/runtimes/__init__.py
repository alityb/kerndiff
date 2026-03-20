from __future__ import annotations

from typing import Protocol


class Backend(Protocol):
    def compile(
        self,
        source_path: str,
        kernel_name: str,
        arch: str,
        dtype: str,
        buf_elems: int,
        call_expr: str | None,
    ) -> str:
        """Compile source to a runnable artifact. Returns path to binary or script."""
        ...

    def extract_ptx(self, source_path: str, arch: str) -> dict[str, int]:
        """Return instruction-class -> static count dict."""
        ...

    def run_cmd(self, artifact: str, kernel_name: str, iters: int, l2_flush: int) -> list[str]:
        """Return argv list to time the kernel. stdout must be latency in microseconds."""
        ...

    def ncu_cmd(
        self, ncu_path: str, artifact: str, kernel_name: str,
        metrics: str, launch_count: int,
    ) -> list[str]:
        """Return argv list to run NCU against the artifact."""
        ...


def dispatch(filepath: str) -> Backend:
    """Return the appropriate backend for a given file extension."""
    if filepath.endswith(".py"):
        from kerndiff.backends.triton import TritonBackend
        return TritonBackend()
    elif filepath.endswith(".cu"):
        from kerndiff.backends.cuda import CUDABackend
        return CUDABackend()
    else:
        ext = filepath.rsplit(".", 1)[-1] if "." in filepath else "(none)"
        raise SystemExit(f"error: unsupported file type .{ext} — expected .cu or .py")
