from __future__ import annotations

import sys

from kerndiff.compiler import (
    compile_kernel,
    _find_nvcc,
    _find_cuda_lib_dir,
)
from kerndiff.ptx import extract_ptx as _extract_ptx


class CUDABackend:
    """Backend for .cu files — compiles with nvcc, runs as native binary."""

    def compile(
        self,
        source_path: str,
        kernel_name: str,
        arch: str,
        dtype: str,
        buf_elems: int,
        call_expr: str | None,
    ) -> str:
        return compile_kernel(
            source_path, kernel_name,
            arch=arch, mock=False,
            kernel_call=call_expr, dtype=dtype, buf_elems=buf_elems,
        )

    def extract_ptx(self, source_path: str, arch: str) -> dict[str, int]:
        return _extract_ptx(source_path, arch=arch)

    def run_cmd(self, artifact: str, kernel_name: str, iters: int, l2_flush: int) -> list[str]:
        cmd = [artifact, "--kernel", kernel_name, "--iters", str(iters)]
        if l2_flush > 0:
            cmd.extend(["--l2-flush", str(l2_flush)])
        return cmd

    def ncu_cmd(
        self, ncu_path: str, artifact: str, kernel_name: str,
        metrics: str, launch_count: int,
    ) -> list[str]:
        return [
            ncu_path,
            "--csv",
            "--metrics", metrics,
            "--launch-count", str(launch_count),
            "--cache-control", "all",
            "--clock-control", "base",
            artifact,
            "--kernel", kernel_name,
            "--iters", "1",
        ]
