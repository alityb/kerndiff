from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path

from kerndiff import cli_state

KERNEL_RE = re.compile(r"__global__\s+\w+\s+(\w+)\s*\(")
TRITON_KERNEL_RE = re.compile(r"@triton\.jit\s+def\s+(\w+)")


@dataclass
class KernelSelection:
    names: list[str]
    mode: str


def detect_sm_arch(gpu_name: str) -> str | None:
    name = gpu_name.lower()
    if "h100" in name:
        return "sm_90"
    if "h200" in name:
        return "sm_90"
    if "a100" in name:
        return "sm_80"
    if "a10" in name:
        return "sm_86"
    if "l40" in name:
        return "sm_89"
    if "4090" in name:
        return "sm_89"
    if "4080" in name:
        return "sm_89"
    if "3090" in name:
        return "sm_86"
    if "3080" in name:
        return "sm_86"
    if "v100" in name:
        return "sm_70"
    return None


def _scan_kernels(path: str) -> list[str]:
    text = Path(path).read_text()
    if path.endswith(".py"):
        return TRITON_KERNEL_RE.findall(text)
    return KERNEL_RE.findall(text)


def resolve_kernel_name(file_a: str, file_b: str, fn_name: str | None) -> str:
    if fn_name:
        return fn_name
    kernels_a = _scan_kernels(file_a)
    kernels_b = _scan_kernels(file_b)
    if len(kernels_a) == 1 and len(kernels_b) == 1 and kernels_a[0] == kernels_b[0]:
        return kernels_a[0]

    common = sorted(set(kernels_a) & set(kernels_b))

    if common and sys.stdin.isatty():
        print("multiple kernels found — pick one (or use --fn):", file=sys.stderr)
        for i, name in enumerate(common, 1):
            print(f"  [{i}] {name}", file=sys.stderr)
        try:
            choice = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit("\naborted")
        if choice.isdigit():
            idx = int(choice) - 1
            if 0 <= idx < len(common):
                return common[idx]
        if choice in common:
            return choice
        raise SystemExit(f"error: invalid selection '{choice}'")

    msg = [
        "error: could not auto-detect kernel — please specify --fn",
        f"  {file_a}: {', '.join(kernels_a) or '(none)'}",
        f"  {file_b}: {', '.join(kernels_b) or '(none)'}",
    ]
    raise SystemExit("\n".join(msg))


def resolve_all_kernels(file_a: str, file_b: str) -> list[str]:
    kernels_a = set(_scan_kernels(file_a))
    kernels_b = set(_scan_kernels(file_b))
    if not kernels_a:
        raise SystemExit(f"error: no kernels found in {os.path.basename(file_a)}")
    if not kernels_b:
        raise SystemExit(f"error: no kernels found in {os.path.basename(file_b)}")
    common = sorted(kernels_a & kernels_b)
    if not cli_state.SUPPRESS_STDERR:
        for name in sorted(kernels_a - kernels_b):
            print(f"  skipping {name} (not in {os.path.basename(file_b)})", file=sys.stderr)
        for name in sorted(kernels_b - kernels_a):
            print(f"  skipping {name} (not in {os.path.basename(file_a)})", file=sys.stderr)
    if not common:
        raise SystemExit(
            f"error: no kernels in common between {os.path.basename(file_a)} and {os.path.basename(file_b)}"
        )
    return common


def resolve_kernel_selection(file_a: str, file_b: str, fn_name: str | None, all_kernels: bool) -> KernelSelection:
    if all_kernels:
        return KernelSelection(names=resolve_all_kernels(file_a, file_b), mode="all")
    if fn_name:
        return KernelSelection(names=[fn_name], mode="explicit")
    return KernelSelection(names=[resolve_kernel_name(file_a, file_b, None)], mode="auto")


def resolve_git_baseline(filepath: str, at_ref: str = "HEAD") -> tuple[str, str]:
    abs_path = Path(filepath).resolve()

    result = subprocess.run(
        ["git", "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
        cwd=abs_path.parent,
    )
    if result.returncode != 0:
        raise SystemExit("error: single-file mode requires a git repo (run inside a git repo, or pass two files)")

    repo_root = Path(result.stdout.strip())
    rel_path = abs_path.relative_to(repo_root)

    tracked = subprocess.run(
        ["git", "ls-files", "--error-unmatch", str(rel_path)],
        capture_output=True,
        cwd=repo_root,
    )
    if tracked.returncode != 0:
        raise SystemExit(f"error: {rel_path} is not tracked by git (git add it first)")

    if at_ref == "HEAD":
        display_ref = "HEAD"
    else:
        short = subprocess.run(
            ["git", "rev-parse", "--short", at_ref],
            capture_output=True,
            text=True,
            cwd=repo_root,
        )
        display_ref = short.stdout.strip() if short.returncode == 0 else at_ref

    head_content = subprocess.run(
        ["git", "show", f"{at_ref}:{rel_path}"],
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    if head_content.returncode != 0:
        raise SystemExit(
            f"error: file not found in {at_ref}: {rel_path}\n"
            f"  (commit it first, or pass two files explicitly)"
        )

    suffix = abs_path.suffix or ".cu"
    tmp = tempfile.NamedTemporaryFile(
        suffix=suffix,
        prefix="kerndiff_head_",
        delete=False,
        mode="w",
    )
    tmp.write(head_content.stdout)
    tmp.close()
    cli_state.TEMP_PATHS.append(tmp.name)

    display_label = f"{display_ref}:{rel_path}"
    return tmp.name, display_label
