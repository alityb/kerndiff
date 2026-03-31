from __future__ import annotations

import atexit
import os
import shutil
import sys

TEMP_PATHS: list[str] = []
SUPPRESS_STDERR = False
DEFER_WARNINGS = False
WATCH_HISTORY: list[dict] = []


def _cleanup_temp_paths() -> None:
    for path in TEMP_PATHS:
        if os.path.isdir(path):
            shutil.rmtree(path, ignore_errors=True)
        elif os.path.exists(path):
            try:
                os.remove(path)
            except OSError:
                pass


atexit.register(_cleanup_temp_paths)


def status_line(label: str, status: str, suffix: str = "") -> str:
    return f"  {label:<34}{status:>8}{('  ' + suffix) if suffix else ''}"


def emit_status(label: str, status: str, suffix: str = "") -> None:
    if SUPPRESS_STDERR:
        return
    print(status_line(label, status, suffix), file=sys.stderr)


def warn(msg: str, warnings: list[str]) -> None:
    warnings.append(msg)
    if SUPPRESS_STDERR or DEFER_WARNINGS:
        return
    print(f"warning: {msg}", file=sys.stderr)


def color_warn(msg: str, use_color: bool) -> str:
    if use_color:
        return f"  \033[33m⚠\033[0m  {msg}"
    return f"  !  {msg}"
