from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:
    try:
        import tomli as tomllib  # type: ignore[no-redef]
    except ModuleNotFoundError:
        tomllib = None  # type: ignore[assignment]


def find_config() -> Path | None:
    """Walk up from cwd looking for kerndiff.toml, stopping at repo root."""
    current = Path.cwd()
    for parent in [current, *current.parents]:
        candidate = parent / "kerndiff.toml"
        if candidate.exists():
            return candidate
        if (parent / ".git").exists():
            break
    return None


def load_config(path: Path) -> dict:
    """Parse kerndiff.toml, return raw dict."""
    if tomllib is None:
        print("note: install tomli for kerndiff.toml support", file=sys.stderr)
        return {}
    with open(path, "rb") as f:
        return tomllib.load(f)


_VALID_CONFIG_KEYS = {
    "fn", "dtype", "elems", "noise_threshold", "max_runs", "min_runs",
    "warmup", "arch", "call", "pipeline",
}

_CONFIG_TYPES = {
    "fn": str,
    "dtype": str,
    "elems": int,
    "noise_threshold": (int, float),
    "max_runs": int,
    "min_runs": int,
    "warmup": int,
    "arch": str,
    "call": str,
    "pipeline": int,
}


def apply_config(args: argparse.Namespace, config: dict, kernel_name: str | None = None) -> argparse.Namespace:
    """Fill in unset args from config. CLI flags always win.

    Priority: CLI flag > [kernels.<fn>] > [defaults] > built-in defaults.
    """
    defaults = config.get("defaults", {})

    # Warn about unknown keys in [defaults]
    for key in defaults:
        if key not in _VALID_CONFIG_KEYS:
            print(f"warning: unknown config key '{key}' ignored", file=sys.stderr)

    # Merge kernel-specific overrides on top of defaults
    merged = dict(defaults)
    if kernel_name:
        kernel_section = config.get("kernels", {}).get(kernel_name, {})
        for key in kernel_section:
            if key not in _VALID_CONFIG_KEYS:
                print(f"warning: unknown config key '{key}' in [kernels.{kernel_name}] ignored", file=sys.stderr)
        merged.update(kernel_section)

    # Warn about unknown top-level keys
    for key in config:
        if key not in ("defaults", "kernels"):
            print(f"warning: unknown config section '{key}' ignored", file=sys.stderr)

    # Map config keys to argparse attribute names
    CONFIG_MAP = {
        "fn": "fn_name",
        "dtype": "dtype",
        "elems": "elems",
        "noise_threshold": "noise_threshold",
        "max_runs": "max_runs",
        "min_runs": "min_runs",
        "warmup": "warmup",
        "arch": "arch",
        "call": "call_expr",
        "pipeline": "pipeline",
    }

    # Built-in defaults (to detect if CLI changed a value)
    BUILTIN_DEFAULTS = {
        "fn_name": None,
        "dtype": "float",
        "elems": 1 << 22,
        "noise_threshold": 1.0,
        "max_runs": 50,
        "min_runs": 10,
        "warmup": 32,
        "arch": "sm_90",
        "call_expr": None,
        "pipeline": 1,
    }

    for config_key, arg_attr in CONFIG_MAP.items():
        if config_key not in merged:
            continue
        value = merged[config_key]
        # Type-check the config value
        expected = _CONFIG_TYPES.get(config_key)
        if expected and not isinstance(value, expected):
            print(f"warning: config key '{config_key}' has wrong type "
                  f"(expected {expected.__name__ if isinstance(expected, type) else 'number'}, "
                  f"got {type(value).__name__}) — ignored", file=sys.stderr)
            continue
        current = getattr(args, arg_attr, None)
        builtin = BUILTIN_DEFAULTS.get(arg_attr)
        # Only apply config if CLI value matches the built-in default (i.e., user didn't set it)
        if current == builtin:
            setattr(args, arg_attr, value)

    return args
