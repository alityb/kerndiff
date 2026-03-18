from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

FIXTURES_DIR = Path(__file__).resolve().parent / "fixtures"


def parse_ptx_instructions(ptx_text: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for raw_line in ptx_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("//") or line.startswith(".") or line.endswith(":"):
            continue
        tokens = line.replace(";", "").split()
        if not tokens:
            continue
        if tokens[0].startswith("@") and "%" in tokens[0] and len(tokens) > 1:
            tokens = tokens[1:]
        if not tokens:
            continue
        mnemonic = tokens[0]
        instruction_class = ".".join(mnemonic.split(".")[:2])
        counts[instruction_class] = counts.get(instruction_class, 0) + 1
    return counts


def extract_ptx(source_path: str, arch: str = "sm_90") -> dict[str, int]:
    with tempfile.TemporaryDirectory(prefix="kerndiff_ptx_") as tmpdir:
        out_path = Path(tmpdir) / "kernel.ptx"
        result = subprocess.run(
            ["nvcc", "-ptx", f"-arch={arch}", "-o", str(out_path), source_path],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError((result.stderr or result.stdout or "nvcc failed").strip())
        return parse_ptx_instructions(out_path.read_text())


def diff_ptx(v1: dict[str, int], v2: dict[str, int]) -> list[dict]:
    rows = []
    for key in sorted(set(v1) | set(v2)):
        v1_count = v1.get(key, 0)
        v2_count = v2.get(key, 0)
        if v1_count == v2_count:
            continue
        delta_pct = ((v2_count - v1_count) / max(v1_count, 1)) * 100.0
        rows.append({"instruction": key, "v1": v1_count, "v2": v2_count, "delta_pct": delta_pct})
    return sorted(rows, key=lambda row: abs(row["delta_pct"]), reverse=True)


def load_fixture(prefix: str) -> dict[str, int]:
    return json.loads((FIXTURES_DIR / f"{prefix}_ptx_counts.json").read_text())
