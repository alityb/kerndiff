# kerndiff

`kerndiff` compares two CUDA or Triton kernel implementations on the same GPU and reports:
- end-to-end latency (adaptive multi-run timing)
- Nsight Compute counters
- derived metrics and deltas
- static PTX instruction diffs
- a compact roofline summary

It is designed for practical iteration: change a kernel, run one command, and see if it is faster and why.

## Install

```bash
pip install -e .
```

## Quickstart

```bash
kerndiff examples/vec_add_v1.cu examples/vec_add_v2.cu --fn vec_add
```

Mock mode (no GPU required):

```bash
kerndiff --mock examples/vec_add_v1.cu examples/vec_add_v2.cu --fn vec_add --no-color
```

## Common workflows

Compare two files:

```bash
kerndiff v1.cu v2.cu --fn my_kernel
```

Compare `HEAD` vs working copy for one tracked file:

```bash
kerndiff kernel.cu --fn my_kernel
```

Run all common kernels in two files:

```bash
kerndiff a.cu b.cu --all
```

Write JSON output:

```bash
kerndiff a.cu b.cu --fn k --format json > result.json
```

Write JSON to file while keeping stderr progress:

```bash
kerndiff a.cu b.cu --fn k --export-json result.json
```

## Key options

- `--fn NAME`: kernel name
- `--all`: profile all common kernels
- `--call "kernel<<<...>>>(...)"`: override launch expression
- `--dtype {float,half,int,int4}`: harness buffer dtype
- `--elems N`: harness buffer size
- `--min-runs N`, `--max-runs N`: adaptive timing bounds
- `--noise-threshold PCT`: CV stop threshold
- `--warmup N`: warmup iterations
- `--format {term,json}`: output format
- `--output FILE`: write output to file
- `--export-json FILE`: write JSON file and keep stderr progress
- `--no-color`: disable ANSI colors
- `--gpu N`: GPU index
- `--arch sm_XX`: target SM architecture
- `--mock`: fixture-backed run without GPU

## Output interpretation

Primary signal:
- `latency` row and verdict line (`v2 is ... faster/slower`)

Frequent supporting signals:
- `global_load_eff`: memory coalescing quality
- `dram_bw`: achieved memory bandwidth
- `l2_hit_rate`: cache locality/reuse
- `register_count` and `sm_occupancy`: register pressure tradeoff
- `stall_*`: dominant stall sources

Roofline row:
- reports `memory` vs `compute` bound and estimated headroom

PTX diff:
- static instruction counts only (not dynamic execution counts)

## Repository layout

This repository uses a `src` layout:

- `src/kerndiff/`: package code
- `src/kerndiff/runtimes/`: runtime-specific compilation and execution adapters (`cuda`, `triton`)
- `examples/`: benchmark kernels
- `tests/`: test suite

## Development

Run tests:

```bash
python -m pytest tests/ -q
```

## Troubleshooting

If Nsight Compute metrics are missing:
- ensure `ncu` is installed
- run with elevated permissions if needed on your system

If `nvcc` is not found:
- ensure CUDA toolkit is installed or available in your Python CUDA environment
