# GPU basic Measurements evidence protocol

This file defines the evidence run for the first public cuCIM implementation of
VIPP's **Measure Objects** and **Measure Objects + Intensity** nodes. It is a run
protocol, not a performance result: no timing numbers are claimed here.

The generated canonical artifacts will be:

- `measurements-cucim-windows-rtx5090.json` — strict machine-readable evidence.
- `measurements-cucim-windows-rtx5090.md` — a generated readable view of the
  same evidence.

The harness keeps source provenance operation-owned. It fingerprints only
`core/measurements.py`, `core/gpu/cucim_measurements.py`, and the harness itself;
shared registry refactors therefore do not invalidate unchanged scientific
evidence. The embedded semantic contract separately pins both implementation
IDs, cuCIM 26.06.00, the supported dtype/parameter region, parity policy,
memory model, resident packed ABI, and mandatory typed host finalizer.

## Required full-profile coverage

- Basic morphology and morphology-plus-intensity tables in 2D and 3D.
- Leading spatial blocks in C order and explicitly reordered spatial axes.
- Arbitrary sparse positive IDs, one ID spanning disconnected components, and
  empty label images that retain a zero-row schema.
- Native nonnegative `int32` labels and bool, `uint8`, `uint16`, and finite
  `float32` intensity inputs.
- Calibrated axes, physical columns, exact table schema/order/units/Python
  scalar types, operation-owned numeric tolerances, and repeat determinism.
- Explicit rejection of unsupported label/intensity dtypes, non-native label
  byte order, negative labels, non-finite float intensity, mismatched input
  shapes, and extended measurement columns.
- Truthful synchronized stage progress, cancellation for both operations,
  post-cancellation reuse, and zero private-pool residue.
- Production memory-bound coverage and large confocal-like 3D volumes and 2D
  leading-plane stacks.

GPU timing is split into three views:

1. Resident packed compute, which is diagnostic only.
2. Resident compute plus the mandatory D2H and typed-table finalizer.
3. Full public execution including authored H2D inputs, compute, D2H, and the
   typed-table finalizer.

CPU/GPU screening uses the third view because a measurement node cannot expose
its private packed matrix as the public result.

## Run on the RTX development environment

From the repository root, with `src` on `PYTHONPATH`:

```powershell
$env:PYTHONPATH = (Resolve-Path .\src)
& .\.venv-gpu-cu13\Scripts\python.exe scripts\benchmark_gpu_measurements.py --profile full
& .\.venv-gpu-cu13\Scripts\python.exe scripts\benchmark_gpu_measurements.py --validate-existing docs\benchmarks\measurements-cucim-windows-rtx5090.json
```

If the GPU environment lives in the neighboring development worktree, use its
interpreter while retaining this worktree's `PYTHONPATH`. The harness imports no
CUDA or numeric library for `--help` or `--validate-existing`, so those commands
remain suitable for CPU-only CI.

Only artifacts produced by a successful full run and a subsequent strict
validation should be committed. Quick-profile artifacts are development smoke
results and must not replace the canonical full-profile files.
