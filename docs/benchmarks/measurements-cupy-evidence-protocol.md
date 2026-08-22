# GPU basic Measurements evidence protocol

This file defines the evidence run for VIPP's production pure-CuPy
implementations of **Measure Objects** and **Measure Objects + Intensity**. It is
a run protocol, not a portable performance claim.

The generated canonical artifacts are:

- `measurements-cupy-windows-rtx5090.json` — strict machine-readable evidence.
- `measurements-cupy-windows-rtx5090.md` — a generated readable view of the same
  evidence.

The earlier `measurements-cucim-*` files remain immutable historical evidence
for the retired provider. They are not accepted by the current CuPy schema.

The harness fingerprints only `core/measurements.py`,
`core/gpu/cupy_measurements.py`, and the harness itself. Its embedded semantic
contract separately pins both CuPy implementation IDs, the supported
dtype/parameter region, parity policy, memory model, resident packed ABI, and
mandatory typed host finalizer.

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
  shapes, and each of the five extended measurement-column groups.
- Truthful synchronized stage progress, cancellation for both operations,
  post-cancellation reuse, and zero private-pool residue.
- Production memory-bound coverage and large confocal-like 3D volumes and 2D
  leading-plane stacks.

The provider lazily compiles CuPy RawKernels and materializes small Euler
coefficient arrays. The run initializes both spatial ranks before opening any
per-call private pool so process-lifetime caches are not misclassified as
transactional leaks. Every subsequent private execution pool must still drain
to zero used and zero reserved bytes.

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
& .\.venv-gpu-cu13\Scripts\python.exe scripts\benchmark_gpu_measurements.py --validate-existing docs\benchmarks\measurements-cupy-windows-rtx5090.json
```

The harness imports no CUDA or numeric library for `--help` or
`--validate-existing`, so those commands remain suitable for CPU-only CI. Only
artifacts produced by a successful full run and subsequent strict validation
should be committed. Quick-profile artifacts must not replace the canonical
full-profile files.
