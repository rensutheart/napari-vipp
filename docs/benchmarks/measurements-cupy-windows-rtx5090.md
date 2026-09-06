# GPU basic Measurements evidence

Generated: `2026-09-06T17:15:58.650023+00:00`

This is machine-local evidence from **NVIDIA GeForce RTX 5090** on Windows with Python 3.12.9. It is not a portable performance claim or a durable optimizer record.

The public GPU timing includes the mandatory packed-result transfer and typed `TableData` finalizer. Resident packed-compute timing is shown only to explain where time is spent.

## Admission and lifecycle

- Admission cases: **11** (all passed)
- Scientifically ineligible cases rejected: **11**
- Progress/cancellation lifecycle cases: **2**
- Parity policy: `basic-measurement-table-v1`
- Private CUDA pools after every case: **0 used / 0 reserved bytes**

Admission covers 2D/3D, leading blocks, reordered/calibrated axes, sparse and repeated IDs, zero-row tables, all supported intensity dtypes, exact schema/order/units/scalar types, deterministic repeats, cancellation, and post-cancellation reuse.

## Performance

| Workload | Rows | CPU (s) | GPU resident packed (s) | GPU resident + table (s) | GPU full public (s) | Full speedup | Screen | Peak VRAM / bound |
|---|---:|---:|---:|---:|---:|---:|---|---:|
| 256² morphology | 4 | 0.000985 | 0.001528 | 0.001848 | 0.001899 | 0.52× | CPU | 1.8 / 81.5 MiB |
| 256² morphology + uint16 intensity | 4 | 0.001189 | 0.006487 | 0.006258 | 0.006009 | 0.20× | CPU | 2.0 / 92.8 MiB |
| 512² morphology | 16 | 0.002658 | 0.001453 | 0.001815 | 0.002095 | 1.27× | GPU-CuPy | 7.3 / 134.0 MiB |
| 512² morphology + uint16 intensity | 16 | 0.004323 | 0.011649 | 0.011409 | 0.011753 | 0.37× | CPU | 7.8 / 179.0 MiB |
| 1024² morphology | 64 | 0.009662 | 0.001574 | 0.002330 | 0.003009 | 3.21× | GPU-CuPy | 29.2 / 350.0 MiB |
| 1024² morphology + uint16 intensity | 64 | 0.044785 | 0.005499 | 0.006232 | 0.007418 | 6.04× | GPU-CuPy | 31.2 / 575.0 MiB |
| 2048² morphology | 96 | 0.019776 | 0.001754 | 0.002779 | 0.005260 | 3.76× | GPU-CuPy | 116.6 / 1400.0 MiB |
| 2048² morphology + uint16 intensity | 96 | 0.282568 | 0.005942 | 0.007394 | 0.012657 | 22.33× | GPU-CuPy | 124.6 / 2300.0 MiB |
| 32×256×256 morphology | 96 | 0.018424 | 0.004424 | 0.006037 | 0.006845 | 2.69× | GPU-CuPy | 58.3 / 820.0 MiB |
| 32×256×256 morphology + uint16 intensity | 96 | 0.135143 | 0.008417 | 0.009746 | 0.011708 | 11.54× | GPU-CuPy | 62.3 / 1270.0 MiB |
| 8×512² plane-wise morphology | 128 | 0.139609 | 0.011532 | 0.012788 | 0.014115 | 9.89× | GPU-CuPy | 15.1 / 460.0 MiB |
| 6×512² plane-wise morphology + float32 intensity | 96 | 0.089668 | 0.068667 | 0.069345 | 0.074709 | 1.20× | GPU-CuPy | 18.6 / 550.0 MiB |
| 64×512×512 confocal-like volume + uint16 intensity | 96 | 1.149008 | 0.020509 | 0.020472 | 0.034910 | 32.91× | GPU-CuPy | 498.5 / 10160.0 MiB |
| 16×1024² confocal-like plane-wise morphology | 1024 | 0.763934 | 0.024062 | 0.035023 | 0.046388 | 16.47× | GPU-CuPy | 96.3 / 3520.0 MiB |
| 256² morphology + uint16 intensity, 1,024 objects | 1024 | 0.188515 | 0.005926 | 0.015892 | 0.017014 | 11.08× | GPU-CuPy | 2.0 / 92.8 MiB |

## Historical-provider comparison

The preserved `measurements-cucim-windows-rtx5090.json` artifact contains 14 matching case IDs and input SHA-256 values. Comparing transfer-inclusive medians, the production CuPy provider is faster in 14 of 14 matched cases: **1.94×** geometric mean, with a **1.36–3.61×** range. This comparison is the basis for removing cuCIM from the active measurement and installation paths; the old artifact remains immutable historical evidence.

## Method notes

- CPU samples are complete typed-table calls.
- GPU resident packed samples end at a synchronized device matrix.
- GPU resident + table samples include D2H and typed host finalization.
- GPU full public samples additionally include authored input transfers.
- Screening compares CPU with the full public GPU boundary.
- The memory bound is the production `cupy-basic-measurements-memory-v1` model including its uncertainty reserve.

Reproduce with:

```powershell
python scripts/benchmark_gpu_measurements.py --profile full
python scripts/benchmark_gpu_measurements.py --validate-existing docs/benchmarks/measurements-cupy-windows-rtx5090.json
```
