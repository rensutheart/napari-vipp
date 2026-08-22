# GPU basic Measurements evidence

Generated: `2026-08-22T15:06:22.695934+00:00`

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
| 256² morphology | 4 | 0.001003 | 0.001616 | 0.001907 | 0.002009 | 0.50× | CPU | 1.8 / 81.5 MiB |
| 256² morphology + uint16 intensity | 4 | 0.001248 | 0.005966 | 0.007272 | 0.006416 | 0.19× | CPU | 2.0 / 92.8 MiB |
| 512² morphology | 16 | 0.002719 | 0.001617 | 0.001985 | 0.002190 | 1.24× | GPU-CuPy | 7.3 / 134.0 MiB |
| 512² morphology + uint16 intensity | 16 | 0.004240 | 0.012752 | 0.012974 | 0.012730 | 0.33× | CPU | 7.8 / 179.0 MiB |
| 1024² morphology | 64 | 0.009803 | 0.001669 | 0.002326 | 0.002945 | 3.33× | GPU-CuPy | 29.2 / 350.0 MiB |
| 1024² morphology + uint16 intensity | 64 | 0.045217 | 0.005729 | 0.006582 | 0.007781 | 5.81× | GPU-CuPy | 31.2 / 575.0 MiB |
| 2048² morphology | 96 | 0.020046 | 0.001792 | 0.002908 | 0.005455 | 3.67× | GPU-CuPy | 116.6 / 1400.0 MiB |
| 2048² morphology + uint16 intensity | 96 | 0.282927 | 0.006269 | 0.007734 | 0.010557 | 26.80× | GPU-CuPy | 124.6 / 2300.0 MiB |
| 32×256×256 morphology | 96 | 0.018076 | 0.004604 | 0.005830 | 0.008214 | 2.20× | GPU-CuPy | 58.3 / 820.0 MiB |
| 32×256×256 morphology + uint16 intensity | 96 | 0.142206 | 0.009086 | 0.010140 | 0.011296 | 12.59× | GPU-CuPy | 62.3 / 1270.0 MiB |
| 8×512² plane-wise morphology | 128 | 0.142564 | 0.011530 | 0.012854 | 0.014185 | 10.05× | GPU-CuPy | 15.1 / 460.0 MiB |
| 6×512² plane-wise morphology + float32 intensity | 96 | 0.090530 | 0.069012 | 0.070046 | 0.072048 | 1.26× | GPU-CuPy | 18.6 / 550.0 MiB |
| 64×512×512 confocal-like volume + uint16 intensity | 96 | 1.099115 | 0.020167 | 0.021843 | 0.038003 | 28.92× | GPU-CuPy | 498.5 / 10160.0 MiB |
| 16×1024² confocal-like plane-wise morphology | 1024 | 0.687418 | 0.024120 | 0.030996 | 0.043843 | 15.68× | GPU-CuPy | 96.3 / 3520.0 MiB |
| 256² morphology + uint16 intensity, 1,024 objects | 1024 | 0.167790 | 0.005739 | 0.015261 | 0.015371 | 10.92× | GPU-CuPy | 2.0 / 92.8 MiB |

## Historical-provider comparison

The preserved `measurements-cucim-windows-rtx5090.json` artifact contains 14 matching case IDs and input SHA-256 values. Comparing transfer-inclusive medians, the production CuPy provider is faster in 14 of 14 matched cases: **1.90×** geometric mean, with a **1.25–3.41×** range. This comparison is the basis for removing cuCIM from the active measurement and installation paths; the old artifact remains immutable historical evidence.

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
