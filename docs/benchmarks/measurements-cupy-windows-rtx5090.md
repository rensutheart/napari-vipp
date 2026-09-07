# GPU basic Measurements evidence

Generated: `2026-09-07T21:39:06.397172+00:00`

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
| 256² morphology | 4 | 0.000977 | 0.001833 | 0.002178 | 0.001974 | 0.49× | CPU | 1.8 / 81.5 MiB |
| 256² morphology + uint16 intensity | 4 | 0.001232 | 0.006360 | 0.007057 | 0.007341 | 0.17× | CPU | 2.0 / 92.8 MiB |
| 512² morphology | 16 | 0.002653 | 0.001685 | 0.002061 | 0.002313 | 1.15× | GPU-CuPy | 7.3 / 134.0 MiB |
| 512² morphology + uint16 intensity | 16 | 0.004547 | 0.014811 | 0.013842 | 0.013792 | 0.33× | CPU | 7.8 / 179.0 MiB |
| 1024² morphology | 64 | 0.010019 | 0.001695 | 0.002523 | 0.003550 | 2.82× | GPU-CuPy | 29.2 / 350.0 MiB |
| 1024² morphology + uint16 intensity | 64 | 0.054232 | 0.007218 | 0.007806 | 0.008514 | 6.37× | GPU-CuPy | 31.2 / 575.0 MiB |
| 2048² morphology | 96 | 0.020398 | 0.001992 | 0.002662 | 0.005815 | 3.51× | GPU-CuPy | 116.6 / 1400.0 MiB |
| 2048² morphology + uint16 intensity | 96 | 0.324129 | 0.008015 | 0.011264 | 0.011380 | 28.48× | GPU-CuPy | 124.6 / 2300.0 MiB |
| 32×256×256 morphology | 96 | 0.018366 | 0.004138 | 0.004914 | 0.006266 | 2.93× | GPU-CuPy | 58.3 / 820.0 MiB |
| 32×256×256 morphology + uint16 intensity | 96 | 0.163308 | 0.007566 | 0.009854 | 0.011665 | 14.00× | GPU-CuPy | 62.3 / 1270.0 MiB |
| 8×512² plane-wise morphology | 128 | 0.159525 | 0.013042 | 0.014031 | 0.018368 | 8.68× | GPU-CuPy | 15.1 / 460.0 MiB |
| 6×512² plane-wise morphology + float32 intensity | 96 | 0.099433 | 0.088723 | 0.090692 | 0.087588 | 1.14× | GPU-CuPy | 18.6 / 550.0 MiB |
| 64×512×512 confocal-like volume + uint16 intensity | 96 | 1.316128 | 0.022378 | 0.024104 | 0.040060 | 32.85× | GPU-CuPy | 498.5 / 10160.0 MiB |
| 16×1024² confocal-like plane-wise morphology | 1024 | 0.706253 | 0.028519 | 0.037638 | 0.051895 | 13.61× | GPU-CuPy | 96.3 / 3520.0 MiB |
| 256² morphology + uint16 intensity, 1,024 objects | 1024 | 0.201463 | 0.007097 | 0.018169 | 0.019021 | 10.59× | GPU-CuPy | 2.0 / 92.8 MiB |

## Historical-provider comparison

The preserved `measurements-cucim-windows-rtx5090.json` artifact contains 14 matching case IDs and input SHA-256 values. Comparing transfer-inclusive medians, the production CuPy provider is faster in 14 of 14 matched cases: **1.76×** geometric mean, with a **1.18–3.47×** range. This comparison is the basis for removing cuCIM from the active measurement and installation paths; the old artifact remains immutable historical evidence.

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
