# GPU basic Measurements evidence

Generated: `2026-08-22T08:14:17.238733+00:00`

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
| 256² morphology | 4 | 0.001040 | 0.001801 | 0.001907 | 0.002012 | 0.52× | CPU | 1.8 / 81.5 MiB |
| 256² morphology + uint16 intensity | 4 | 0.001245 | 0.005645 | 0.006200 | 0.006743 | 0.18× | CPU | 2.0 / 92.8 MiB |
| 512² morphology | 16 | 0.002878 | 0.002058 | 0.002130 | 0.002138 | 1.35× | GPU-CuPy | 7.3 / 134.0 MiB |
| 512² morphology + uint16 intensity | 16 | 0.004448 | 0.011840 | 0.012813 | 0.014116 | 0.32× | CPU | 7.8 / 179.0 MiB |
| 1024² morphology | 64 | 0.010524 | 0.001808 | 0.002505 | 0.003260 | 3.23× | GPU-CuPy | 29.2 / 350.0 MiB |
| 1024² morphology + uint16 intensity | 64 | 0.050650 | 0.005885 | 0.007215 | 0.008714 | 5.81× | GPU-CuPy | 31.2 / 575.0 MiB |
| 2048² morphology | 96 | 0.020638 | 0.002071 | 0.002787 | 0.006521 | 3.16× | GPU-CuPy | 116.6 / 1400.0 MiB |
| 2048² morphology + uint16 intensity | 96 | 0.306995 | 0.006660 | 0.007905 | 0.012271 | 25.02× | GPU-CuPy | 124.6 / 2300.0 MiB |
| 32×256×256 morphology | 96 | 0.018196 | 0.004826 | 0.006181 | 0.007723 | 2.36× | GPU-CuPy | 58.3 / 820.0 MiB |
| 32×256×256 morphology + uint16 intensity | 96 | 0.153685 | 0.009383 | 0.014058 | 0.012057 | 12.75× | GPU-CuPy | 62.3 / 1270.0 MiB |
| 8×512² plane-wise morphology | 128 | 0.150134 | 0.012108 | 0.014142 | 0.016109 | 9.32× | GPU-CuPy | 15.1 / 460.0 MiB |
| 6×512² plane-wise morphology + float32 intensity | 96 | 0.093274 | 0.074869 | 0.073445 | 0.083391 | 1.12× | GPU-CuPy | 18.6 / 550.0 MiB |
| 64×512×512 confocal-like volume + uint16 intensity | 96 | 1.186538 | 0.021279 | 0.023335 | 0.036471 | 32.53× | GPU-CuPy | 498.5 / 10160.0 MiB |
| 16×1024² confocal-like plane-wise morphology | 1024 | 0.709484 | 0.023690 | 0.032354 | 0.047091 | 15.07× | GPU-CuPy | 96.3 / 3520.0 MiB |
| 256² morphology + uint16 intensity, 1,024 objects | 1024 | 0.171164 | 0.006357 | 0.015738 | 0.016277 | 10.52× | GPU-CuPy | 2.0 / 92.8 MiB |

## Historical-provider comparison

The preserved `measurements-cucim-windows-rtx5090.json` artifact contains 14 matching case IDs and input SHA-256 values. Comparing transfer-inclusive medians, the production CuPy provider is faster in 14 of 14 matched cases: **1.78×** geometric mean, with a **1.24–3.40×** range. This comparison is the basis for removing cuCIM from the active measurement and installation paths; the old artifact remains immutable historical evidence.

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
