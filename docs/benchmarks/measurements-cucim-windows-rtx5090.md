# GPU basic Measurements evidence

Generated: `2026-08-04T13:43:52.902003+00:00`

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
| 256² morphology | 4 | 0.001039 | 0.004363 | 0.004784 | 0.006850 | 0.15× | CPU | 2.1 / 77.0 MiB |
| 256² morphology + uint16 intensity | 4 | 0.001251 | 0.009831 | 0.011172 | 0.010876 | 0.11× | CPU | 2.2 / 85.8 MiB |
| 512² morphology | 16 | 0.002978 | 0.004460 | 0.005245 | 0.005840 | 0.51× | CPU | 8.3 / 116.0 MiB |
| 512² morphology + uint16 intensity | 16 | 0.004838 | 0.015688 | 0.016272 | 0.017456 | 0.28× | CPU | 8.8 / 151.0 MiB |
| 1024² morphology | 64 | 0.010737 | 0.004404 | 0.005156 | 0.005731 | 1.87× | GPU-cuCIM | 33.2 / 272.0 MiB |
| 1024² morphology + uint16 intensity | 64 | 0.053175 | 0.010891 | 0.011830 | 0.012509 | 4.25× | GPU-cuCIM | 35.2 / 435.0 MiB |
| 2048² morphology | 96 | 0.020447 | 0.004691 | 0.005714 | 0.008840 | 2.31× | GPU-cuCIM | 132.6 / 1040.0 MiB |
| 2048² morphology + uint16 intensity | 96 | 0.308376 | 0.012404 | 0.014096 | 0.018843 | 16.37× | GPU-cuCIM | 140.6 / 1740.0 MiB |
| 32×256×256 morphology | 96 | 0.024868 | 0.009883 | 0.010503 | 0.011651 | 2.13× | GPU-cuCIM | 66.3 / 580.0 MiB |
| 32×256×256 morphology + uint16 intensity | 96 | 0.170783 | 0.019695 | 0.021251 | 0.023455 | 7.28× | GPU-cuCIM | 70.3 / 930.0 MiB |
| 8×512² plane-wise morphology | 128 | 0.154561 | 0.031287 | 0.033465 | 0.037300 | 4.14× | GPU-cuCIM | 15.1 / 272.0 MiB |
| 6×512² plane-wise morphology + float32 intensity | 96 | 0.101507 | 0.105645 | 0.101934 | 0.115008 | 0.88× | CPU | 19.6 / 325.0 MiB |
| 64×512×512 confocal-like volume + uint16 intensity | 96 | 1.133357 | 0.024480 | 0.026023 | 0.047417 | 23.90× | GPU-cuCIM | 562.5 / 7440.0 MiB |
| 16×1024² confocal-like plane-wise morphology | 1024 | 0.713705 | 0.100252 | 0.117846 | 0.122483 | 5.83× | GPU-cuCIM | 92.3 / 1920.0 MiB |

## Method notes

- CPU samples are complete typed-table calls.
- GPU resident packed samples end at a synchronized device matrix.
- GPU resident + table samples include D2H and typed host finalization.
- GPU full public samples additionally include authored input transfers.
- Screening compares CPU with the full public GPU boundary.
- The memory bound is the production `cucim-basic-measurements-memory-v1` model including its uncertainty reserve.

Reproduce with:

```powershell
python scripts/benchmark_gpu_measurements.py --profile full
python scripts/benchmark_gpu_measurements.py --validate-existing docs/benchmarks/measurements-cucim-windows-rtx5090.json
```
