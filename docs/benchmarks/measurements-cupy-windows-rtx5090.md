# GPU basic Measurements evidence

Generated: `2026-08-29T13:48:52.246699+00:00`

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
| 256² morphology | 4 | 0.001084 | 0.001634 | 0.002160 | 0.002668 | 0.41× | CPU | 1.8 / 81.5 MiB |
| 256² morphology + uint16 intensity | 4 | 0.001282 | 0.006224 | 0.006429 | 0.006627 | 0.19× | CPU | 2.0 / 92.8 MiB |
| 512² morphology | 16 | 0.003048 | 0.001637 | 0.002105 | 0.002242 | 1.36× | GPU-CuPy | 7.3 / 134.0 MiB |
| 512² morphology + uint16 intensity | 16 | 0.004386 | 0.012100 | 0.012575 | 0.013280 | 0.33× | CPU | 7.8 / 179.0 MiB |
| 1024² morphology | 64 | 0.010481 | 0.001740 | 0.002721 | 0.003348 | 3.13× | GPU-CuPy | 29.2 / 350.0 MiB |
| 1024² morphology + uint16 intensity | 64 | 0.047300 | 0.006129 | 0.006807 | 0.007975 | 5.93× | GPU-CuPy | 31.2 / 575.0 MiB |
| 2048² morphology | 96 | 0.021003 | 0.001824 | 0.002763 | 0.005511 | 3.81× | GPU-CuPy | 116.6 / 1400.0 MiB |
| 2048² morphology + uint16 intensity | 96 | 0.287644 | 0.006570 | 0.008589 | 0.012414 | 23.17× | GPU-CuPy | 124.6 / 2300.0 MiB |
| 32×256×256 morphology | 96 | 0.018970 | 0.005022 | 0.006340 | 0.007773 | 2.44× | GPU-CuPy | 58.3 / 820.0 MiB |
| 32×256×256 morphology + uint16 intensity | 96 | 0.146309 | 0.009542 | 0.011115 | 0.013158 | 11.12× | GPU-CuPy | 62.3 / 1270.0 MiB |
| 8×512² plane-wise morphology | 128 | 0.155012 | 0.013433 | 0.014185 | 0.015782 | 9.82× | GPU-CuPy | 15.1 / 460.0 MiB |
| 6×512² plane-wise morphology + float32 intensity | 96 | 0.097714 | 0.074268 | 0.077413 | 0.084316 | 1.16× | GPU-CuPy | 18.6 / 550.0 MiB |
| 64×512×512 confocal-like volume + uint16 intensity | 96 | 1.188198 | 0.020772 | 0.022297 | 0.036116 | 32.90× | GPU-CuPy | 498.5 / 10160.0 MiB |
| 16×1024² confocal-like plane-wise morphology | 1024 | 0.765644 | 0.027727 | 0.038837 | 0.044767 | 17.10× | GPU-CuPy | 96.3 / 3520.0 MiB |
| 256² morphology + uint16 intensity, 1,024 objects | 1024 | 0.181659 | 0.006061 | 0.016543 | 0.016058 | 11.31× | GPU-CuPy | 2.0 / 92.8 MiB |

## Historical-provider comparison

The preserved `measurements-cucim-windows-rtx5090.json` artifact contains 14 matching case IDs and input SHA-256 values. Comparing transfer-inclusive medians, the production CuPy provider is faster in 14 of 14 matched cases: **1.77×** geometric mean, with a **1.31–2.74×** range. This comparison is the basis for removing cuCIM from the active measurement and installation paths; the old artifact remains immutable historical evidence.

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
