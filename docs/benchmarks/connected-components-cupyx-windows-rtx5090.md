# Connected Components CPU/CuPyX evidence

Generated: `2026-08-06T08:33:23.887129+00:00`

This is machine-local screening evidence, not a portable performance claim.

## Outcome

- Exact admission cases: **16 / 16 passed**.
- GPU: **NVIDIA GeForce RTX 5090**; CUDA runtime `13020`.
- Cancellation and cleanup: **pass** at synchronized leading-block boundaries.
- Memory estimate covered every observed private-pool high-water mark: **True**.
- Labels must match SciPy bit for bit as native `int32`; equivalence up to relabeling is insufficient.
- Leading non-spatial blocks are independent and restart label IDs at one.

## Timing method

- CPU: case-cold call plus warm host medians.
- GPU resident: synchronized compute with input already on device and output left resident.
- GPU transfer-inclusive: host-to-device, compute, device-to-host, then synchronization.
- GPU case-cold: empty private allocator pool after one process-level runtime warmup.
- VRAM observation: isolated CuPy pool reserved high-water; it is not device-wide telemetry.

## Measured workloads

| Workload | Shape | Pattern / connectivity | CPU median | GPU resident | GPU transfer-inclusive | E2E speedup | VRAM observed / estimated | Choice |
|---|---:|---|---:|---:|---:|---:|---:|---|
| plane-256-sparse-face | 256×256 | sparse / Face connected | 359 µs | 679 µs | 899 µs | 0.40× | 0.34 MiB / 0.75 MiB | CPU |
| plane-256-dense-full | 256×256 | dense / Full connectivity | 644 µs | 746 µs | 903 µs | 0.71× | 0.32 MiB / 0.75 MiB | CPU |
| plane-256-checkerboard-face | 256×256 | checkerboard / Face connected | 479 µs | 829 µs | 936 µs | 0.51× | 0.70 MiB / 0.75 MiB | CPU |
| plane-512-sparse-face | 512×512 | sparse / Face connected | 1.12 ms | 710 µs | 1.66 ms | 0.68× | 1.35 MiB / 3.00 MiB | CPU |
| plane-512-dense-full | 512×512 | dense / Full connectivity | 2.79 ms | 898 µs | 1.68 ms | 1.67× | 1.25 MiB / 3.00 MiB | GPU-CuPyX |
| plane-512-checkerboard-face | 512×512 | checkerboard / Face connected | 1.78 ms | 645 µs | 1.29 ms | 1.38× | 2.78 MiB / 3.00 MiB | GPU-CuPyX |
| plane-1024-sparse-face | 1024×1024 | sparse / Face connected | 4.68 ms | 601 µs | 2.93 ms | 1.60× | 5.35 MiB / 12.00 MiB | GPU-CuPyX |
| plane-1024-dense-full | 1024×1024 | dense / Full connectivity | 10.91 ms | 959 µs | 3.02 ms | 3.61× | 5.00 MiB / 12.00 MiB | GPU-CuPyX |
| plane-1024-checkerboard-face | 1024×1024 | checkerboard / Face connected | 8.03 ms | 737 µs | 2.89 ms | 2.78× | 11.08 MiB / 12.00 MiB | GPU-CuPyX |
| plane-2048-sparse-face | 2048×2048 | sparse / Face connected | 20.80 ms | 1.13 ms | 9.52 ms | 2.19× | 21.38 MiB / 48.00 MiB | GPU-CuPyX |
| plane-2048-dense-full | 2048×2048 | dense / Full connectivity | 41.80 ms | 1.06 ms | 8.07 ms | 5.18× | 20.01 MiB / 48.00 MiB | GPU-CuPyX |
| plane-2048-checkerboard-face | 2048×2048 | checkerboard / Face connected | 48.75 ms | 1.05 ms | 7.88 ms | 6.19× | 44.29 MiB / 48.00 MiB | GPU-CuPyX |
| volume-48x128x128-sparse-face | 48×128×128 | sparse / Face connected | 4.89 ms | 1.12 ms | 3.03 ms | 1.61× | 4.01 MiB / 9.00 MiB | GPU-CuPyX |
| volume-64x192x192-dense-full | 64×192×192 | dense / Full connectivity | 74.09 ms | 951 µs | 4.85 ms | 15.29× | 11.25 MiB / 27.00 MiB | GPU-CuPyX |
| volume-32x256x256-checkerboard-face | 32×256×256 | checkerboard / Face connected | 19.70 ms | 843 µs | 4.78 ms | 4.12× | 22.15 MiB / 24.00 MiB | GPU-CuPyX |
| volume-64x512x512-sparse-face | 64×512×512 | sparse / Face connected | 89.01 ms | 1.48 ms | 30.19 ms | 2.95× | 85.31 MiB / 192.00 MiB | GPU-CuPyX |
| stack-8x512x512-sparse-face | 8×512×512 | sparse / Face connected | 10.51 ms | 4.99 ms | 9.00 ms | 1.17× | 10.16 MiB / 11.75 MiB | GPU-CuPyX |
| stack-4x1024x1024-dense-full | 4×1024×1024 | dense / Full connectivity | 45.48 ms | 2.80 ms | 10.43 ms | 4.36× | 20.00 MiB / 27.00 MiB | GPU-CuPyX |

## Plane crossover screening

- `plane-checkerboard-face`: resident crossover `512`; transfer-inclusive crossover `512` among tested extents `[256, 512, 1024, 2048]`.
- `plane-dense-full`: resident crossover `512`; transfer-inclusive crossover `512` among tested extents `[256, 512, 1024, 2048]`.
- `plane-sparse-face`: resident crossover `512`; transfer-inclusive crossover `1024` among tested extents `[256, 512, 1024, 2048]`.

## Interpretation

Sparse masks with many isolated components can keep CPU competitive to larger extents; dense and checkerboard workloads can strongly favor resident CuPyX. Auto selection should therefore use measured workload records rather than a size-only rule.
