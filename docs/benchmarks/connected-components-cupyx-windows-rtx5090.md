# Connected Components CPU/CuPyX evidence

Generated: `2026-08-02T11:53:22.696209+00:00`

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
| plane-256-sparse-face | 256×256 | sparse / Face connected | 242 µs | 576 µs | 736 µs | 0.33× | 0.34 MiB / 0.75 MiB | CPU |
| plane-256-dense-full | 256×256 | dense / Full connectivity | 651 µs | 600 µs | 778 µs | 0.84× | 0.32 MiB / 0.75 MiB | CPU |
| plane-256-checkerboard-face | 256×256 | checkerboard / Face connected | 287 µs | 650 µs | 762 µs | 0.38× | 0.70 MiB / 0.75 MiB | CPU |
| plane-512-sparse-face | 512×512 | sparse / Face connected | 1.62 ms | 1.09 ms | 1.79 ms | 0.90× | 1.35 MiB / 3.00 MiB | CPU |
| plane-512-dense-full | 512×512 | dense / Full connectivity | 2.57 ms | 760 µs | 1.39 ms | 1.85× | 1.25 MiB / 3.00 MiB | GPU-CuPyX |
| plane-512-checkerboard-face | 512×512 | checkerboard / Face connected | 1.79 ms | 601 µs | 1.32 ms | 1.35× | 2.78 MiB / 3.00 MiB | GPU-CuPyX |
| plane-1024-sparse-face | 1024×1024 | sparse / Face connected | 3.47 ms | 598 µs | 2.36 ms | 1.47× | 5.35 MiB / 12.00 MiB | GPU-CuPyX |
| plane-1024-dense-full | 1024×1024 | dense / Full connectivity | 9.71 ms | 811 µs | 3.07 ms | 3.16× | 5.00 MiB / 12.00 MiB | GPU-CuPyX |
| plane-1024-checkerboard-face | 1024×1024 | checkerboard / Face connected | 9.92 ms | 789 µs | 3.12 ms | 3.18× | 11.08 MiB / 12.00 MiB | GPU-CuPyX |
| plane-2048-sparse-face | 2048×2048 | sparse / Face connected | 19.80 ms | 1.27 ms | 7.65 ms | 2.59× | 21.38 MiB / 48.00 MiB | GPU-CuPyX |
| plane-2048-dense-full | 2048×2048 | dense / Full connectivity | 40.49 ms | 1.29 ms | 7.67 ms | 5.28× | 20.01 MiB / 48.00 MiB | GPU-CuPyX |
| plane-2048-checkerboard-face | 2048×2048 | checkerboard / Face connected | 27.71 ms | 820 µs | 6.40 ms | 4.33× | 44.29 MiB / 48.00 MiB | GPU-CuPyX |
| volume-48x128x128-sparse-face | 48×128×128 | sparse / Face connected | 7.08 ms | 692 µs | 1.83 ms | 3.87× | 4.01 MiB / 9.00 MiB | GPU-CuPyX |
| volume-64x192x192-dense-full | 64×192×192 | dense / Full connectivity | 75.60 ms | 897 µs | 5.02 ms | 15.05× | 11.25 MiB / 27.00 MiB | GPU-CuPyX |
| volume-32x256x256-checkerboard-face | 32×256×256 | checkerboard / Face connected | 21.35 ms | 913 µs | 4.17 ms | 5.12× | 22.15 MiB / 24.00 MiB | GPU-CuPyX |
| volume-64x512x512-sparse-face | 64×512×512 | sparse / Face connected | 107.68 ms | 1.22 ms | 28.50 ms | 3.78× | 85.31 MiB / 192.00 MiB | GPU-CuPyX |
| stack-8x512x512-sparse-face | 8×512×512 | sparse / Face connected | 8.32 ms | 4.91 ms | 8.09 ms | 1.03× | 10.16 MiB / 11.75 MiB | GPU-CuPyX |
| stack-4x1024x1024-dense-full | 4×1024×1024 | dense / Full connectivity | 40.24 ms | 2.91 ms | 10.69 ms | 3.77× | 20.00 MiB / 27.00 MiB | GPU-CuPyX |

## Plane crossover screening

- `plane-checkerboard-face`: resident crossover `512`; transfer-inclusive crossover `512` among tested extents `[256, 512, 1024, 2048]`.
- `plane-dense-full`: resident crossover `256`; transfer-inclusive crossover `512` among tested extents `[256, 512, 1024, 2048]`.
- `plane-sparse-face`: resident crossover `512`; transfer-inclusive crossover `1024` among tested extents `[256, 512, 1024, 2048]`.

## Interpretation

Sparse masks with many isolated components can keep CPU competitive to larger extents; dense and checkerboard workloads can strongly favor resident CuPyX. Auto selection should therefore use measured workload records rather than a size-only rule.
