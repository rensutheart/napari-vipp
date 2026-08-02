# Connected Components CPU/CuPyX evidence

Generated: `2026-08-02T11:33:09.562601+00:00`

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
| plane-256-sparse-face | 256×256 | sparse / Face connected | 253 µs | 566 µs | 793 µs | 0.32× | 0.34 MiB / 0.75 MiB | CPU |
| plane-256-dense-full | 256×256 | dense / Full connectivity | 659 µs | 595 µs | 864 µs | 0.76× | 0.32 MiB / 0.75 MiB | CPU |
| plane-256-checkerboard-face | 256×256 | checkerboard / Face connected | 311 µs | 596 µs | 1.09 ms | 0.29× | 0.70 MiB / 0.75 MiB | CPU |
| plane-512-sparse-face | 512×512 | sparse / Face connected | 1.16 ms | 593 µs | 1.19 ms | 0.97× | 1.35 MiB / 3.00 MiB | CPU |
| plane-512-dense-full | 512×512 | dense / Full connectivity | 3.14 ms | 790 µs | 1.17 ms | 2.68× | 1.25 MiB / 3.00 MiB | GPU-CuPyX |
| plane-512-checkerboard-face | 512×512 | checkerboard / Face connected | 2.86 ms | 1.04 ms | 1.28 ms | 2.23× | 2.78 MiB / 3.00 MiB | GPU-CuPyX |
| plane-1024-sparse-face | 1024×1024 | sparse / Face connected | 5.64 ms | 1.01 ms | 3.46 ms | 1.63× | 5.35 MiB / 12.00 MiB | GPU-CuPyX |
| plane-1024-dense-full | 1024×1024 | dense / Full connectivity | 15.11 ms | 1.17 ms | 3.75 ms | 4.03× | 5.00 MiB / 12.00 MiB | GPU-CuPyX |
| plane-1024-checkerboard-face | 1024×1024 | checkerboard / Face connected | 11.50 ms | 751 µs | 2.38 ms | 4.84× | 11.08 MiB / 12.00 MiB | GPU-CuPyX |
| plane-2048-sparse-face | 2048×2048 | sparse / Face connected | 21.49 ms | 779 µs | 8.56 ms | 2.51× | 21.38 MiB / 48.00 MiB | GPU-CuPyX |
| plane-2048-dense-full | 2048×2048 | dense / Full connectivity | 47.05 ms | 1.33 ms | 7.82 ms | 6.02× | 20.01 MiB / 48.00 MiB | GPU-CuPyX |
| plane-2048-checkerboard-face | 2048×2048 | checkerboard / Face connected | 60.77 ms | 1.41 ms | 11.61 ms | 5.24× | 44.29 MiB / 48.00 MiB | GPU-CuPyX |
| volume-48x128x128-sparse-face | 48×128×128 | sparse / Face connected | 7.73 ms | 921 µs | 2.92 ms | 2.65× | 4.01 MiB / 9.00 MiB | GPU-CuPyX |
| volume-64x192x192-dense-full | 64×192×192 | dense / Full connectivity | 91.19 ms | 1.72 ms | 7.21 ms | 12.64× | 11.25 MiB / 27.00 MiB | GPU-CuPyX |
| volume-32x256x256-checkerboard-face | 32×256×256 | checkerboard / Face connected | 24.50 ms | 725 µs | 4.36 ms | 5.62× | 22.15 MiB / 24.00 MiB | GPU-CuPyX |
| volume-64x512x512-sparse-face | 64×512×512 | sparse / Face connected | 140.38 ms | 1.47 ms | 37.54 ms | 3.74× | 85.31 MiB / 192.00 MiB | GPU-CuPyX |
| stack-8x512x512-sparse-face | 8×512×512 | sparse / Face connected | 9.35 ms | 6.75 ms | 8.48 ms | 1.10× | 10.16 MiB / 11.75 MiB | GPU-CuPyX |
| stack-4x1024x1024-dense-full | 4×1024×1024 | dense / Full connectivity | 64.18 ms | 6.61 ms | 13.55 ms | 4.74× | 20.00 MiB / 27.00 MiB | GPU-CuPyX |

## Plane crossover screening

- `plane-checkerboard-face`: resident crossover `512`; transfer-inclusive crossover `512` among tested extents `[256, 512, 1024, 2048]`.
- `plane-dense-full`: resident crossover `256`; transfer-inclusive crossover `512` among tested extents `[256, 512, 1024, 2048]`.
- `plane-sparse-face`: resident crossover `512`; transfer-inclusive crossover `1024` among tested extents `[256, 512, 1024, 2048]`.

## Interpretation

Sparse masks with many isolated components can keep CPU competitive to larger extents; dense and checkerboard workloads can strongly favor resident CuPyX. Auto selection should therefore use measured workload records rather than a size-only rule.
