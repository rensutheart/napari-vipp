# CuPy Richardson-Lucy large-stack performance

- Generated: `2026-08-06T08:51:03.965592+00:00`
- Device: `NVIDIA GeForce RTX 5090`
- Host processor: `Intel64 Family 6 Model 165 Stepping 3, GenuineIntel`
- Iterations: `25`
- Filter epsilon: `1e-08`
- Paired warm rounds: `3`

This is machine-local production-path evidence, not a portable speed
claim or a reusable optimizer record. It is a short three-pair
descriptive screen on deliberately expensive workloads. Disk I/O and
input generation are excluded. GPU end-to-end time
includes Image/PSF H2D transfer, synchronized resident compute, output
D2H transfer, and private allocator cleanup. Scientific parity passed
before the observed screening winner was reported.

| Workload | Voxels | CPU median | GPU end-to-end | GPU resident | Transfer | Speedup | Screen winner |
|---|---:|---:|---:|---:|---:|---:|:---|
| Large 3D shape stress (synthetic) | 67,108,864 | 144.137 s | 1.524 s | 1.284 s | 0.112 s | 94.58x | GPU-CuPy |
| Medium 3D shape stress (synthetic) | 16,777,216 | 34.968 s | 0.411 s | 0.373 s | 0.029 s | 85.06x | GPU-CuPy |
| Private real-acquisition single-channel ZYX volume | 8,507,700 | 24.381 s | 0.551 s | 0.518 s | 0.025 s | 45.03x | GPU-CuPy |

## Interpretation

- The CPU and GPU columns compare the same authored operation and
  parameters. Resident time is shown only to explain pipeline-context
  gains; the screen winner uses transfer-inclusive GPU time. Three
  paired rounds are enough for a descriptive large-stack result but
  do not replace VIPP's longer durable node-benchmark evidence.
- Synthetic cases are shape-and-memory stress tests, not claims that
  independent random voxels reproduce confocal image statistics. The
  optional private ND2 volume supplies the real-acquisition anchor.
- `filter_epsilon=1e-8` is the currently admitted measured point, not
  an assertion that it is inherently the only useful GPU epsilon.
- Large-stack results do not broaden the scientific region. New
  epsilon, iteration, PSF, dtype, or safety-option regions require a
  versioned numerical study across adversarial fixtures.
- The optional private ND2 case publishes workload metadata and timings
  but no path, filename, content digest, or pixels. Its generated
  Gaussian timing PSF is not a measured restoration-quality PSF.
