# CuPy Richardson-Lucy large-stack performance

- Generated: `2026-08-04T22:38:05.293212+00:00`
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
| Private real-acquisition single-channel ZYX volume | 8,507,700 | 24.898 s | 0.414 s | 0.379 s | 0.017 s | 60.20x | GPU-CuPy |
| Medium 3D shape stress (synthetic) | 16,777,216 | 35.997 s | 0.455 s | 0.372 s | 0.032 s | 79.14x | GPU-CuPy |
| Large 3D shape stress (synthetic) | 67,108,864 | 137.820 s | 1.517 s | 1.215 s | 0.175 s | 90.87x | GPU-CuPy |

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
