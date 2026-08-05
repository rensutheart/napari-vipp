# CuPy Richardson-Lucy large-stack performance

- Generated: `2026-08-05T09:28:26.062095+00:00`
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
| Large 3D shape stress (synthetic) | 67,108,864 | 141.837 s | 1.448 s | 1.152 s | 0.124 s | 97.97x | GPU-CuPy |
| Medium 3D shape stress (synthetic) | 16,777,216 | 36.138 s | 0.456 s | 0.394 s | 0.032 s | 79.36x | GPU-CuPy |
| Private real-acquisition single-channel ZYX volume | 8,507,700 | 25.512 s | 0.593 s | 0.549 s | 0.026 s | 43.05x | GPU-CuPy |

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
