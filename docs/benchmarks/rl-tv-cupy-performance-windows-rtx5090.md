# CuPy Richardson-Lucy TV large-stack performance

- Generated: `2026-07-29T17:45:34.955198+00:00`
- Device: `NVIDIA GeForce RTX 5090`
- Host processor: `Intel64 Family 6 Model 165 Stepping 3, GenuineIntel`
- Iterations: `25`
- TV regularization: `0.002`
- TV epsilon: `1e-06`
- Filter epsilon: `1e-12`
- Denominator floor: `0.05`
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
| Private real-acquisition single-channel ZYX volume | 8,507,700 | 36.004 s | 0.489 s | 0.447 s | 0.023 s | 73.66x | GPU-CuPy |
| Medium 3D shape stress (synthetic) | 16,777,216 | 58.816 s | 0.537 s | 0.470 s | 0.028 s | 109.46x | GPU-CuPy |

## Interpretation

- The CPU and GPU columns compare the same authored operation and
  parameters. Resident time is shown only to explain pipeline-context
  gains; the screen winner uses transfer-inclusive GPU time. Three
  paired rounds are enough for a descriptive large-stack result but
  do not replace VIPP's longer durable node-benchmark evidence.
- Synthetic cases are shape-and-memory stress tests, not claims that
  independent random voxels reproduce confocal image statistics. The
  optional private ND2 volume supplies the real-acquisition anchor.
- This evidence uses the exact positive shipped RL-TV defaults:
  `tv_regularization=0.002`, `tv_epsilon=1e-6`,
  `filter_epsilon=1e-12`, and `denominator_floor=0.05`.
- Large-stack results do not broaden the scientific region. New TV
  weight, epsilon, floor, iteration, PSF, dtype, or safety-option
  regions require a versioned numerical study across adversarial
  fixtures.
- The optional private ND2 case publishes workload metadata and timings
  but no path, filename, content digest, or pixels. Its generated
  Gaussian timing PSF is not a measured restoration-quality PSF.
