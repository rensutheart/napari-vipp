# CuPy Sigma Filter admission and performance evidence

- Generated: `2026-08-02T00:09:55.102861+00:00`
- Device: `NVIDIA GeForce RTX 5090`
- Profile: `full`
- Exact admission cases: `10`
- Matched rejection cases: `10`
- Timed cases: `18`
- Cancellation and cleanup: `pass`

Every admission and timed workload compared the production CPU operation
with the resident CuPy provider bit for bit, including float32 signed zero
and subnormal arithmetic. Timings are a short machine-local screen, not a
portable claim or durable optimizer record. Case-cold GPU timings include
allocations and transfers but not first-process JIT, because admission
compiled the provider first.

| Case | Shape | Radius | CPU warm | GPU case-cold E2E | GPU warm E2E | GPU resident | Transfers | E2E speedup | 95% paired lower | Choice |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|
| 256² plane, r=0.5 | 256×256 | 0.5 | 0.004604 s | 0.016713 s | 0.000672 s | 0.000374 s | 0.000156 s | 6.85x | 5.95x | CPU |
| 512² plane, r=0.5 | 512×512 | 0.5 | 0.020366 s | 0.022080 s | 0.001095 s | 0.000546 s | 0.000365 s | 18.60x | 16.92x | CPU |
| 1024² plane, r=0.5 | 1024×1024 | 0.5 | 0.084091 s | 0.014488 s | 0.001877 s | 0.000722 s | 0.001066 s | 44.80x | 42.48x | GPU-CuPy |
| 2048² plane, r=0.5 | 2048×2048 | 0.5 | 0.370669 s | 0.006433 s | 0.005394 s | 0.001475 s | 0.004034 s | 68.71x | 65.58x | GPU-CuPy |
| 256² plane, r=2 | 256×256 | 2 | 0.017010 s | 0.009998 s | 0.000772 s | 0.000461 s | 0.000197 s | 22.03x | 20.43x | CPU |
| 512² plane, r=2 | 512×512 | 2 | 0.055378 s | 0.014207 s | 0.001112 s | 0.000622 s | 0.000264 s | 49.78x | 49.78x | GPU-CuPy |
| 1024² plane, r=2 | 1024×1024 | 2 | 0.229352 s | 0.025249 s | 0.002626 s | 0.001272 s | 0.001329 s | 87.33x | 81.76x | GPU-CuPy |
| 2048² plane, r=2 | 2048×2048 | 2 | 0.940072 s | 0.012016 s | 0.011187 s | 0.003864 s | 0.007225 s | 84.03x | 72.24x | GPU-CuPy |
| 256² plane, r=5 | 256×256 | 5 | 0.056627 s | 0.002801 s | 0.001178 s | 0.000760 s | 0.000224 s | 48.06x | 43.54x | GPU-CuPy |
| 512² plane, r=5 | 512×512 | 5 | 0.207303 s | 0.002969 s | 0.001822 s | 0.001259 s | 0.000576 s | 113.77x | 94.71x | GPU-CuPy |
| 1024² plane, r=5 | 1024×1024 | 5 | 0.815627 s | 0.006397 s | 0.005714 s | 0.003870 s | 0.001939 s | 142.74x | 139.35x | GPU-CuPy |
| 2048² plane, r=5 | 2048×2048 | 5 | 3.378208 s | 0.046262 s | 0.022722 s | 0.015533 s | 0.007263 s | 148.68x | 141.14x | GPU-CuPy |
| 256² plane, r=10 | 256×256 | 10 | 0.187525 s | 0.002916 s | 0.002231 s | 0.001985 s | 0.000187 s | 84.04x | 80.18x | GPU-CuPy |
| 512² plane, r=10 | 512×512 | 10 | 0.680677 s | 0.004722 s | 0.004075 s | 0.003821 s | 0.000688 s | 167.04x | 164.95x | GPU-CuPy |
| 1024² plane, r=10 | 1024×1024 | 10 | 2.712747 s | 0.034108 s | 0.014667 s | 0.012841 s | 0.001835 s | 184.95x | 178.99x | GPU-CuPy |
| 2048² plane, r=10 | 2048×2048 | 10 | 10.467474 s | 0.070814 s | 0.060270 s | 0.048785 s | 0.004458 s | 173.68x | 172.20x | GPU-CuPy |
| 8×512² plane-wise stack, r=2 | 8×512×512 | 2 | 0.464511 s | 0.028664 s | 0.004873 s | 0.002269 s | 0.003656 s | 95.33x | 91.76x | GPU-CuPy |
| 4×1024² plane-wise stack, r=10 | 4×1024×1024 | 10 | 10.301479 s | 0.058296 s | 0.055703 s | 0.048089 s | 0.004489 s | 184.93x | 183.38x | GPU-CuPy |

## Reviewed crossover screen

- **Radius 0.5:** GPU cleared both gates from 1024² on this machine.
- **Radius 2:** GPU cleared both gates from 512² on this machine.
- **Radius 5:** GPU cleared both gates from 256² on this machine.
- **Radius 10:** GPU cleared both gates from 256² on this machine.

The smallest confident GPU extent is bounded by the tested grid. Do not extrapolate below it or to another machine.

## Gates and interpretation

- The lower bound of the paired 95% bootstrap interval must be at least `1.20x`.
- Median end-to-end saving must exceed `5%` of CPU time or `0.020 s`, whichever is larger.
- GPU end-to-end includes H2D, synchronized compute, and D2H.
- Resident timing models a pipeline that already holds the image on GPU.
- Disk I/O and synthetic image generation are excluded.
- Re-run VIPP's optimizer on the actual workload before persisting a
  backend choice; this report must not be copied across machines.

## Exact coverage

- `axes:explicit-channel`
- `axes:leading-planes`
- `boundary:nearest-clamp`
- `dtype:float32`
- `dtype:uint16`
- `dtype:uint8`
- `fallback:exclude-center`
- `fallback:full-mean`
- `float32:negative-zero`
- `float32:subnormal-sample`
- `float32:subnormal-square`
- `minimum-fraction:0`
- `minimum-fraction:0.2`
- `minimum-fraction:0.8`
- `minimum-fraction:1`
- `plane:tiny`
- `radius:0.5`
- `radius:10`
- `radius:2`
- `radius:5`
- `restore:float32-half-up`
- `sigma-width:0-inclusive`
- `sigma-width:default`

## Matched rejection coverage

- `reject:byte-order`
- `reject:channel-axis`
- `reject:dtype`
- `reject:float32-square-overflow`
- `reject:minimum-fraction`
- `reject:nonfinite`
- `reject:outlier-aware`
- `reject:radius`
- `reject:sigma-width`
