# CuPy Sigma Filter admission and performance evidence

- Generated: `2026-08-02T01:00:30.982896+00:00`
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
| 256² plane, r=0.5 | 256×256 | 0.5 | 0.004669 s | 0.006500 s | 0.000671 s | 0.000342 s | 0.000157 s | 6.96x | 6.05x | CPU |
| 512² plane, r=0.5 | 512×512 | 0.5 | 0.021025 s | 0.001587 s | 0.000892 s | 0.000473 s | 0.000299 s | 23.57x | 19.58x | GPU-CuPy |
| 1024² plane, r=0.5 | 1024×1024 | 0.5 | 0.097682 s | 0.019162 s | 0.001922 s | 0.001096 s | 0.001098 s | 50.83x | 47.86x | GPU-CuPy |
| 2048² plane, r=0.5 | 2048×2048 | 0.5 | 0.359067 s | 0.029519 s | 0.005994 s | 0.001462 s | 0.003706 s | 59.91x | 55.74x | GPU-CuPy |
| 256² plane, r=2 | 256×256 | 2 | 0.015812 s | 0.001349 s | 0.000682 s | 0.000455 s | 0.000178 s | 23.17x | 18.76x | CPU |
| 512² plane, r=2 | 512×512 | 2 | 0.057129 s | 0.001888 s | 0.001034 s | 0.000640 s | 0.000276 s | 55.23x | 50.98x | GPU-CuPy |
| 1024² plane, r=2 | 1024×1024 | 2 | 0.239458 s | 0.003140 s | 0.002395 s | 0.001353 s | 0.001317 s | 100.00x | 94.28x | GPU-CuPy |
| 2048² plane, r=2 | 2048×2048 | 2 | 0.921119 s | 0.010019 s | 0.011100 s | 0.003822 s | 0.007209 s | 82.99x | 81.59x | GPU-CuPy |
| 256² plane, r=5 | 256×256 | 5 | 0.054851 s | 0.002033 s | 0.001238 s | 0.000759 s | 0.000242 s | 44.30x | 40.27x | GPU-CuPy |
| 512² plane, r=5 | 512×512 | 5 | 0.198131 s | 0.002548 s | 0.001852 s | 0.001277 s | 0.000662 s | 106.98x | 91.81x | GPU-CuPy |
| 1024² plane, r=5 | 1024×1024 | 5 | 0.781809 s | 0.007153 s | 0.005735 s | 0.003745 s | 0.001857 s | 136.33x | 124.90x | GPU-CuPy |
| 2048² plane, r=5 | 2048×2048 | 5 | 3.215362 s | 0.023592 s | 0.022586 s | 0.015306 s | 0.004082 s | 142.36x | 138.33x | GPU-CuPy |
| 256² plane, r=10 | 256×256 | 10 | 0.191017 s | 0.002884 s | 0.002407 s | 0.001890 s | 0.000303 s | 79.37x | 73.57x | GPU-CuPy |
| 512² plane, r=10 | 512×512 | 10 | 0.669202 s | 0.004817 s | 0.004109 s | 0.003491 s | 0.000581 s | 162.84x | 151.90x | GPU-CuPy |
| 1024² plane, r=10 | 1024×1024 | 10 | 2.556832 s | 0.014987 s | 0.014814 s | 0.012756 s | 0.002017 s | 172.59x | 173.05x | GPU-CuPy |
| 2048² plane, r=10 | 2048×2048 | 10 | 10.266807 s | 0.059856 s | 0.060059 s | 0.049125 s | 0.003679 s | 170.95x | 168.20x | GPU-CuPy |
| 8×512² plane-wise stack, r=2 | 8×512×512 | 2 | 0.449594 s | 0.005439 s | 0.004802 s | 0.002303 s | 0.003851 s | 93.62x | 86.65x | GPU-CuPy |
| 4×1024² plane-wise stack, r=10 | 4×1024×1024 | 10 | 10.174047 s | 0.058068 s | 0.055786 s | 0.047831 s | 0.003830 s | 182.38x | 180.78x | GPU-CuPy |

## Reviewed crossover screen

- **Radius 0.5:** GPU cleared both gates from 512² on this machine.
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
