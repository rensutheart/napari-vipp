# Representative real-acquisition ND2 GPU benchmark

Date: 2026-07-28
Status: machine-local Phase 1 evidence; not a portable performance promise

## Source and benchmark selection

The source was a representative local two-channel ND2 acquisition. It remains
outside the repository and is not redistributed. This report deliberately
omits its filename, workstation path, project identifiers, and content digest;
it does not contain image pixels.

Direct `nd2` 0.11.3 metadata described one `uint16` acquisition with actual
axis order `TZCYX` and shape `19 x 9 x 2 x 1150 x 822`: 19 time points, 9 Z
planes, 2 channels, and 1150 x 822 pixels per plane. The channels were `488`
(488 nm excitation, 515 nm emission) and `561` (561 nm excitation, 620 nm
emission). X/Y sampling was 0.11 micrometers and Z spacing was 0.2
micrometers. The decoded array contains 646,585,200 bytes; the source file was
648,404,992 bytes.

The production-profile comparison used the full-resolution central plane at
`T=9, Z=4` from each channel. This is real acquisition data, not a crop or
synthetic fixture. It avoids presenting a repeated 647 MB CPU run as routine
local benchmarking: one default CPU rolling-ball call already took about 9.6
seconds per plane. Full-stack figures below are explicitly projections, not
benchmark records for another workload shape.

The code revision was `c3e96daff5788a359e00df5ed0d012b34efe561e` on
`codex/gpu-cross-platform-support`. The exact admitted environment fingerprint
was `893868c40b0d305a035b44e9d346caafd7dc6946b6e9ed9f5c4aa3271a25ef69`:
native Windows, CPython 3.12, CuPy/CuPyX 14.1.1, cuCIM 26.6.0/26.06.00, CUDA
runtime 13.2, driver API 13.3, and an NVIDIA GeForce RTX 5090.

## Method

The registered production-node adapter benchmarked these UI defaults:

- Subtract Background: radius 50 px, dark background, smoothing enabled,
  negative clipping enabled, and `2D YX` processing;
- Median Filter: 5 x 5; and
- Gaussian Blur: sigma 1.2.

Every admitted GPU comparison performed scientific parity before warm timing,
one untimed warmup, randomized paired CPU/GPU rounds, 2,000 paired-bootstrap
resamples, and a 95% confidence bound. Adaptive stability checks reached 21
paired rounds for every CPU/GPU comparison. Timings include a fresh detached
host input, H2D transfer, synchronized GPU work, D2H transfer, host result, and
private allocator cleanup. ND2 disk I/O is excluded. Every GPU result passed
its operation's Phase 1 parity policy, and every private runtime pool returned
to zero live and reserved bytes.

The paired median speedup is `median(CPU_i / GPU_i)` over randomized paired
rounds. It is intentionally not the ratio of the independently rounded median
columns.

Native `uint16` Gaussian is an explicit CPU region in the Phase 1 policy. Its
CPU value below is the median of seven warm calls; no unsupported GPU result
was timed or used to expand the admitted scientific region.

## Native uint16 results and Auto choices

| Node | Channel | CPU median | GPU end-to-end median | Paired median speedup | Lower 95% bound | Auto choice |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Subtract Background | 488 | 9,594.934 ms | 96.597 ms | 99.130x | 93.571x | cuCIM GPU |
| Subtract Background | 561 | 9,584.920 ms | 79.910 ms | 120.920x | 86.718x | cuCIM GPU |
| Median Filter | 488 | 292.046 ms | 4.353 ms | 67.130x | 52.511x | CuPyX GPU |
| Median Filter | 561 | 293.292 ms | 5.709 ms | 52.106x | 41.920x | CuPyX GPU |
| Gaussian Blur | 488 | 11.086 ms | not admitted | n/a | n/a | CPU |
| Gaussian Blur | 561 | 10.764 ms | not admitted | n/a | n/a | CPU |

The first cuCIM background call included about 4.15 seconds of cold provider
and kernel initialization. Its steady-state end-to-end median was 96.6 ms;
later cold calls in the initialized process were much smaller. Interactive UI
guidance should distinguish first-use latency from steady-state Auto evidence.

For a native-`uint16` chain of Subtract Background -> Gaussian Blur -> Median
Filter, the measured per-node selection is therefore:

`cuCIM GPU -> CPU -> CuPyX GPU`

Summing the independently measured end-to-end node medians gives about 112 ms
per 488 plane and 96 ms per 561 plane, compared with about 9.90 seconds per
plane for an all-CPU chain. This sum is conservative about GPU transfers but is
not a measurement of the resident graph executor as a single transaction.

## Explicit float32 Gaussian comparison

Converting the same planes explicitly to `float32` creates a different admitted
workload. The conversion is exact for the source's 12-bit integer values but is
not implicit VIPP behavior.

| Channel | CPU median | GPU end-to-end median | Paired median speedup | Lower 95% bound | Auto choice |
| --- | ---: | ---: | ---: | ---: | --- |
| 488 | 11.691 ms | 7.812 ms | 1.542x | 1.263x | CPU |
| 561 | 12.252 ms | 8.302 ms | 1.443x | 1.321x | CPU |

Parity and confidence passed, but the absolute saving was only about 4 ms per
plane. That is below the local 10 ms noise floor, so Auto correctly retained
CPU. A larger float32 stack requires its own exact workload evidence.

## Full-acquisition projection

The acquisition contains 342 channel planes. A strictly linear projection of
the measured plane medians gives approximately 56.4 minutes for the example
three-node chain entirely on CPU and 35.6 seconds with the measured selective
choices. This is useful only as an order-of-magnitude planning estimate. It
does not account for full-array batching, resident graph segments, disk I/O,
thermal behavior, or memory pressure, and it must not be persisted as the Auto
record for the full acquisition.

## ND2 metadata finding

At the benchmarked revision, VIPP reported this array as `TCZYX` even though
the direct reader's ordered sizes and array layout were `TZCYX`. `nd2` exposes
`ND2File.sizes` as a read-only mapping proxy, while VIPP's ND2 axis helper
accepted only a concrete `dict` and therefore used its five-dimensional
fallback order. The benchmark bypassed the faulty metadata with explicit
`data[t, z, c, y, x]` indexing.

The defect is mostly invisible to operations that use only the trailing Y/X
axes, but it is unsafe for channel-aware or 3D processing because it swaps the
semantic Z and channel positions. It must be fixed and regression-tested before
this acquisition is used as 3D metadata evidence.
