# SimpleITK CPU qualification follow-on — 2026-09-22

Status: machine-local benchmark completed; **median is the only production
replacement justified by this pass**. Other candidates remain benchmark-only.
This supplements the broad initial Gaussian/RL/median/components comparison,
using the actual new median adapter and a focused second set of candidates.

## Decision

- **Median:** use the qualified adapter with the conservative minimum XY plane
  area of 65,536 pixels. Across the 26 timed eligible cases the actual VIPP
  runtime was **2.02–6.41× faster** than the unchanged SciPy reference, with
  bitwise-identical checked outputs. Small 128 × 128 planes keep SciPy:
  SimpleITK was slower for width 3, while larger widths could save only a few
  milliseconds or tens of milliseconds in this bounded screen. The simple
  conservative gate avoids adding many machine-specific dispatch rules.
- **Background estimation/subtraction:** unchanged. These operations use
  scikit-image rolling-ball estimation with optional **mean**, not median,
  presmoothing. SimpleITK exposes no matching rolling-ball entry point; a flat
  morphological opening is not an equivalent algorithm. Even the separately
  checked SimpleITK mean replacement changes floating-point results, so it was
  rejected before timing or production integration.
- **Binary opening/closing:** unchanged. Odd XY box widths and zero boundaries
  can be reproduced exactly by composing SimpleITK primitives, but performance
  is mixed: width 5 is slower, width 11 in 2D is only 1.25–1.35× faster, and the
  tested slice-wise 3D implementation is 4.2–11.7× slower. That does not justify
  another conditional backend for these nodes.
- **Euclidean distance transform:** retain the current implementation for now.
  The benchmark candidate is promising: **4.37× faster at 2048 × 2048** and
  **3.88× at 16 × 256 × 256**, with bitwise-identical checked outputs. The
  512 × 512 case saves only 3.91 ms (1.29×). A production change needs its own
  full input/axes/block, empty/degenerate, range, metadata, provenance, memory,
  lifecycle and downstream-consumer qualification. This result is not that
  qualification and does not authorize a silent method substitution.
- Gaussian, RL/RL-TV and connected components are intentionally unchanged.

See the [implementation contract](../simpleitk-cpu-qualification.md), the
[reproducible harness](../../scripts/benchmark_simpleitk_cpu.py), and the
[raw evidence](simpleitk-cpu-qualification-2026-09-22.json). The raw evidence retains each timed repetition,
warmup/output check, dependency version and exact source hash.
The interpreter's private directory was removed from the shareable JSON;
measurements and recorded source hashes are unchanged. The harness now records
only the interpreter filename, so later reports do not disclose home paths.

## Environment and interpretation

- Windows 11; Intel Core i5-10500 (6 cores / 12 logical processors); CPU only.
- Python 3.12.9;
  SimpleITK 2.5.6;
  NumPy 2.5.1;
  SciPy 1.18.0;
  scikit-image 0.26.0.
- SimpleITK uses 12 threads/work units for this screen. The production median
  adapter independently caps per-instance threads/work units at 12.
- Started 2026-09-22T08:08:03.879243+00:00; finished 2026-09-22T08:10:11.966248+00:00.
- 50 timed cases: 35 median, 12 opening/closing and 3 distance-transform cases.
  Each backend has one warmup plus **three** timed repetitions in seeded,
  shuffled backend order; tables show the median.
- All timings include NumPy↔SimpleITK conversion, padding, actual computation
  and output copying. Median runtime also includes its production eligibility
  scan/dispatch. Imports, image generation, garbage collection, output
  comparisons and input hashes are outside the warm timed region. This is not
  GUI, graph scheduling or whole-workflow latency.
- CPU jobs were run serially with other agent tests held during timing.
  This is one normal desktop machine, not a locked-down benchmark host.
  Noise/ordering remains visible: the largest max/min ratio among three warm
  repetitions is 1.77. Precise multipliers are not portable guarantees.
- Median fixtures are deterministic random uint8/uint16 intensities and finite
  positive float32 intensities. Mask fixtures have blurred sparse seeds,
  irregular regions, holes and edge-touching objects. These are synthetic
  correctness/performance fixtures, not representative of every microscopy
  image, mask topology or noise distribution.
- Conservative preflight requires approximately 48 bytes per input voxel plus
  512 MiB spare RAM. No peak-memory speed comparison is claimed. The median
  adapter's 64 MiB chunk bound covers padded input, not all ITK/copy buffers.
- The production median eligibility contract also admits native uint32,
  int8/int16/int32 and float64 under its restrictions; those have separate
  correctness tests but **no timing claims from this three-dtype sweep**.
  Correctness coverage extends through canonical odd size 51; this timing
  sweep covers widths 3, 5, 11 and selected width 21 only.
- All 50 warmup output comparisons match byte-for-byte and preserve read-only
  input buffers. Timed outputs are discarded after input-integrity checks;
  output equality is checked at warmup, not on every timed repetition.

### First call is not the warm timing

In a fresh child process, the first actual 512 × 512 uint16 width-5 median
operation took **123.76 ms**, including its lazy SimpleITK import and complete
operation. The immediately following call took **22.58 ms**. Python/VIPP
startup and input generation are excluded from both. SimpleITK was absent
from the module cache before the first call and present afterward. This
startup cost is not hidden in the warm speed-up claims.

## Median: all timed cases

Values are milliseconds. `Forced ITK` bypasses only the dispatch performance
gate; it still uses the actual adapter, including symmetric halo/copies.
`VIPP runtime` invokes the public operation with the gate. Speed-up compares
SciPy to actual VIPP runtime, so fallback rows appropriately stay near 1×.

| Shape | Dtype | Width | SciPy ms | Forced ITK ms | VIPP runtime ms | Runtime speed-up | Selected backend |
|---|---|---:|---:|---:|---:|---:|---|
| 128 × 128 | uint8 | 3 | 2.05 | 3.75 | 2.04 | 1.00× | scipy |
| 128 × 128 | uint8 | 5 | 5.28 | 3.65 | 5.30 | 1.00× | scipy |
| 128 × 128 | uint8 | 11 | 20.54 | 5.82 | 20.94 | 0.98× | scipy |
| 128 × 128 | uint16 | 3 | 2.04 | 3.50 | 2.13 | 0.96× | scipy |
| 128 × 128 | uint16 | 5 | 5.32 | 3.93 | 5.31 | 1.00× | scipy |
| 128 × 128 | uint16 | 11 | 20.32 | 6.24 | 20.31 | 1.00× | scipy |
| 128 × 128 | float32 | 3 | 2.12 | 3.73 | 2.17 | 0.98× | scipy |
| 128 × 128 | float32 | 5 | 5.44 | 3.97 | 5.37 | 1.01× | scipy |
| 128 × 128 | float32 | 11 | 20.01 | 6.54 | 20.07 | 1.00× | scipy |
| 256 × 256 | uint8 | 3 | 7.85 | 4.27 | 3.88 | 2.02× | simpleitk |
| 256 × 256 | uint8 | 5 | 20.68 | 5.27 | 5.32 | 3.89× | simpleitk |
| 256 × 256 | uint8 | 11 | 79.24 | 17.31 | 14.94 | 5.30× | simpleitk |
| 256 × 256 | uint16 | 3 | 7.96 | 3.76 | 3.57 | 2.23× | simpleitk |
| 256 × 256 | uint16 | 5 | 21.00 | 6.59 | 6.95 | 3.02× | simpleitk |
| 256 × 256 | uint16 | 11 | 80.10 | 17.72 | 16.23 | 4.94× | simpleitk |
| 256 × 256 | float32 | 3 | 8.27 | 3.83 | 3.67 | 2.25× | simpleitk |
| 256 × 256 | float32 | 5 | 20.95 | 8.25 | 8.32 | 2.52× | simpleitk |
| 256 × 256 | float32 | 11 | 80.09 | 18.33 | 22.54 | 3.55× | simpleitk |
| 512 × 512 | uint8 | 3 | 30.95 | 6.83 | 6.74 | 4.60× | simpleitk |
| 512 × 512 | uint8 | 5 | 82.56 | 15.70 | 15.29 | 5.40× | simpleitk |
| 512 × 512 | uint8 | 11 | 319.29 | 67.11 | 57.79 | 5.53× | simpleitk |
| 512 × 512 | uint16 | 3 | 31.28 | 6.22 | 5.52 | 5.67× | simpleitk |
| 512 × 512 | uint16 | 5 | 82.55 | 23.24 | 24.06 | 3.43× | simpleitk |
| 512 × 512 | uint16 | 11 | 320.01 | 64.33 | 59.28 | 5.40× | simpleitk |
| 512 × 512 | float32 | 3 | 32.35 | 6.93 | 8.63 | 3.75× | simpleitk |
| 512 × 512 | float32 | 5 | 83.60 | 25.92 | 26.31 | 3.18× | simpleitk |
| 512 × 512 | float32 | 11 | 327.64 | 66.67 | 65.08 | 5.03× | simpleitk |
| 512 × 512 | uint16 | 21 | 1074.54 | 189.62 | 167.56 | 6.41× | simpleitk |
| 512 × 512 | float32 | 21 | 1079.27 | 205.73 | 194.77 | 5.54× | simpleitk |
| 2048 × 2048 | uint16 | 5 | 1328.95 | 291.65 | 295.15 | 4.50× | simpleitk |
| 2048 × 2048 | float32 | 5 | 1344.77 | 378.24 | 377.32 | 3.56× | simpleitk |
| 16 × 256 × 256 | uint16 | 5 | 331.54 | 100.90 | 63.72 | 5.20× | simpleitk |
| 16 × 256 × 256 | float32 | 5 | 335.56 | 100.90 | 121.01 | 2.77× | simpleitk |
| 32 × 512 × 512 | uint16 | 5 | 2677.02 | 674.67 | 636.82 | 4.20× | simpleitk |
| 32 × 512 × 512 | float32 | 5 | 2683.08 | 752.29 | 759.41 | 3.53× | simpleitk |

The exact replacement retains the original slice-wise XY semantics. It does
not mix Z, time or channel planes, and its symmetric halo reproduces SciPy's
half-sample-reflect boundary. This is not a switch to SimpleITK's default
border extension. Non-admitted inputs retain the authoritative SciPy path.

## Additional candidates: all timed cases

All masks are Boolean. Morphology remains slice-wise XY, even for a ZYX input.
Distance is genuinely 3D for the ZYX case and uses unit voxel spacing to match
VIPP's existing operation. A multiplier below 1× means SimpleITK is slower.

| Operation | Shape | Box width | VIPP reference ms | ITK candidate ms | Candidate speed-up |
|---|---|---:|---:|---:|---:|
| opening | 512 × 512 | 5 | 6.76 | 18.90 | 0.36× |
| opening | 512 × 512 | 11 | 22.55 | 18.11 | 1.25× |
| closing | 512 × 512 | 5 | 6.73 | 19.03 | 0.35× |
| closing | 512 × 512 | 11 | 22.41 | 17.77 | 1.26× |
| distance | 512 × 512 | — | 17.35 | 13.44 | 1.29× |
| opening | 2048 × 2048 | 5 | 105.92 | 290.58 | 0.36× |
| opening | 2048 × 2048 | 11 | 358.26 | 267.86 | 1.34× |
| closing | 2048 × 2048 | 5 | 107.13 | 279.38 | 0.38× |
| closing | 2048 × 2048 | 11 | 349.56 | 259.11 | 1.35× |
| distance | 2048 × 2048 | — | 421.21 | 96.32 | 4.37× |
| opening | 16 × 256 × 256 | 5 | 26.96 | 316.09 | 0.09× |
| opening | 16 × 256 × 256 | 11 | 89.31 | 375.08 | 0.24× |
| closing | 16 × 256 × 256 | 5 | 27.38 | 309.07 | 0.09× |
| closing | 16 × 256 × 256 | 11 | 88.65 | 368.97 | 0.24× |
| distance | 16 × 256 × 256 | — | 140.47 | 36.22 | 3.88× |

### Why these candidates are semantically comparable

**Opening/closing:** SimpleITK's default composite-filter border handling was
not substituted. The candidate explicitly composes dilation and erosion on
the original image bounds, with background outside the image on **both**
stages. Its box radii are [r, r] or [r, r, 0]. Even widths are rejected: SciPy's
even-width footprint origin is not an odd-radius conversion.

**Distance:** directly applying SignedMaurer to VIPP's foreground would put
zero on foreground contour voxels and give a different result. The tested
candidate instead applies it to the **inverted** Boolean image and retains
distances only at original foreground voxels. This yields distances to
background voxel centres. Squared-distance and physical-spacing options are
explicitly disabled; output remains float32. All-foreground/empty cases
explicitly retain SciPy's existing behavior rather than silently introducing
a different border or infinity convention.

105 additional small checks cover 2D/3D empty-foreground, all-foreground,
checkerboard, random and edge-touching masks, including singleton dimensions
and kernels larger than a plane. All matched byte-for-byte, retained input
buffers and returned detached outputs. Harness tests separately exercise the
same conventions; this bounded subset does not qualify every possible
production input or consumer.

## Background presmoothing: equivalence rejected

The nearest-border 3 × 3 (or 3 × 3 × 3) SimpleITK mean filter has matching
support and boundary intent, but its floating-point accumulation differs
from SciPy's separable uniform filter. These are small differences, not proof
that either algorithm is scientifically wrong; they simply fail the exact
replacement requirement. Input values span 0–65,535.

| Shape | Dtype | Changed output values | Maximum absolute difference |
|---|---|---:|---:|
| 13 × 17 | float32 | 20 | 3.90625000e-3 |
| 13 × 17 | float64 | 170 | 2.18278728e-11 |
| 5 × 13 × 17 | float32 | 169 | 3.90625000e-3 |
| 5 × 13 × 17 | float64 | 829 | 2.91038305e-11 |

There was no need to benchmark this rejected substage broadly. Integer
restoration could conceal some differences for some inputs, but this does
not establish equality for the node's floating-point or general inputs.
Background subtraction also depends on the subsequent rolling-ball result,
so median speed-ups cannot be extrapolated to either background node.

## Reproduction

Use an environment containing the declared package dependencies and run:

```powershell
python scripts/benchmark_simpleitk_cpu.py --output /path/to/new-evidence.json
```

For an isolated dependency target, add `--dependencies /path/to/dependencies`.
Use `--profile smoke` for a short run; `--qualify-only --skip-cold-start` runs
only small correctness checks. `--resume` requires the same source hashes and
settings and preserves previously completed/failed records. Use a new output
name for a changed implementation or a fresh repetition; existing evidence
is never overwritten silently. An append-only `.events.jsonl` sibling logs
activity, and atomic JSON checkpoints retain completed cases. A case's
in-progress timing messages remain in the event log if execution is stopped.

No GPU results, installation build, registration method, portable optimizer
record or broad replacement decision is implied by this report.

## Provenance

Base Git commit: `39edd4492cdd78ea73134b8a441361be945f9449`. The adapter was under
development on this worktree; the following hashes identify the exact source
used by this run rather than merely the parent commit:

- `scripts/benchmark_simpleitk_cpu.py`: `e5263eca1b1cf31bc3789be032f08620ff99d3cb7b9acc580fa3f54b485984b3`
- `src/napari_vipp/core/operations.py`: `532beae9585887a3d9795375f0a708a40aba24d6605bb4b0d6b6d662c4e9cb94`
- `src/napari_vipp/core/simpleitk_filters.py`: `3dbeb710ca5d847106420a6d7093e4b1d75734f5aece7406d4f76d264823720c`

Additional inspector/runtime provenance work after this measurement does not
change what was timed. If the adapter's behavior changes, rerun with a new
evidence name instead of attributing these timings to the changed source.

## Algorithm references

- [SciPy median filter and reflect boundaries](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.median_filter.html)
- [SciPy binary opening](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.binary_opening.html)
- [SimpleITK binary erosion and boundary controls](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1BinaryErodeImageFilter.html)
- [SciPy Euclidean distance transform](https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.distance_transform_edt.html)
- [SimpleITK SignedMaurer distance filter](https://simpleitk.org/doxygen/latest/html/classitk_1_1simple_1_1SignedMaurerDistanceMapImageFilter.html)
- [scikit-image rolling-ball background estimation](https://scikit-image.org/docs/stable/api/skimage.restoration.html#skimage.restoration.rolling_ball)
