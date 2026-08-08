# GPU Phase 4 Sigma Filter implementation record

- Date: 2026-08-02
- Branch: `codex/gpu-cross-platform-support`
- CPU status: public node and frozen scientific contract implemented
- GPU status: exact reviewed CuPy region is a normal public Custom and Auto
  candidate on the GPU branch

## Outcome and visibility rule

This phase adds `Sigma Filter` (`sigma_filter`) under `Filtering > Smoothing &
Denoising` as a complete CPU/GPU vertical slice. The authoritative CPU node is
public and works in ordinary workflows independently of whether CUDA is
installed. Its GPU implementation is `cupy-sigma-filter-v1`, a lazy fused CuPy
`RawKernel` that participates in the existing policy, facts, planning, memory,
benchmark, residency, cache-provenance, progress, cancellation, fallback, and
cleanup contracts.

VIPP exposes accelerator support by validated region, not by operation name
alone. An exact region that has passed scientific parity, memory, synchronized
progress, cancellation, cleanup, runtime, and real-device evidence is a normal
public `Custom` candidate and may become a reviewed Auto default. Auto uses that
default with no compatible history; accelerated-only history causes one
same-surface CPU exploration run before a later matching run applies the
1.20x/20-ms gate. It never silently benchmarks multiple implementations. Only
unfinished or unvalidated candidates remain
`developer_hidden`. Dtypes, parameters, values, layouts, runtimes, and
platforms outside the reviewed region visibly use CPU; VIPP does not change
authored data or parameters to make them eligible.

The retained pre-0.13.0a3 full-profile RTX 5090 record passed exact parity,
matched rejection, cancellation, cleanup, and timing review. The implementation
entered the immutable v4 policy artifact as `public_auto_candidate`, and current
v7 retains that recorded-host policy unchanged: the exact region appears in
ordinary Custom pipelines and is a reviewed Auto default wherever its current
gates pass.
The completed-run exploration sequence may later select CPU or an accelerated
assignment under Auto's conservative gate. This does not wait for
every CPU dtype or operating system to gain a GPU implementation, and branch
visibility is not a released-package or universal-GPU support claim.

## Scientific basis and attribution

The node is a clean-room implementation of the edge-preserving sigma filter
introduced by Jong-Sen Lee:

- Jong-Sen Lee, “Digital image smoothing and the sigma filter,” *Computer
  Vision, Graphics, and Image Processing* 24(2), 1983,
  [DOI 10.1016/0734-189X(83)90047-6](https://doi.org/10.1016/0734-189X(83)90047-6).
- ImageJ/Fiji
  [Sigma Filter Plus documentation](https://imagej.net/ij/plugins/sigma-filter.html),
  by Michael Schmid, with the outlier-aware variant credited to Tony Collins.
- The published
  [Sigma Filter Plus Java source](https://imagej.net/ij/plugins/download/Sigma_Filter_Plus.java)
  and official bytecode are behavioral references and external-fixture inputs;
  VIPP does not translate that source line by line.

“Fiji-compatible” in this report means compatibility with the frozen public
contract below, plus explicitly named and tested deviations. It does not mean
ROI/mask compatibility: the VIPP version-1 node has no ROI input.

## Frozen CPU scientific contract

### Inputs, parameters, axes, and output

The node accepts a non-empty finite native-endian array with exactly one of
these public dtypes; non-native byte order fails closed:

| Input dtype | Output dtype | Restoration |
|---|---|---|
| `uint8` | `uint8` | Cast computed value to float32, add 0.5, floor, clip to 0–255 |
| `uint16` | `uint16` | Cast computed value to float32, add 0.5, floor, clip to 0–65,535 |
| `float32` | `float32` | Direct float32 restoration |

The public parameters are:

| Parameter | Default | Valid region |
|---|---:|---|
| `radius` | 2.0 | finite 0.5 through 10.0, inclusive |
| `sigma_width` | 2.0 | finite and non-negative |
| `minimum_pixel_fraction` | 0.2 | finite 0 through 1, inclusive |
| `outlier_aware` | `True` | boolean only |
| `channel_axis` | `None` | `None` or a valid integer axis leaving two spatial axes |

`channel_axis=None` follows VIPP's scalar-default convention. The final two
non-channel spatial axes resolve to `YX`. Every channel and every other leading
axis index is processed as an independent YX plane. Thus YX, ZYX, YXC, CYX,
explicit non-trailing channel layouts, and TCZYX all retain their authored
shape and axis order while filtering only within each resolved plane.

The source is immutable and the destination is separate. Newly written values
never feed later neighborhoods. At borders, Y and X coordinates are clamped
independently to the nearest sample. Repeated aliases introduced by clamping
count as separate footprint samples, exactly as their offsets prescribe.

### Circular footprint

The footprint helper applies these documented radius plateaus:

```text
1.5 <= radius < 1.75  -> 1.75
2.5 <= radius < 2.85 -> 2.85
r2 = floor(radius * radius) + 1
```

It then traverses every integer `(dy, dx)` in row-major order whose
`dx² + dy² <= r2`. Radius 2.0 has 21 samples; radius 10.0 has 325. This ordered
offset tuple is shared by CPU and GPU. Exact boundary and `nextafter` tests
lock the discontinuities instead of relying on a visually similar disk.

### Arithmetic and selection

Every neighborhood sample is represented as float32. Its square is computed at
the float32 precision boundary, then both the sample and square are accumulated
in fixed offset order into float64 sums. For footprint size `N`:

```text
mean = sum(samples) / N
variance = sum(float32(sample * sample)) / N - mean * mean
variance = max(variance, +0.0)
lower = center - sigma_width * sqrt(variance)
upper = center + sigma_width * sqrt(variance)
minimum_count = ceil(N * minimum_pixel_fraction)
```

The second ordered pass selects every sample satisfying the inclusive
`lower <= sample <= upper` interval. The interval is centered on the immutable
source pixel, not on the neighborhood mean.

If `selected_count >= minimum_count`, the output is the selected-sample mean.
Otherwise:

- `outlier_aware=False` returns the full-footprint mean; or
- `outlier_aware=True` returns `(full_sum - center) / (N - 1)`.

The outlier-aware fallback removes one logical center sample. It does not remove
all repeated border aliases that happen to refer to the same source coordinate.
Unsigned restoration is half-up, not NumPy ties-to-even.

### Two intentional differences from the published plugin

VIPP records two narrow numerical stabilizations rather than claiming that
they are exact Fiji output:

1. **Exact ceiling.** VIPP uses the mathematical
   `ceil(N * minimum_pixel_fraction)`. The published Java code approximates it
   with `(int)(N * fraction + 0.999999)`, which differs in a narrow boundary
   region.
2. **Deterministic variance clamp.** A population variance is mathematically
   non-negative. VIPP clamps every cancellation-induced negative value to
   positive zero before `sqrt`. The published plugin can form NaN and, at a
   zero minimum fraction, restore that NaN to unsigned zero.

Both choices are shared by CPU and GPU, locked by named tests, and form part of
the versioned VIPP scientific identity.

## Independent Fiji evidence

The frozen fixture `sigma_filter_fiji_reference_v1.json` was not produced by
VIPP's Python implementation or by a second Python oracle. The maintainer tool
downloads and verifies the official plugin source and bytecode plus ImageJ
1.54p, compiles only a small headless Java adapter, invokes the official
`Sigma_Filter_Plus.doFiltering` method, and restores unsigned output through
ImageJ's own `ByteProcessor`/`ShortProcessor` path.

The fixture pins hashes for:

- published plugin source and class;
- ImageJ 1.54p jar;
- the generator script and Java harness;
- every input and external output; and
- Java/ImageJ execution provenance.

Fourteen cases match the externally executed plugin output exactly. They cover
constants, gradients, a hard edge, hot/dead pixels, tiny planes, clamped border
aliases, radius 0.5/2/10 behavior, both fallbacks, half-up restoration, and an
inclusive threshold/next-below pair. Two additional cases freeze the exact-
ceiling and negative-variance differences above and assert that VIPP does
*not* match the external output for those deliberate reasons. ROI and dialog/
stack orchestration are bypassed because they are outside the node contract;
the filtering kernel and ImageJ restoration are not replaced.

## GPU implementation

`cupy-sigma-filter-v1` is provided by
`napari_vipp.core.gpu.cupy_sigma:sigma_filter`. CuPy owns the CUDA runtime,
resident array domain, stream, allocator, and implementation-library identity.
The provider is therefore declared as CuPy, not inaccurately as CuPyX or cuCIM.
All optional CUDA imports occur only when an accelerator call is requested.

For arbitrary leading planes, channel positions, and non-contiguous inputs, the
provider transposes the two resolved YX axes to the end and explicitly creates
one contiguous float32 device workspace. It allocates a distinct typed output,
flattens only the leading plane indices, and restores the original axis order
as a contiguous resident result. It never transfers an image to the host and
never constructs an `N pixels × K footprint samples` tensor.

One output thread scans the ordered footprint twice: the first pass builds the
full mean/variance; the second applies the inclusive interval and fallback.
The kernel is compiled with:

```text
--fmad=false
--prec-div=true
--prec-sqrt=true
```

Fused multiply-add is disabled so GPU accumulation does not silently cross a
CPU rounding boundary. Precise division and square root support the branch-
sensitive interval. CuPy 14.1.1 appends `--ftz=true`; VIPP no longer requests
the contradictory `--ftz=false` option because the duplicate flags can make
NVRTC compilation fail. The kernel converts float32 samples to double from
their bits and constructs subnormal float32 squares/results by bits. That
preserves subnormal signs and magnitudes instead of silently flushing them to
zero and changing a selection or fallback decision.

The provider uses 64-row tiles. This limits radius-10 kernel duration on a
Windows display GPU, permits truthful synchronized progress, and provides an
honest cancellation boundary without pretending to interrupt a running CUDA
kernel.

## Exact public GPU region and parity

The public accelerator region is:

- non-empty native-endian `uint8`, `uint16`, or `float32` input;
- every authored value finite;
- radius 0.5–10 inclusive, including the two canonical plateaus;
- finite non-negative sigma width;
- finite minimum fraction 0–1 inclusive;
- boolean outlier-aware setting;
- `channel_axis=None` or a valid integer leaving two resolved YX axes;
- arbitrary independent leading planes and channel positions;
- contiguous or non-contiguous input, with the explicit copy charged to the
  memory and timing models; and
- for `float32`, complete finite extrema facts and magnitude no greater than
  the float32 square-workspace limit.

Non-native byte order, other unsupported dtypes, empty data, non-finite values,
square-overflow magnitude,
invalid axes, invalid parameters, ROI/mask semantics, and unqualified runtime/
platform combinations retain CPU or the CPU error contract.

`sigma-dtype-parity-v1` requires equal public shape and dtype. `uint8` and
`uint16` outputs are bitwise exact. Float32 output requires equal finite masks,
completely finite values, equal zero masks and signed-zero bits, NRMSE no more
than `2e-6`, and:

```text
max_abs <= 1e-6 + 4 * eps(float32) * max(1, input_peak, CPU_peak)
```

Maximum float32 ULP distance is diagnostic. Separate adversarial fixtures lock
selection membership and the two fallback branches, so the aggregate bound
cannot hide a scientifically different branch. The canonical real-device
admission matrix additionally requires exact outputs for its branch-sensitive
integer and float32 cases before timing begins.

No CPU/GPU cache-equivalence group is declared. Cache and benchmark identities
retain the exact implementation ID/version, runtime/library, parity and other
policy IDs, workload bytes/shape/dtype/parameters, scientific stack, device,
environment, and memory scope.

## Memory, residency, progress, cancellation, and fallback

`cupy-sigma-filter-memory-v1` accounts for resident input and typed output plus:

- a complete contiguous float32 canonical workspace;
- a worst-case complete typed axis-restoration staging buffer;
- the bounded radius-10 table of 325 `(dy, dx)` int32 offsets;
- the float32-validation status value; and
- the standard first-use uncertainty and runtime reserve.

There is no image-sized neighborhood tensor. The runtime can keep the result
resident for a same-domain downstream CuPy/CuPyX/cuCIM segment when the planner
admits zero-copy interoperability; public outputs and caches remain host-only.

CPU progress has two truthful phases: bounded 64-row validation blocks followed
by bounded 64-row calculation blocks. It checks cancellation at block boundaries
and during long footprint scans. GPU progress reports one completed 64-row tile
only after the current CUDA stream is synchronized, then checks cancellation
before the next tile. The final update follows axis restoration and
synchronization.

Success, parity failure, cancellation, error, and benchmark exit use the shared
transactional runtime cleanup contract. A missing package, failed runtime/library
probe, out-of-domain workload, memory rejection, or unsupported platform is a
typed decision before execution where possible. Auto may visibly retry a
classified unavailable/support/OOM segment once on CPU; strict Custom GPU
intent reports the actionable failure. An unclassified provider or scientific
error is not hidden by fallback.

## Workflow and user-facing integration

The CPU operation is registered in the normal node library and therefore uses
the established palette, workflow-v4 serialization, snapshots, generated
Python, batch preservation, export, metadata history, and widget parameter
paths. The node card can display the actual CPU or `GPU · CuPy` badge after an
accepted run. Custom node benchmarking and the whole-pipeline optimizer use
the Sigma-specific parity gate and exact workload/environment identity.

Workflow files persist portable authored compute intent, not resolved hardware,
local benchmark samples, or experimental admission. Existing workflows need no
schema bump for the node. Batch, generated Python/CLI, and export continue to
preserve compute intent while executing through their existing CPU path until
the later durable-execution phase.

## Real-device validation and timing

The canonical evidence runner is `scripts/benchmark_gpu_sigma.py`. Its full
profile has ten exact admission cases, ten matched rejection cases, and
eighteen timed deterministic synthetic workloads. Admission covers every public
dtype; radii 0.5, 2, 5, and 10; sigma-width zero/default; minimum fractions 0,
0.2, 0.8, and 1; both fallbacks; nearest borders; half-up restoration;
negative zero and float32 subnormal samples/squares; leading planes; explicit
channels; and tiny planes. Rejection covers non-native byte order, unsupported
dtype, radius, sigma width, minimum fraction, outlier type, channel axis,
non-finite input, and square overflow.

Every timed workload passed production CPU/GPU bitwise parity before measurement.
GPU end-to-end samples include host-to-device transfer, synchronized resident
compute, device-to-host transfer, and private-scope cleanup. First-process JIT
is reported separately from warm execution; no partial or parity-failing result
is used to support performance admission.

Seven-round machine-local medians from selected cases are:

| Workload | Radius | CPU | GPU end-to-end | GPU resident | E2E speedup | Screen |
|---|---:|---:|---:|---:|---:|:---|
| 256² plane | 0.5 | 0.004669 s | 0.000671 s | 0.000342 s | 6.96x | CPU |
| 512² plane | 0.5 | 0.021025 s | 0.000892 s | 0.000473 s | 23.57x | GPU-CuPy |
| 1024² plane | 0.5 | 0.097682 s | 0.001922 s | 0.001096 s | 50.83x | GPU-CuPy |
| 256² plane | 2 | 0.015812 s | 0.000682 s | 0.000455 s | 23.17x | CPU |
| 512² plane | 2 | 0.057129 s | 0.001034 s | 0.000640 s | 55.23x | GPU-CuPy |
| 2048² plane | 10 | 10.266807 s | 0.060059 s | 0.049125 s | 170.95x | GPU-CuPy |
| 8×512² stack | 2 | 0.449594 s | 0.004802 s | 0.002303 s | 93.62x | GPU-CuPy |
| 4×1024² stack | 10 | 10.174047 s | 0.055786 s | 0.047831 s | 182.38x | GPU-CuPy |

The 256² radius-0.5 and radius-2 cases are deliberately CPU choices despite
large timing ratios: their 4.00-ms and 15.13-ms absolute median savings remained
below the 20-ms material-saving gate. Radius 0.5 first cleared both Auto gates at
512²: its 20.13-ms saving exceeded the material gate, while its paired 95%
speedup lower bound was 19.58x against the 1.20x confidence gate. Radius 2 also
cleared at 512²; radii 5 and 10 cleared at the smallest tested 256² extent. Both
stack cases selected GPU. These are bounded observations over the tested grid,
not permission to extrapolate below it.

The full artifact passed 10/10 exact admission cases and all 18 timed workloads
bit for bit, including float32 signed zero and subnormal arithmetic. All ten
invalid-region cases produced matched CPU/GPU rejections. Synchronized
cancellation passed, and the private allocator ended with zero used and zero
reserved bytes. The source/environment fingerprints are current for the
recorded policy sources. See the
[readable evidence](benchmarks/sigma-filter-cupy-windows-rtx5090.md) and
[machine-readable record](benchmarks/sigma-filter-cupy-windows-rtx5090.json).

These short results guide selection only on the named host. They are not a
portable speed promise or durable optimizer record; VIPP re-evaluates the exact
workload and environment, and CPU remains a correct outcome for small calls.

## Platform and packaging limits

The first executable CUDA policy is the branch's exact native-Windows CPython
3.12 environment with CuPy 14.1.1 and the reviewed NumPy 2.5.1, SciPy 1.18.0,
and scikit-image 0.26.0 CPU reference stack. The canonical Auto/Prefer-GPU host
is an RTX 5090 with the recorded CUDA runtime/driver APIs and compute capability.
Different package versions, runtime, or Python ABI fail closed. Starting in
0.13.0a3, a compatible secondary NVIDIA device can enter explicit Custom
execution or local parity-gated Find-Fastest qualification under the exact
supported software stack; that local result is not portable evidence.

Source-current 0.13.0a3 validation fixed the CuPy 14.1.1 compile failure and
passed Sigma parity on an RTX 4050 Laptop GPU at compute capability 8.9. This
bounded result complements rather than replaces the historical RTX 5090 record.
Native Linux and portable Auto/Prefer-GPU evidence for RTX 40-series Windows
laptops remain named validation targets; WSL2 is secondary evidence. Current
CUDA has no macOS target, so this provider is CPU-only on macOS. The separate M1
Max Metal/MPS/MLX study can add a future provider only after operation-level
scientific and lifecycle gates pass. CuPy and CUDA remain optional: base
installation, plugin discovery, workflow loading, and CPU execution must not
import them.

## Deferred scope

- Fiji ROI/mask behavior, because the version-1 node has no ROI input contract.
- Dtypes beyond `uint8`, `uint16`, and finite square-safe `float32`.
- Broader runtime, driver, GPU, OS, Python, and scientific-stack versions.
- GPU execution in batch, generated Python/CLI, and export surfaces.
- A CPU/GPU cache-equivalence claim.
- Any automatic cast, precision change, footprint approximation, boundary
  change, or alternate high-level cuCIM/CuPyX algorithm.

## Ordered next suggested steps

Connected Components is now complete as Phase 5, including exact SciPy `int32`
label IDs, independent leading-block resets, resident CuPyX execution, lifecycle
evidence, and packaged compute-policy artifact v5. See the
[Phase 5 implementation record](gpu-phase5-connected-components-implementation-report.md).

1. **Measurements** — support ordered labels-plus-intensity inputs and an exact
   typed host-table finalizer preserving schema, order, units, calibration,
   public scalar types, and missing values.
2. **Convert Dtype and inexpensive residency bridges** — accelerate only
   explicit authored operations that preserve their CPU semantics; never
   synthesize a cast to improve a benchmark.
3. **Native Linux and Windows laptop evidence** — collect clean-install,
   parity, memory, cancellation, cleanup, and end-to-end evidence on supported
   Linux hosts and available RTX 40-series Windows laptops.
4. **Apple M1 Max study** — evaluate Metal/MPS/MLX with unified-memory
   accounting and retain CPU fallback unless operation-level gates pass.
5. **cuCIM/Clara feature-complete investigation** — perform the named
   time-boxed packaging/upstream review toward a maintainable complete route.
6. **Batch, generated Python/CLI, and export** — add durable GPU execution and
   provenance only after core interactive coverage and lifecycle contracts are
   stable.
