# GPU Phase 3A Canny and Otsu implementation record

- Date: 2026-07-29
- Branch: `codex/gpu-cross-platform-support`
- Status: exact-mask providers, validated public candidate regions, and
  source-current RTX 5090 admission/performance evidence complete

## Outcome and visibility rule

This phase adds GPU implementations for VIPP's Canny and Otsu nodes without
changing their CPU operations, defaults, workflow parameters, axes, or public
boolean outputs. Both implementations are normal public candidates on the GPU
branch, not developer-only demonstrations.

Provider visibility is region-specific. A region that has passed scientific
parity and its required memory, progress, cancellation, cleanup, and runtime
checks is visible in normal `Custom` pipelines and may be a reviewed Auto
default in that exact admitted region. Auto uses that default with no compatible
history; accelerated-only history causes one same-surface CPU exploration run
before a later matching run applies the 1.20x/20-ms gate. It never silently
benchmarks multiple implementations, so CPU can still be the correct
eligibility or learned performance decision. `developer_hidden` is reserved
for incomplete or unvalidated work. A
dtype, parameter, shape, runtime, or platform outside a public region visibly
resolves or falls back to CPU. VIPP does not hide a validated provider merely
because it cannot cover every CPU call, and it never changes authored data or
parameters to make a call eligible.

Public visibility on this development branch is not a blanket claim that every
GPU, operating system, or released VIPP package is supported. The executable
policy remains authoritative for each exact workload.

## Canny exact-mask provider

The operation is `canny_edges`; its accelerator is
`cupyx-canny-edges-exact-v1`, provided by
`napari_vipp.core.gpu.cupy_canny:canny_edges`. CuPy owns the resident array and
kernel domain. A custom CuPy kernel reproduces SciPy's observable outside-in
correlation order, while CuPyX supplies the remaining compatible primitives.
Optional CUDA imports remain lazy.

The adapter mirrors the CPU operation rather than calling a second high-level
Canny implementation. It preserves:

- explicit conversion of each scalar plane to `float32`;
- VIPP's BT.601 RGB/RGBA reduction when `channel_axis` is authored;
- trailing-YX plane semantics with independent arbitrary leading blocks;
- the constant-boundary Gaussian correction and SciPy-order arithmetic;
- float32 Sobel magnitude construction;
- scikit-image-compatible bilinear non-maximum suppression, including tie and
  mixed float32/float64 interpolation behavior;
- ordered low/high quantiles with quantile thresholds only;
- eight-connected hysteresis; and
- the exact shape-preserving boolean edge mask.

The custom correlation and non-maximum-suppression CUDA kernels preserve the
reviewed accumulation and tie behavior. The initial public region accepts bool,
`uint8`, and `uint16` input. Sigma uses
the CPU operation's canonical non-negative value and is admitted from 0 through
12. Low and high quantiles must be finite, inside `[0, 1]`, and ordered. Empty,
invalid-axis, invalid-quantile, and invalid-rank calls retain the CPU error
contract; `float32`, wider dtypes, non-finite input, or sigma above 12 visibly
use CPU. In particular, CUDA flush-to-zero behavior for subnormal float32
intermediates can alter final edge bits even when every authored value is finite,
so an all-finite float32 GPU claim would not be scientifically exact.

### Why raw cuCIM Canny was not admitted

The final edge mask, not availability of a GPU primitive, is the scientific
contract. Direct cuCIM Canny comparisons disagreed with the CPU reference on
adversarial ramp, checker, and edge/tie structures. Those mismatches make the
raw high-level implementation ineligible for this versioned region even when
ordinary fixtures look similar. The exact CuPy/CuPyX adapter was therefore
selected. cuCIM can be reconsidered under a different implementation ID only if
an adapter passes the same final-mask gate across the complete declared region.

## Otsu exact-mask provider

The operation is `otsu_threshold`; its accelerator is
`cupy-otsu-threshold-exact-v1`, provided by
`napari_vipp.core.gpu.cupy_otsu:otsu_threshold`. Image-sized finite masking,
histogram construction, and final threshold comparison remain on the GPU. Only
the bounded histogram is copied to the host, where the established NumPy
float64 cumulative arithmetic and first-maximum tie break are used. The result
returns to the resident CuPy domain as a boolean mask.

The provider uses a VIPP-owned bounded atomic `uint64` histogram kernel for
exact integer bin indices. CuPy 14.1.1's CUB-backed `bincount` can privatize a
wide histogram per resident sweep block and reserve hundreds of MiB that is not
described by the logical workload. The bounded kernel removes that opaque
workspace and an unnecessary image-sized `uint64`-to-`int64` cast without
changing histogram counts or the CPU threshold finalizer.

This deliberate bounded host finalizer preserves all of the CPU contract:

- scalar boolean input is an identity;
- integer histograms retain exact native levels and offsets, including signed
  and unsigned 64-bit values whose occupied span fits the existing guard;
- the maximum exact integer intensity span is 65,536 levels, with the existing
  explanatory error beyond it;
- floating input retains NumPy-compatible histogram-edge construction,
  finite-only statistics, and 2 through 65,536 bins;
- constant, empty, and all-non-finite cases retain the CPU result/error behavior;
- `Stack histogram` and independent `Slice histogram` scopes are preserved;
- explicit RGB/RGBA input uses the same BT.601 luma conversion; and
- foreground remains the strict finite `image > threshold` mask.

The initial public region includes bool, all NumPy signed/unsigned integer
widths, and float16/32/64. Bool and integer types up through 16 bits need no
extrema scan because their complete dtype domain fits the 65,536-level guard.
Wide 32/64-bit integers require complete native minimum and maximum facts before
device work. Multi-plane Slice scope whose per-plane spans cannot be proved from
global facts visibly remains on CPU. Luma-converted and native floating data
require a valid histogram-bin count. Other value kinds use the explicit
CPU/error path.

A raw cuCIM Otsu threshold scalar was useful feasibility evidence, but scalar
agreement alone cannot establish VIPP's boolean identity, native-level integer
guard, non-finite policy, slice scope, luma conversion, tie behavior, or final
mask. The exact adapter keeps those semantics versioned and reviewable.

## Parity, memory, progress, cancellation, and cleanup

Both implementations use `mask-bitwise-v1`: output shape and boolean dtype must
match and every public mask bit must be identical to the CPU prepared-call
reference before timing evidence is accepted. A close edge map, equivalent
histogram threshold, or similar segmentation is not sufficient.

`cupyx-canny-exact-memory-v1` accounts conservatively for the resident input,
boolean output, concurrent RGB cast/product/luma workspace where present,
Gaussian/Sobel/magnitude arrays,
masks, non-maximum-suppression output, labeling state, and CUDA-kernel overhead.
`cupy-otsu-histogram-memory-v1` accounts for resident input/output, finite and
comparison masks, concurrent luma/float workspaces where present, exact integer
or comparison-binned NumPy-edge histograms up to the declared bound, the
bounded atomic counts array, and transfer/finalizer overhead. Admission
uses the shared discrete-VRAM or unified-memory budget; runtime OOM still follows
the transactional cleanup and visible fallback policy.

Canny reports one milestone per completed scalar plane. Otsu reports one
completed stack histogram or one milestone per slice histogram. GPU milestones
are emitted only after the current stream is synchronized, so progress means
completed work and device failures surface before the bar advances.
Cancellation is cooperative at those scientifically honest boundaries. Success,
parity failure, cancellation, error, and benchmark exit must leave no live or
reserved VIPP-owned device values under the shared runtime cleanup contract.

## Validation and timing evidence

Focused tests cover CPU-mask equality, dtypes, axes/luma, leading blocks,
boundaries, ties, constants, error behavior, progress, cancellation, read-only
input, residency, memory admission, planning, fallback, and lazy optional
imports. Wider deterministic exploratory matrices were also used while fixing
the exact Canny arithmetic and Otsu histogram finalizer. Those validation gates
make the declared regions normal public `Auto`/`Custom` candidates now; they
are not developer-hidden.

The source-current schema-v3
[`canny-otsu-cupy-windows-rtx5090.md`](benchmarks/canny-otsu-cupy-windows-rtx5090.md)
record passed its existing-record validator. All 28 admission cases produced
bitwise-identical CPU/GPU boolean masks. Three-sample machine-local medians were:

| Workload | Operation | CPU | GPU end-to-end | GPU resident | End-to-end speedup |
|---|---|---:|---:|---:|---:|
| Structured synthetic 8x1024x1024 `uint16` | Canny | 0.6812 s | 0.0349 s | 0.0325 s | 19.51x |
| Structured synthetic 8x1024x1024 `uint16` | Otsu | 0.0455 s | 0.0077 s | 0.0023 s | 5.92x |
| Private real-acquisition single-channel ZYX volume | Canny | 0.7450 s | 0.0454 s | 0.0455 s | 16.40x |
| Private real-acquisition single-channel ZYX volume | Otsu | 0.0414 s | 0.0078 s | 0.0024 s | 5.28x |

On the synthetic stack, Canny used 72,516,608 private-pool bytes within a
157,286,400-byte admitted peak. Otsu used 84,377,600 bytes within a
200,048,642-byte admitted peak. Both providers also passed cooperative
cancellation and drained their private allocator to zero used and zero reserved
bytes. The private source's path, filename, pixels, and content-derived identity
are absent from the artifact.

These are short descriptive results from one native-Windows RTX 5090 host, not
portable performance guarantees or durable optimizer choices. Auto and the
pipeline optimizer still evaluate the user's exact workload and environment;
CPU remains a successful outcome when it is faster or evidence is inconclusive.

The schema-v3 evidence record includes strict source/environment fingerprint
integrity checks and a closed private-source metadata shape, as well as final mask
parity, synchronized resident and transfer-inclusive timings, memory-model
admission versus observed peak, cancellation/cleanup results, and both realistic
large stacks and representative acquisition data without publishing private
paths, filenames, pixels, or content-derived identities. Node and pipeline
optimization retain exact-workload parity-before-timing, and Auto remains
evidence-driven for the actual workload/runtime; CPU is the normal decision when
benefit evidence is absent or inconclusive.

## Platform and packaging limits

The initial runtime is CUDA through CuPy on the branch's exact reviewed
native-Windows gate: CUDA runtime API 13.2 (`13020`), driver API 13.3 (`13030`),
and an NVIDIA GeForce RTX 5090 with compute capability 12.0. CUDA 12 remains a
qualification-only track outside public admission. Exact comparison-defined
admission also requires the reviewed CPU reference versions: NumPy 2.5.1, SciPy
1.18.0, and scikit-image 0.26.0. Missing or changed scientific-stack provenance
produces a typed visible CPU decision before CUDA probing. CPU-only plugin
discovery and execution remain import-safe.
Supported native Linux, secondary RTX 40-series Windows laptops, and clean
environment recreation still require their named evidence runs. WSL2 remains
secondary evidence. NVIDIA CUDA has no macOS target, so macOS remains CPU-only
for these providers while the separate M1 Max Metal/MPS/MLX study is pending.

The raw cuCIM Canny rejection is a scientific-parity result, not merely a
Windows packaging limitation. The broader cuCIM/Clara feature-complete
investigation remains a separate near-term task and should not be reduced to a
permanent skimage-only fork.

## Ordered next suggested steps

1. **Connected components** — preserve CPU connectivity, independent leading
   blocks, `int32` output, and exact deterministic label numbering; include any
   required canonicalizer in parity, memory, and timing.
2. **Measurements** — support ordered labels-plus-intensity inputs and an exact
   typed host-table finalizer with schema, row/column order, units, calibration,
   public scalar types, and missing-value parity.
3. **Convert Dtype and inexpensive residency bridges** — accelerate only
   explicit authored operations that preserve their CPU semantics; never
   synthesize a cast or bridge to improve a benchmark.
4. **Native Linux and Windows laptop validation** — collect clean-install,
   parity, memory, cancellation, cleanup, and end-to-end evidence on supported
   Linux hosts and available RTX 40-series Windows laptops.
5. **Apple M1 Max study** — evaluate Metal/MPS/MLX providers with unified-memory
   accounting and retain CPU fallback unless operation-level gates pass.
6. **cuCIM/Clara feature-complete investigation** — conduct the named
   time-boxed packaging/upstream review toward a maintainable feature-complete
   integration.
7. **Batch, generated Python/CLI, and export** — add durable GPU execution and
   provenance after the core interactive operation and lifecycle contracts are
   stable.
