# Production GPU implementation plan

Date: 2026-08-04
Product-direction revision: 2026-08-04
Status: Phase 1 is implemented headlessly on
`codex/gpu-cross-platform-support`; the Pass 4 application slice now includes
the four CPU/Auto/Prefer-GPU/Custom policies, workflow-v4 compute intent,
setup/memory diagnostics, selected-node benchmark review, and a Custom-only
review-first whole-pipeline optimizer.
Phase 2B adds ordinary CuPy/CuPyX Richardson-Lucy,
ordered-multi-input/single-output exact benchmarking, and a process-wide
per-device accelerator lease. Phase 2C adds CuPy/CuPyX
Richardson-Lucy TV with separate lambda-zero and positive-TV scientific
profiles. Phase 3A adds exact-mask CuPy/CuPyX Canny and CuPy Otsu. The validated
regions for all of these providers are normal public
Auto/Prefer-GPU/Custom candidates
on this branch; unsupported regions visibly use CPU. Phase 4 adds the public
CPU Sigma Filter and a clean-room CuPy RawKernel provider whose exact reviewed
region is also a normal public Auto/Prefer-GPU/Custom candidate. Its source-current
RTX 5090 record passed 10 exact admission cases, 10 matched rejections, 18
bitwise-exact timed workloads, cancellation, and cleanup. Canonical Canny/Otsu
numerical evidence for this source revision records 28/28 exact admission cases,
memory/lifecycle proof, and separate synthetic/real-acquisition timings.
Phase 5 adds exact CuPyX Connected Components for boolean 2D/3D masks. Its
source-current record passed 16/16 exact native-`int32` admission cases,
deterministic leading-block resets, synchronized block-boundary cancellation,
cleanup, and conservative memory coverage. The strict v5 packaged policy
artifact appends this public candidate without changing historical v1-v4 bytes.
Phase 6 adds public cuCIM candidates for the basic schemas of Measure Objects
and Measure Objects + Intensity. Its source-current RTX 5090 record passed all
11 admission cases, 11 matched rejections, two lifecycle cases, and zero
transaction-pool residue; the public typed table is finalized only after the
mandatory host boundary. The strict v6 policy artifact appends these candidates
without changing historical v1-v5 bytes.
`developer_hidden` is reserved for unfinished or
unvalidated work, while named release/platform gates remain region-specific.
Cross-platform review: 2026-07-15
cuCIM native-Windows evidence update: 2026-07-16
cuCIM Windows port-plan update: 2026-07-16

## Purpose and fixed constraints

This plan converts the GPU spike into reviewable production work without
changing current CPU results by accident. It is grounded in the current
`PrototypePipeline`, detached `PipelineRunRequest`, host-only interactive cache,
batch manifest, workflow v4, and generated-Python contracts.

The following constraints and approved product directions are non-negotiable:

- VIPP's base installation and CPU execution are supported on Windows, macOS,
  and Linux. NVIDIA GPU execution is supported only on native Windows and
  supported Linux distributions. macOS is CPU-only for the NVIDIA-only phase.
- The main toolbar exposes four execution modes: `CPU`, `Auto`, `Prefer GPU`,
  and `Custom`. New interactive sessions default
  to `Auto`. Prefer GPU considers every reviewed public provider, including
  `public_custom`, without applying the CPU-versus-GPU benefit gate. It does
  not bypass scientific, dtype, parameter, shape, dependency, environment, or
  memory gates; never synthesizes a cast; requires visible fallback; and gives
  unsupported nodes an explained ordinary CPU decision. Complete comparable
  GPU evidence selects the fastest GPU, otherwise stable implementation-ID
  order supplies a deterministic choice. Developer-hidden providers remain
  excluded unless explicit experimental admission is enabled.
  `Custom` exposes concise authored per-node preferences: `Auto for this node`,
  `CPU`, and one `GPU · <library>` choice per validated library. `Best
  GPU` appears only when at least two libraries compete. Exact implementation
  pins remain available to advanced/headless workflows and a loaded pin stays
  visibly represented until the user replaces it. Custom also exposes the
  node/pipeline benchmark actions described below. Per-node preferences remain
  serialized but dormant in CPU, Auto, and Prefer GPU. Older workflows and
  callers remain CPU until they explicitly adopt the new execution contract.
- CuPy owns the first CUDA runtime/array substrate. CuPyX and cuCIM are
  implementation libraries that may share that substrate only when zero-copy
  array, stream, device, allocator, and lifetime interoperability is proved.
  cuCIM is admitted operation by operation, initially for high-value skimage
  functions. `cucim.clara` is temporarily outside the first compute milestone,
  but a time-boxed feature-complete Windows/Clara investigation is a named
  near-term deliverable rather than an indefinite omission. PyTorch remains out
  of the initial CUDA operation family; Metal/MPS and MLX are later Apple-
  accelerator research candidates behind the same provider contracts.
- The base installation, plugin discovery, workflow loading, generated Python,
  and CPU execution must work without importing an optional GPU package.
- `core/` remains free of Qt and napari imports. Qt presents decisions; it does
  not make implementation, fallback, memory, or scientific-parity decisions.
- Existing CPU functions in `core/operations.py` remain the scientific
  reference. A GPU path must not change an algorithm, default, boundary mode,
  dtype conversion, initialization, PSF rule, clipping rule, or axis meaning.
- A device array may exist only inside the runtime execution scope. It must not
  appear in `PrototypePipeline.outputs`, `node_outputs`, a `SourcePayload`, a
  napari layer, an exported result, or a batch writer.
- RTX 4050 measurements are evidence for ordering and feasibility, not a
  universal `Auto` policy. Unknown workload-policy regions resolve to CPU;
  missing local completed-run history does not, because Auto then uses the
  reviewed default for an otherwise admitted region. Accelerated-only history
  schedules one same-surface CPU exploration run before later measured
  selection.
- GPU work is introduced behind contracts and promotion gates. The first
  headless vertical slice covers Subtract Background/Rolling-Ball Background,
  median, and 2D/3D Gaussian. Subsequent completed slices add ordinary RL,
  RL-TV, Otsu, Canny, Sigma Filter, Connected Components, and the basic schemas
  of Measure Objects and Measure Objects + Intensity. Convert Dtype and
  inexpensive residency bridges are next; other reasonable nodes follow in
  evidence-driven families.
- Admission and visibility are region-specific. Once a provider region has
  passed its scientific-parity and required memory, progress, cancellation,
  cleanup, and runtime gates, it is visible as a normal `Custom` and Prefer
  GPU candidate; it may participate in `Auto` after the applicable end-to-end
  benefit gate.
  `developer_hidden` is only for incomplete or unvalidated work. Unsupported
  dtype, parameter, shape, environment, or platform regions visibly resolve or
  fall back to CPU rather than hiding a validated provider or coercing the
  authored workflow. Branch visibility is not a blanket released-package or
  cross-platform support claim.
- Capability declarations are dtype-explicit and designed for bool, common
  microscopy integers, float32, float64, and non-finite policies from the
  beginning. A provider may still expose only the dtype/parameter regions that
  have passed parity and memory gates; unsupported regions use a visible CPU
  decision or fallback rather than an implicit conversion.
- Dtype is therefore part of GPU eligibility, not an incidental implementation
  detail. VIPP may explain that an explicit, scientifically appropriate
  conversion can unlock another implementation, but must never insert such a
  conversion merely to improve a benchmark. For example, the current Gaussian
  GPU region is finite `float32`; native `uint16` remains CPU-only. Ordinary RL
  initially requires explicit finite `float32` for both Image and PSF and
  returns shape-preserving `float32`. Converting
  integer values of magnitude up to 2^24 to `float32` with the Convert Dtype
  node's `Scaling = Preserve` option is exact (including all `uint16` values);
  its default `Rescale` option intentionally remaps the range. The workflow's
  public dtype, downstream semantics, range, writers, cache identity, and
  RAM/VRAM use still change and require review.
- Benchmarking is transactional and scientific: it never runs writers or other
  side effects, never changes live caches until the user accepts a choice, and
  excludes an implementation immediately if the candidate output fails its
  parity policy.
- Existing unrelated working-tree changes, especially the current graph,
  parameter-visibility, toolbar, and axis-semantics work, must be preserved.

Delivery remains split into small reviewable passes. Phase 1 groups the
headless contracts, execution/benchmark substrate, reproducible developer
environments, and the first three operation families. Later phases add the
interactive controls, persistence/batch/export surfaces, deconvolution and
segmentation waves, broader node coverage, and release-grade platform gates.

### Phase 1 implementation status (2026-07-28)

The development branch now contains the Phase 1 headless vertical slice:

- the then-current immutable CPU/Auto/Custom contracts, per-node preference,
  typed fallback, operation,
  policy, benchmark, scientific-cache, and execution-report contracts;
- a lazy provider registry, graph/device planner, transactional executor,
  private CuPy allocation scope, cleanup/OOM recovery, and host-only public
  result boundaries;
- production adapters for cuCIM Rolling-Ball/Subtract
  Background and CuPyX median plus 2D/3D Gaussian, each with explicit admitted
  dtype/parameter regions and conservative CPU decisions outside them;
- parity-before-timing node benchmarking, exact local benchmark fingerprints,
  a transfer-aware graph optimizer, and separate scientific-result identity;
  and
- reproducible CUDA 12/13 setup/doctor paths, including checksum-controlled
  installation of the pinned native-Windows cuCIM research wheel.

On the named native-Windows RTX 5090 host, every initial operation has passed a
real finite-float32 execution region, the compatible Background → Gaussian →
Median graph runs as one device-resident segment with one H2D and one D2H
boundary, and a deliberately constrained allocation proves classified OOM
cleanup followed by successful reuse. These results complete the local Phase 1
implementation gate. Their validated regions are normal public candidates in
the development toolbar; unsupported regions visibly remain on CPU. This does
not claim portable Auto calibration or release-wide support. A second Windows
RTX 40-series tier, supported Linux
hosts, clean packaging/JIT evidence, and the M1 Max provider study remain named
promotion or Phase 2+ gates. The concise implementation and validation handoff
is the [GPU Phase 1 implementation record](gpu-phase1-implementation-report.md).
The final Phase 1/pre-UI branch-wide validation completed with 2,375 passes and
two expected, documented integer-Gaussian xfails; the post-record real-device-focused run
completed with 167 passes and no skips.

### Phase 2 interactive slice status (2026-07-29)

At that dated milestone, the branch had the then-current compact toolbar
`CPU`/`Auto`/`Custom` policy selector (before Prefer GPU was added),
Settings-menu mirror, simplified dynamic Custom per-node choices, accepted
CPU/CuPy/cuCIM node badges with muted stale state, amber CPU fallback, and one
message-strip component whose major/actionable paths are severity-classified.
The current optimizer update adds a distinct per-node lock and removes the
ambiguous whole-analysis override.
Policy edits participate in in-session undo/redo, per-node cache provenance is
scoped to exact actual implementation and node-local compute intent, and the
toolbar collapses dynamically before primary graph actions can be compressed.
Small interactive Auto updates remain on CPU; background-eligible Auto work and
explicit GPU selections use the detached compute service. Full-width message
treatment is reserved for actionable failures. Workflow-v4 now persists only
portable authored compute intent, while batch/generated/export readers preserve
that block but still execute on CPU. Compute setup runs off the GUI thread and
reports system RAM plus dedicated or unified accelerator memory. Selected-node
benchmarking is review-before-apply and stores raw evidence only in a
machine-local cache.

The Custom-only, review-first `Find fastest` pipeline analysis is now
implemented. The current backend is only the starting assignment. Every
scientifically eligible CPU/CuPy/cuCIM implementation participates for every
unlocked node; only a separate explicit node lock constrains the search, and
applying a winning assignment does not lock it. The analysis requires a
calculated coherent graph, detaches source arrays and
workflow state, excludes writers/side effects from private execution, reuses or
measures each eligible unlocked exact workload, measures directional transfers
against a current free/total VRAM snapshot and active cap/reserve, and solves a
bounded graph-global assignment including transfer and liveness memory costs.
The proposed and current
assignments are then run privately for operation-specific parity and paired
end-to-end timing. Validation covers each changed node and every affected
retained/terminal/tunnel boundary, and every run must prove the exact requested
decision map and environment, no fallback, and successful accelerator cleanup.
No changed assignment is offered unless parity and synchronization pass,
the measured saving exceeds the greater of 5% or 10 ms, and the paired lower
confidence speedup bound is above 1.0. A current or authored CPU/Best
GPU/library/exact selection is not implicitly a lock. Analysis does not mutate
preferences, locks, or live caches.
The analysis identity binds graph, exact source content/state, retention,
environment, and per-node workload. Apply rechecks the editor graph, exact
source bytes/metadata/image state, compute request and lock state, current actual
assignment,
and a fresh probe of the exact candidate environment, then writes one undoable
authored-intent edit and invalidates only branches downstream of changed
choices.

The optimizer fails closed for missing or stale benchmark identity, incomplete
candidate/resident timing, unavailable or unknown VRAM, unsupported transfer
runtimes, unsafe retained writer paths, no feasible assignment, no material
benefit, cancellation, or deadline expiry. This first application version covers
one shared accelerator runtime and single-output operations supported by exact
node benchmarking, including nodes with multiple ordered inputs; it does not
execute writers, support multi-output benchmarks, optimize batch/generated
surfaces, or synthesize estimates for unsupported nodes. Its current dialog shows
the reviewed assignment, fixed/excluded rows, validated totals, and confidence
bound; detailed transfer-boundary, peak-VRAM, and per-refusal drill-down remain
presentation work.

Structural cache reuse is independently chained to exact scientific context:
source content/state, operation parameters and incoming topology, upstream
result contexts, and actual versioned implementations. It fails closed on
missing provenance while preserving exact upstream caches across unrelated
downstream preference edits.

The earlier post-slice full repository run completed with 2,431 passes. After
the whole-pipeline optimizer and exact cache-provenance integration, the full
repository run completed with 2,539 passes and the same two expected
integer-Gaussian xfails on 2026-07-28. Focused core whole-pipeline optimizer and
graph optimizer validation completed with 39 passes. A separate focused
optimizer/coordinator/dialog suite completed with 34 passes. After adversarial
assignment, parity, cleanup, transfer-lifetime, and cache-provenance hardening,
the final branch-wide run completed with **2,549 passes and the same two expected
xfails**; repository Ruff and `git diff --check` were clean.

### Phase 2B ordinary Richardson-Lucy status (2026-07-29)

The branch now contains a public-candidate `rl-cupy-f32-v1` implementation
backed by CuPy and `cupyx.scipy.signal`. It preserves the ordinary CPU operation's
prepared-call parameters and zero-fill `same` convolution semantics, accepts 2D
or 3D spatial data with arbitrary leading blocks, keeps Image, PSF, output, and
image-sized intermediates device-resident, and returns shape-preserving
`float32`. Its admitted region requires an explicit finite `float32` Image and
PSF, a resolved matching 2D/3D PSF rank, non-empty compatible extents, positive
PSF mass, odd PSF extents, the default-safe normalization/clipping/scale
options, `filter_epsilon` exactly at the evidence-backed `1e-8` point, and 1..25
iterations. The unchanged CPU default (`1e-12`), every other epsilon, and longer
runs visibly remain on CPU; VIPP neither alters a scientific threshold nor
truncates a run.
VIPP does not insert a dtype conversion. A reviewed `Convert Dtype` node can
unlock the candidate when
that representation change is scientifically appropriate.

The production benchmark adapter and application coordinator now accept one or
more ordered inputs for a single-output pure operation. Every input is detached,
byte-hashed, transferred, included in memory accounting, and rechecked for
staleness; changing only the PSF therefore invalidates exact RL evidence. Writer
and multi-output benchmarks still fail closed. Typed planning propagates RL's
fixed-`float32` output metadata and conservative array facts without inspecting a
device result.

A fair process-wide lease serializes accelerator work for each
`(runtime_id, device_id)` key. The lease is reentrant for the owning thread,
cancellable and deadline-aware, releases on every exit path, and permits work on
different device keys to proceed independently. Device execution, transfer
measurement, exact node benchmarking, and each pipeline-optimizer GPU
subtransaction use this common lease, preventing simultaneous same-device VIPP
work. Lease wait time consumes the same absolute analysis deadline. Holding one
lease across the optimizer's entire paired evidence window remains required to
prevent unrelated work from interleaving between subtransactions.

Large FFT-backed providers additionally run with new plan caching disabled only
inside the VIPP private allocator scope. Pre-existing per-thread/device plans
and cache limits are held and restored exactly after private-pool cleanup. A
512×512 RL regression that formerly retained 8,821,760 private bytes now proves
zero terminal live/reserved/out-of-pool bytes across two same-runtime runs. The
versioned RL FFT memory model admits 55,973,460 bytes for that 512×512/13×13
workload, bounding its 33,554,432-byte observed private-plus-out-of-pool peak.

Richardson-Lucy reports synchronized progress after each completed iteration of
each leading block and checks cancellation at those boundaries. Its exact
benchmark parity policy requires matching shape and `float32` dtype, identical
finite masks with completely finite output, NRMSE `<= 2e-6`, and
`max_abs <= 1e-6 + 5e-6 * reference_peak`; maximum float32 ULP distance is
reported as a diagnostic, not a separate pass gate. Final focused validation on
2026-07-29 completed with **315 passes** and 18 warnings, including real CUDA
provider and exact benchmark paths, lease contention, multi-input invalidation,
optimizer reuse, typed metadata, cleanup, progress, and cancellation. The final
branch-wide run completed with **2,675 passes, 2 documented xfails, and 83
warnings**. This is development-host evidence, not a public support or full
cross-platform promotion claim.

The initial numerical rectangle was narrowed after real-device adversarial
validation. Across 164 normalized nonnegative float32 2D/3D fixtures, exactly
`1e-8` passed the production gate through 25 iterations with a worst normalized
gate score of 0.864348. The threshold response was not monotonic: `1e-7` failed
one fixture at 25 iterations, and `1e-6` failed one as early as 10. At 50
iterations, `1e-8` failed four fixtures. A separate matrix rejected the
provisional `1e-10` point, and 40 even-PSF comparison fixtures had 14 failures
at 25 iterations for every tested epsilon. Policy therefore admits at most 25
iterations, exactly `1e-8`, odd PSF extents, and default-safe options. Optimizer
selection retains the exact-workload parity gate, and neither CPU defaults nor
tolerances changed.

Large-stack timing on the same development host passed exact parity for a
private 8.51-million-voxel ND2 `ZYX` volume and deterministic 16.78/67.11-million
voxel 3D shape stresses. Transfer-inclusive CuPy medians were 0.551, 0.411, and
1.524 seconds versus CPU medians of 24.381, 34.968, and 144.137 seconds: paired
median speedups of 45.03x, 85.06x, and 94.58x. This was a three-pair descriptive
screen. Observed device peaks of 0.697, 1.098, and 4.502 GiB stayed within
admitted bounds of 1.361, 2.111, and 7.720 GiB, respectively. This is not a
durable optimizer record or portable hardware promise. The
[versioned timing summary](benchmarks/rl-cupy-performance-windows-rtx5090.md)
retains resident/transfer timing, memory, cleanup, environment, and raw-sample
context.

### Phase 2C Richardson-Lucy TV status (2026-07-29)

The branch now contains public-candidate `rl-tv-cupy-f32-v1`, reusing the
ordinary RL input, PSF, block, progress, cancellation, transfer, lease, and
cleanup substrate. It preserves the CPU recurrence's constant `0.5`
initialization, zero-extension FFT convolution, flipped PSF, minus sign,
repeated central/one-sided `gradient` stencil, unit spacing, threshold branch,
denominator floor, per-iteration sanitization, non-negativity clamp, defaults,
and fixed `float32` output. The CPU operation and shipped workflows are
unchanged.

Admission deliberately distinguishes two scientific profiles. With
`tv_regularization == 0`, the TV branch is inactive and the provider inherits
ordinary RL's strict `filter_epsilon == 1e-8`, 1..25-iteration region and parity
gate. Positive TV is initially admitted only for the exact shipped tuple
`tv_regularization=0.002`, `tv_epsilon=1e-6`, `filter_epsilon=1e-12`, and
`denominator_floor=0.05`, at exactly 10 or 25 iterations, with the same
finite-float32, odd-PSF, resolved-rank, and default-safe-option requirements.
Other positive-TV iteration counts remain on CPU until their nonlinear
trajectories are measured; lambda-zero retains ordinary RL's 1..25 region.

Positive TV has a separately versioned nonlinear gate: equal shape/dtype and
finite/non-negative contract, NRMSE `<= 0.005`, and
`max_abs <= 1e-6 + 0.005 * reference_peak`. The ordinary RL gate was not reused
because the regularized recurrence amplifies tiny cross-library convolution and
reduction-order differences: 113 of the inherited 164 adversarial fixtures
missed that much tighter gate at 25 iterations. Under the RL-TV-specific screen,
all 164 inherited fixtures passed at 10 and 25 iterations, with worst normalized
gate scores 0.45744 and 0.44384. An independently constructed 96-fixture
2D/3D holdout also had zero failures at both iteration counts, with worst scores
0.22686 and 0.24209. Maintained phantoms additionally gate feature recovery,
MSE, flux, borders, and floor/threshold diagnostics. This evidence supports
public visibility only for the exact profiles; calibrated biological datasets
and cross-platform replication remain mandatory before broader restoration,
release, or platform claims.

Machine-local positive-TV timing at 25 iterations passed exact parity for the
private 8.51-million-voxel ND2 `ZYX` volume and a deterministic
16.78-million-voxel 3D shape stress. Transfer-inclusive CuPy medians were 0.450
and 0.684 seconds versus CPU medians of 34.862 and 56.817 seconds: paired median
speedups of 78.61x and 83.02x. Observed device peaks of 0.934 and 1.873 GiB
were bounded by final admitted limits of 1.876 and 3.127 GiB, respectively
(1.501 and 2.502 GiB before uncertainty).
These are short descriptive RTX 5090 measurements, not portable performance or
universal Auto claims; the
[versioned timing summary](benchmarks/rl-tv-cupy-performance-windows-rtx5090.md)
retains paired samples, transfer/resident timing, memory, cleanup, environment,
and source-currentness context.

Current limits remain explicit: RL and RL-TV expose only their exact validated
regions; portable broad Auto admission is not claimed; exact benchmarking still requires one output and
excludes writers; pipeline optimization still supports
one accelerator runtime; native Linux and secondary Windows hardware evidence
is pending; and the UI's separately captured optimizer inputs have not yet been
replaced with one immutable application snapshot. Batch, generated Python/CLI,
and standalone export now use the same admitted CPU/GPU implementations and
durable execution contract as the interactive service.

### Phase 3A Canny and Otsu status (2026-07-29)

The branch now contains two exact-mask public candidates. Canny uses a
CuPy/CuPyX adapter (`cupyx-canny-edges-exact-v1`) that mirrors the CPU operation
instead of delegating to a high-level GPU Canny call. It preserves float32 plane
conversion, constant-boundary Gaussian/Sobel arithmetic, bilinear
non-maximum-suppression tie behavior, eight-connected hysteresis, ordered
quantiles, leading blocks, and explicit BT.601 RGB/RGBA luma conversion. The
initial region accepts bool, `uint8`, and `uint16`, with canonical sigma from 0
through 12 and the existing quantile parameter contract. Authored `float32`
uses a visible CPU decision because CUDA subnormal flush-to-zero can change final
mask bits even for finite input. A custom correlation kernel preserves SciPy's
observable outside-in accumulation order inside the admitted integer/bool region.
Raw cuCIM Canny was rejected for this region because adversarial edge masks were
not exactly equal; library availability does not override final-mask parity.

Otsu uses a CuPy adapter (`cupy-otsu-threshold-exact-v1`) that keeps image-sized
finite masking, histogram construction, and final threshold comparison on the
device, then transfers only a bounded histogram for the authoritative NumPy
float64 cumulative arithmetic and first-maximum tie break. It preserves boolean
identity, exact native integer levels and offsets (subject to the existing
65,536-level span guard), float16/32/64 histogram edges, non-finite handling,
constant/error behavior, stack/slice scope, BT.601 luma conversion, and strict
`image > threshold` output. A raw cuCIM threshold scalar alone was insufficient
to establish those complete public semantics. A bounded atomic `uint64`
histogram replaces CuPy/CUB's device-occupancy-dependent wide-histogram
workspace while preserving the exact counts.

Both providers declare fixed boolean output and `mask-bitwise-v1` parity.
Memory admission covers their image-sized device intermediates and bounded
histograms; completed plane/histogram milestones are reported only after stream
synchronization, with cancellation checked at honest operation boundaries.
All public exact candidates additionally require the reviewed CPU reference
stack (NumPy 2.5.1, SciPy 1.18.0, scikit-image 0.26.0); a missing or changed
version visibly retains CPU before GPU probing. Regions outside the declarations
visibly use CPU. Focused implementation tests and exploratory parity matrices
validated the exact candidate contracts, which are available in normal
`Auto`/`Custom` pipelines. The source-current schema-v3 canonical record passed 28/28
exact-mask admission cases. On the 8x1024x1024 `uint16` stack, Canny measured
0.6812 seconds CPU versus 0.0349 seconds GPU end-to-end (19.51x), and Otsu
measured 0.0455 versus 0.0077 seconds (5.92x). On the privacy-redacted
8.51-million-voxel ND2 volume, the corresponding speedups were 16.40x and
5.28x. Observed synthetic-stack private-pool peaks were 72,516,608 bytes for
Canny and 84,377,600 bytes for Otsu, both inside admission; cancellation and
zero-residue cleanup passed. The schema-v3 validator also enforces source
fingerprint integrity and the closed, identifier-free private-source metadata
contract. These are machine-local screens, not portable performance or
universal Auto claims. See the
[Canny/Otsu implementation record](gpu-phase3-canny-otsu-implementation-report.md)
and [canonical evidence](benchmarks/canny-otsu-cupy-windows-rtx5090.md).

### Phase 4 Sigma Filter status (2026-08-02)

Phase 4 adds the public `sigma_filter` node under `Filtering > Smoothing &
Denoising` and a public CuPy provider (`cupy-sigma-filter-v1`). The
authoritative CPU operation is a clean-room Lee sigma filter compatible with
the documented behavior of Fiji Sigma Filter Plus. It processes resolved `YX`
planes independently over every channel and leading stack index, uses
nearest/clamped borders, keeps the input immutable, and preserves finite
`uint8`, `uint16`, or `float32` shape/dtype. Version 1 deliberately has no
ROI/mask input.

The scientific contract freezes Fiji's two radius plateaus and circular
footprint, float32 samples and squares widened into ordered float64 sums,
population variance, the inclusive center-relative sigma interval, exact
minimum-count ceiling, both fallback modes, and unsigned half-up restoration.
It has two narrow reviewed differences from the published plugin: VIPP uses
exact `ceil(N * fraction)` instead of the Java approximation, and clamps any
cancellation-induced negative variance to positive zero rather than allowing a
NaN-dependent result. Fourteen independently generated unsigned fixtures from
the official plugin bytecode match exactly; two additional frozen cases prove
and label those intentional deviations.

The GPU provider is a fused, lazy-imported CuPy `RawKernel`, not a CuPyX or
cuCIM primitive. Each output thread scans the ordered footprint twice. One
explicit contiguous float32 staging array handles arbitrary channel positions
and non-contiguous inputs; no host image round-trip and no image-by-footprint
tensor is used. Kernel options disable fused multiply-add and request precise
division/square-root. Explicit bit conversion preserves subnormal float32
samples, squares, and outputs even if NVRTC forces flush-to-zero. Sixty-four-row
tiles bound launch duration, synchronize truthful progress, and provide
cancellation boundaries.

The exact public region is non-empty finite native-endian `uint8`, `uint16`, or `float32`,
radius 0.5–10, finite non-negative sigma width, minimum fraction 0–1, boolean
fallback mode, and a valid optional channel axis leaving two resolved YX axes.
Float32 also requires complete finite extrema and magnitude no greater than the
float32-square-safe bound. Unsigned output requires bitwise parity. Float32 uses
the versioned NRMSE `<= 2e-6` and
`max_abs <= 1e-6 + 4*eps(float32)*max(1,input_peak,CPU_peak)` gate, plus exact
finite/zero/sign masks and separate adversarial selection/fallback decisions.
No CPU/GPU cache-equivalence group is declared.

Memory admission includes resident input/output, complete float32 staging, a
worst-case typed axis-restoration buffer, the bounded 325-offset table, status,
and uncertainty; there is no image-sized neighborhood expansion. Optional CUDA
imports remain lazy, runtime cleanup is transactional, and missing packages or
out-of-region calls retain a visible CPU decision/fallback. The source-current
full-profile RTX 5090 record passed all 10 exact admission cases, 10 matched
rejections, 18 bitwise-exact timed workloads, synchronized cancellation, and
zero-residue cleanup. The implementation and immutable v4 policy artifact both
declared `public_auto_candidate`; current artifact v5 retains that record. The
exact region is visible in ordinary Custom pipelines and can participate in
Auto. Representative end-to-end
speedups were 23.57x at 512²/radius 0.5, 55.23x at 512²/radius 2, 170.95x at
2048²/radius 10, and 93.62x for an 8×512²/radius-2 stack. Radius 0.5 first
cleared both gates at 512²: its 20.13-ms saving exceeded the 20-ms material gate
and its paired 95% speedup lower bound was 19.58x against the 1.20x confidence
gate. Radius 2 also cleared at 512², and radii 5/10 at the smallest tested 256².
Broader dtypes, values, runtimes, and platforms stay on CPU, and these crossovers
are machine-local. See the
[Phase 4 implementation record](gpu-phase4-sigma-filter-implementation-report.md)
and [canonical evidence](benchmarks/sigma-filter-cupy-windows-rtx5090.md).

### Phase 5 Connected Components status (2026-08-02)

Phase 5 promotes the existing public `label_connected_components` node through
an exact CuPyX adapter (`cupyx-connected-components-v1`). The authoritative CPU
contract remains nonzero foreground, SciPy face/full binary connectivity,
shape-preserving native `int32` output, and independent leading spatial blocks
whose deterministic IDs restart at one. GPU parity requires those actual IDs
bit for bit; equivalence only after relabeling fails, and the admitted CuPyX
path needs no canonicalizer.

The public accelerator region is boolean input with resolved 2D or 3D spatial
rank, valid face/full connectivity, fewer than 2,147,483,646 elements per
spatial block, and the pinned native-Windows CuPyX/CPU-reference environment.
Numeric nonzero-mask conversion, 1D labeling, oversized blocks, and unqualified
environments remain visible CPU decisions. Invalid authored rank/mode or
connectivity retains the CPU error contract instead of being mislabeled as a
fallback.

The provider labels each spatial block directly into one resident `int32`
output and can remain in the same CuPy domain as an upstream Otsu mask. The
memory model holds the complete bool input plus `int32` output and seven bytes
of workspace for one active block: 12 bytes per element for a single plane or
volume. Every canonical private-pool high-water observation stayed below that
provider estimate, and cleanup returned used and reserved pool bytes to zero.

Progress and cancellation occur at synchronized leading-block boundaries. This
is useful for stacks, but one plane or one 3D volume is a single atomic CuPyX
call: it cannot truthfully report intermediate completion or cancel mid-volume.
A chunked implementation would need new seam-merging and exact label-order
evidence.

The source-current RTX 5090 record passed all 16 exact admission cases across
2D/3D, face/full connectivity, patterns, leading blocks, deterministic repeats,
and `int32` ID resets. Its timing matrix demonstrates workload-dependent CPU/
GPU choices rather than a size-only rule. The table's faster-median label is
machine-local screening, not a durable optimizer record or direct Auto
assignment; production Auto still requires exact workload/environment,
confidence, absolute-saving, transfer, and neighboring-residency evidence.

The implementation and immutable packaged compute-policy artifact v5 declare
the region `public_auto_candidate`; v5 is a strict v4 extension and historical
v1-v4 resource hashes are unchanged. See the
[Phase 5 implementation record](gpu-phase5-connected-components-implementation-report.md)
and [canonical evidence](benchmarks/connected-components-cupyx-windows-rtx5090.md).

### Phase 6 basic Measurements status (2026-08-04)

Phase 6 promotes the basic schemas of `measure_objects` and
`measure_objects_intensity` through cuCIM implementations. The exact public
region is native non-negative `int32` labels, resolved 2D/3D spatial blocks,
arbitrary sparse positive IDs, and—in the intensity-aware node—a same-shape
native `bool`, `uint8`, `uint16`, or finite `float32` intensity image. Leading
blocks and explicitly resolved spatial axes retain CPU row order, calibration,
units, and independent block semantics.

Extended shape, axis, 2D boundary, derived-ratio, and 2D moment columns remain
on CPU. Otherwise valid non-negative integer label arrays outside native
`int32`, unsupported intensity dtypes, incomplete non-negative or finite facts,
empty spatial blocks, and blocks at or above the compact-label `int32` bound
receive an explicit CPU reason. Boolean/non-integer labels, negative labels, and
invalid or mismatched authored layouts are invalid for both CPU and GPU and
retain the CPU validation error rather than appearing as fallbacks. The provider
never changes a dtype, drops requested columns, or substitutes a different table
schema to gain GPU eligibility.

The resident provider emits a private C-contiguous packed `float64` matrix. A
mandatory D2H transfer and operation-owned host finalizer reconstruct the exact
typed `TableData` only after CUDA cleanup and scope exit. This boundary is part
of scientific identity, parity, memory, timing, and graph optimization. The
result is host-only: the graph solver must charge a later H2D transfer if a
future accelerator operation consumes data derived from the table.

The source-current RTX 5090 profile passed 11 admission, 11 rejection, and two
lifecycle cases with zero per-call pool residue. Full-public results ranged
from CPU wins for small planes and one float32 intensity stack to 1.87x at
1024² morphology, 16.37x at 2048² morphology plus `uint16` intensity, 7.28x on
a 32×256×256 intensity volume, and 23.90x on a 64×512×512 confocal-like volume.
The specs are therefore public Auto candidates, but selection remains exact-
workload and graph-context specific rather than a hard-coded size rule.

cuCIM's tiny lazy 2D/3D region-properties and Euler lookup allocations are
primed in a dedicated process-lifetime module pool before transactional pools
open. They are not result caches; every private execution pool still drains to
zero used and reserved bytes. See the
[Phase 6 implementation record](gpu-phase6-measurements-implementation-report.md)
and [canonical evidence](benchmarks/measurements-cucim-windows-rtx5090.md).

Immediate hardening before this optimizer can support broader operation and
platform claims:

- replace the UI's separately captured workflow, source payload, retention set,
  compute request, and accepted assignment with one immutable application
  snapshot created under a single coherent capture boundary; the worker must
  never derive identity from mutable live pipeline containers. The exact
  candidate environment is now freshly re-probed before apply, but workload and
  retention identity must also be reconstructed from the same coherent snapshot
  so the complete comparison has no time-of-check/time-of-use gap;
- preserve one absolute end-to-end deadline through detachment, byte hashing,
  fact scans, node sub-benchmarks, transfer profiling, solving, and validation.
  Cooperative checks now cover chunked work and subtransactions, but tests must
  continue to prove that no nested service silently receives a fresh full budget;
  monolithic provider calls remain cancel-after-return by necessity and must be
  labelled that way;
- preserve the exact performance-evidence envelope already carried by the
  optimizer: pipeline/workload/environment/device/memory/policy identity must
  stay attached through every future consumer. Never reintroduce a flattened
  `(node, implementation)` view that could clear Auto for a different graph
  context; Custom authored choices may outlive stale evidence, but every
  `fastest` or `optimal` claim must not;
- expand beyond the now-supported ordered-multi-input/single-output path only
  with explicit evidence for multi-output nodes, multiple accelerator runtimes,
  writer-adjacent graphs, and richer retained/previewed materialization. The
  current refusal is preferable to an optimistic estimate;
- make both benchmark dialogs lifecycle-safe: owner shutdown and worker-start
  failure must clear modal state, late results/Apply signals must be ignored,
  and registry cleanup failure must invalidate rather than publish evidence;
- revalidate selected-node source bytes as well as workflow history before
  Apply, move expensive source/environment verification off the modal GUI
  thread, and retain a final cheap generation guard for the commit; and
- prevent unrelated same-device work from entering between paired optimizer
  timing subtransactions, or prove an equivalent contract that excludes lease
  wait/foreign work from the accepted timing evidence.

### Cross-platform support contract

"Works on Windows, macOS, and Linux" applies to the VIPP application, saved
workflows, generated Python, and CPU results. It cannot mean NVIDIA GPU
execution on all three operating systems. NVIDIA's
[CUDA 10.2 release notes](https://docs.nvidia.com/cuda/archive/10.2/pdf/CUDA_Toolkit_Release_Notes.pdf)
state that 10.2 was the last toolkit release to support macOS, while this plan
uses current CuPy 14 with CUDA 12 or 13. Compiling CuPy, cuCIM, PyTorch CUDA
code, or a custom CUDA extension cannot restore a CUDA driver/runtime that
NVIDIA no longer supplies for macOS. If NVIDIA GPU execution on macOS is a
release requirement, this plan is a no-go rather than an implementation risk.

The supported product matrix is:

| Surface | Windows | Linux | macOS |
| --- | --- | --- | --- |
| Base VIPP install, import, CPU execution, workflows, batch, and generated Python | Required and tested | Required and tested | Required and tested |
| NVIDIA GPU execution | Native CuPy/CUDA path on validated x86-64 environments | CuPy/CUDA path on validated NVIDIA-supported glibc distributions and architectures | Not available; `Auto` uses CPU, Custom visible fallback uses CPU with a warning, and strict CUDA selection fails preflight |
| GPU package installation | Platform-marked optional extra | Platform-marked optional extra | CUDA packages are not resolved or installed |
| Saved implementation preferences | Open portably; resolve against the current environment and fallback policy | Open portably; resolve against the current environment and fallback policy | Open portably; CUDA preferences are unavailable and resolve/fail according to the recorded policy |
| Apple GPU research | Not applicable | Not applicable | M1 Max validation host; evaluate a Metal/MPS or MLX runtime after the CUDA substrate is stable, without promising parity or coverage before evidence exists |

This plan does not claim support for every Linux distribution. The release
matrix must name the tested glibc distributions and architectures. A platform
without a compatible wheel is supported for GPU use only after a reproducible
source build, real-device probe, scientific parity suite, and packaging smoke
test pass on that exact platform. Alpine/musl and other unvalidated targets are
CPU-only even if a local experimental build happens to succeed.

Dependency admission follows these rules:

1. Every base dependency must install and pass the CPU/package test matrix on
   Windows, macOS, and Linux for every supported Python version.
2. A CUDA dependency may be a platform-specific optional extra only when the
   base application neither imports nor requires it. The dependency must have a
   maintained wheel or a documented, CI-proven source build for each advertised
   Windows/Linux target.
3. A source-build claim is invalid when the underlying vendor runtime does not
   support the operating system. This explicitly rules out CUDA builds on
   current macOS.
4. A provider with a narrower binary-package matrix than CuPy may remain an
   evaluation candidate when upstream documents a source build. It becomes a
   production option only after reproducible builds, package artifacts, real-
   device probes, scientific parity, and operation-level value are shown
   on every Windows/Linux target where VIPP would advertise that provider.

Current provider audit:

| Library/runtime | Windows | Linux | macOS | Plan decision |
| --- | --- | --- | --- | --- |
| NumPy, SciPy, scikit-image | Existing CPU stack | Existing CPU stack | Existing CPU stack | Keep as the cross-platform scientific reference |
| CuPy/CuPyX 14 + CUDA 12/13 components | Official wheels; source build possible with a supported CUDA toolchain | Official x86-64/aarch64 wheels; source build possible on validated CUDA/glibc targets | No current CUDA runtime or CuPy wheel | Allow only as a lazy, platform-marked Windows/Linux optional provider |
| cuCIM/RAPIDS | No official wheel; pinned `v26.06.00` Python/skimage source wheel reproduced on one native-Windows RTX 5090 host with small downstream patches; native Clara I/O is not in that wheel and requires the separate Windows port | Official wheels and Ubuntu-tested source instructions; named target validation still required | No CUDA runtime | Continue as a narrow implementation-library candidate while the [Windows port plan](cucim-windows-port-plan.md) evaluates maintainable full packaging and Clara support; the current result advances but does not complete Pass 9 |
| PyTorch | Package available; CUDA builds available | Package available; CUDA builds available | Package available with CPU/Metal, not NVIDIA CUDA | Do not add it to the initial CUDA substrate: it is a second large runtime with unproved VIPP operation coverage. Retain MPS as one candidate for the separate Apple feasibility study |
| Apple Metal/MPS or MLX | Not applicable | Not applicable | Native Apple-silicon acceleration with unified memory is technically possible | Investigate after Phase 1 as a separate runtime provider; admit only operation families that preserve VIPP semantics and materially outperform the M1 Max CPU path |

Platform claims must be rechecked before changing dependency ranges or cutting
a GPU release. The primary sources are the
[CuPy installation matrix](https://docs.cupy.dev/en/stable/install.html),
[RAPIDS system requirements](https://docs.rapids.ai/install/),
[cuCIM source-build guide](https://github.com/rapidsai/cucim/blob/main/CONTRIBUTING.md#setting-up-your-build-environment),
[PyTorch local installation matrix](https://docs.pytorch.org/get-started/locally/),
the [PyTorch macOS Metal backend](https://docs.pytorch.org/docs/stable/notes/mps),
and [MLX unified-memory model](https://ml-explore.github.io/mlx/build/html/usage/unified_memory.html).

### Native-Windows cuCIM evidence snapshot

The 2026-07-15/16 follow-up completed the first native-Windows cuCIM sub-gate;
the 2026-07-27 Phase 1 refresh rebuilt and checksum-installed the artifact in
the dedicated VIPP CUDA environment.
The full procedure, package audit, output schemas, timing ranges, and artifacts
are in the
[cuCIM Windows source evaluation](cucim-windows-source-evaluation.md).

No credible third-party Windows binary was found. The official PyPI 26.6.0
files are manylinux x86-64/aarch64 wheels, the RAPIDS conda and nightly channels
publish Linux packages, GitHub releases contain no Windows assets, and upstream
Windows compatibility issue 454 remains open. The audit also found no Windows-
named branch among the 83 current forks; that is supporting evidence, not a
guarantee that no private or obscure build exists.

The pinned historical research result was:

| Build item | Evidence |
| --- | --- |
| Source | cuCIM `v26.06.00`, commit `3c15781c207eab93a317dd9803a6e726fe01f7c4` |
| Artifact | `cucim_cu13-26.6.0-cp312-cp312-win_amd64.whl`, 8,654,879 bytes, SHA-256 `586D3443091EEA67CE2C697BE2C490CA51977A5DBDF894B9318B270977134CF8` |
| Host | Windows 10, Python 3.12.9, RTX 5090 compute capability 12.0, CuPy 14.1.1, CUDA toolkit package 13.2.2, NVCC/nvJitLink 13.2.86, runtime API 13.2 (`13020`), driver API 13.3 (`13030`) |
| Available surface | `cucim.skimage` and `cucim.core` |
| Unavailable surface in this artifact | Native `cucim.clara/libcucim` whole-slide image I/O; feasibility and delivery are owned by the [Windows port plan](cucim-windows-port-plan.md) |
| Clean procedural reproduction | Fresh clone/build/install plus Gaussian, rolling-ball, and labeling real-kernel probe passed |
| Selected upstream tests | Complete median file: 707 passed, 4 skipped; other selected operation tests: 172 passed, 8 skipped, 6 deselected |
| VIPP application environment | Checksum-aware install into `.venv-gpu-cu13`, CuPy/cuCIM probes and `pip check` passed; background adapter 98 passed, including 45 real RTX cases (integer exact, float32 bounded v2) |

The refreshed build removed the earlier NVCC 13.3/13.2-runtime mismatch: CUDA
compiler, runtime, CRT, NVVM, and nvJitLink build components are now pinned to
13.2.86, matching the admitted CuPy 14.1.1 CUDA-minor evidence. nvImageCodec
0.8.0.22 is installed explicitly so the source wheel's declared dependencies
pass `pip check`. The table's archive checksum identifies only that now-missing
historical wheel.

The 2026-08-06 release recipe supersedes that installation procedure. It
materializes the real licence files, corrects Windows/CUDA metadata, exact-pins
the build-backend dependency source, removes the unusable Clara entry point,
and emits a schema-v2 manifest. Two clean builds matched the policy-pinned
canonical payload SHA-256
`d640d1e17bcce15d32d03841997252bf915b63da855e406c35f0d70c5a5ea667`.
Each user still builds and keeps their own wheel privately; the older archive
digest is not an installer input or a hosted artifact.

The historical clean build required three downstream adaptations: put Git for
Windows' `which.exe` on `PATH` for `rapids-build-backend`; replace the materialized
relative `VERSION` symlink and include it in the wheel; and replace one
deprecated NumPy shape assignment in vendored padding code with `reshape` for
strict NumPy 2.5 compatibility. These are packaging/build-compatibility changes,
not image-processing formula changes. The fixed release builder now performs
the broader six-adaptation procedure documented in the source evaluation:
[`scripts/build_cucim_windows.ps1`](../scripts/build_cucim_windows.ps1).

The synchronized RTX 5090 standard benchmark produced primitive-level
feasibility evidence. Times include one input and output transfer. A speedup
above 1 means cuCIM was faster than the named primitive baseline; none of these
rows is production-node admission.

| Primitive workload | Comparison baseline | cuCIM end-to-end speedup | Result |
| --- | --- | ---: | --- |
| Rolling ball 2D / 3D | scikit-image CPU primitive | **265.46x / 528.66x** | Fixture matched; advance to complete VIPP-adapter validation |
| Canny 2D | scikit-image CPU primitive | **17.03x** | Fixture matched; advance to complete VIPP-adapter validation |
| Region-properties table | scikit-image CPU primitive | **10.49x** | Fixture values matched; full VIPP schema/overflow adapter required |
| Connected components 2D / 3D | scikit-image CPU primitive | **2.87x / 2.84x** | Primitive fixture matched; the later exact CuPyX production adapter is implemented, while cuCIM remains a possible complete-adapter comparator |
| Otsu threshold 2D | scikit-image CPU primitive | **2.38x** | Fixture `float32` scalar matched; advance to VIPP histogram/mask validation |
| uint16 31x31 histogram median | CuPyX median | 1.42x | Exact primitive result; map the production-adapter crossover before Auto admission |
| Gaussian 2D / 3D | CuPyX Gaussian | 1.03x / 0.95x | Exact; keep CuPy |
| float32 5x5 median | CuPyX median | 1.08x | Exact; keep CuPy |
| Sobel / binary closing | normalized CuPyX composition | 0.98x / 1.01x | Equivalent; keep CuPy |
| Richardson-Lucy 2D / 3D | explicit CuPyX loop | 0.22x / 0.22x | Allclose but about 4.5x slower; keep explicit CuPy |

All 15 primitive benchmark fixtures passed the recorded numerical comparison.
The region-properties values matched but cuCIM returned narrower area, label,
and bounding-box dtypes than the scikit-image baseline table. The fast histogram median also
requires a dense rectangular footprint. Both constraints must remain explicit
support-policy checks rather than silent behavior changes.

This evidence changes cuCIM from “Windows feasibility unknown” to “promising
narrow Windows implementation-library candidate.” It does not itself admit a
production dependency or a cuCIM-backed VIPP node. Connected Components was
later promoted through an independently validated CuPyX adapter; cuCIM would
still have to pass that complete contract before competing with it.
Pass 9 still requires supported-Linux builds, another Windows GPU tier, CUDA
policy, clean-install/JIT and memory measurements, cancellation behavior,
schema adapters, and optional-extra/CI maintenance. The upstream-versus-
downstream strategy and a bounded attempt to port native Clara are now specified
in the [cuCIM Windows port plan](cucim-windows-port-plan.md). macOS uses the CPU
path initially because it has no NVIDIA CUDA runtime while a separate Apple
Metal/MPS/MLX provider is investigated.

## 1. Target architecture

### 1.1 Components and ownership

The target keeps graph definition, scientific CPU functions, runtime/
implementation code,
execution policy, and UI presentation separate.

| Component | Proposed owner | Responsibility |
| --- | --- | --- |
| Compute request and result contracts | `core/compute.py` | Immutable `CPU`/`Auto`/`Prefer GPU`/`Custom` intent, per-node preferences, valid fallback policy, selected device, precision policy, typed reason codes, and JSON-safe reports. |
| Operation implementation declarations | `core/compute_specs.py` | Immutable declarations associated with operation IDs. Each CPU/CuPy/cuCIM/future implementation declares its runtime, exact dtype/parameter region, parity, memory, progress, and benchmark contracts without importing optional packages. |
| Runtime and implementation registry | `core/compute_registry.py` | Lazy runtime descriptors, entry-point discovery, instance lifetime and capability probing, plus implementation lookup by stable ID. It distinguishes an array runtime (`numpy`, `cuda-cupy`, future `metal-*`) from an implementation library (`cpu`, `cupyx`, `cucim`). |
| Workload policy | `core/compute_policy.py` plus packaged JSON under `src/napari_vipp/compute_policies/` | Deterministic support and benefit decisions from workload, topology, environment, reviewed thresholds, and non-stale local benchmark records. CPU is conservative outside a validated region. |
| Graph planning | `core/device_execution.py` | Scheduled-node closure, implementation assignment, maximal same-runtime device segments, runtime-transition costs, boundary transfers, liveness, memory preflight, fallback planning, and execution reports. |
| CPU/GPU call preparation | a small extraction from `core/pipeline.py` into `core/node_execution.py` | Build validated operation inputs/kwargs and apply existing metadata transforms once, independent of the chosen implementation. The CPU path uses the same extraction. |
| Built-in CUDA/CuPy runtime | `core/gpu/cupy_runtime.py` | Lazy CuPy import, real device probe, device/context and private-pool scope, transfer/synchronization primitives, OOM classification, cleanup, environment identity, and verified sharing rules for cuCIM implementations. |
| GPU implementations | one family-owned module per implementation library under `core/gpu/` | Pure CuPyX or cuCIM operations accepting and returning runtime-owned device arrays. They mirror current CPU semantics and expose no UI behavior. |
| Benchmark and optimizer services | `core/compute_benchmark.py`, `core/compute_benchmark_coordinator.py`, `core/compute_pipeline_optimizer.py`, and `core/compute_pipeline_optimizer_coordinator.py` | Transactional node benchmarking, local fingerprinted result storage, parity-before-timing checks, cold/warm timing, application-safe evidence capture, and whole-pipeline assignment including transfers, residency, runtime switches, and memory. |
| Capability/policy diagnostics | `core/compute_diagnostics.py` | JSON-safe support report, installation diagnosis, policy explanation, memory snapshot, and recent execution/fallback information. |
| Single-run service | `core/execution.py` | Introduced as the mandatory headless/device execution entry in Pass 1, then made the only interactive application entry in Pass 4. It validates a detached workflow, plans, executes, and returns host outputs plus provenance. |
| Interactive presentation | a reusable controller under `ui/compute.py`, composed by `_widget.py` | Main-toolbar mode dropdown, Custom node preferences, distinct per-node optimizer locks, node/pipeline benchmark actions, compact CPU/CuPy/cuCIM badges, RAM/accelerator-memory status, fallback display, and copyable install guidance. No provider import or policy logic. |
| Batch integration | `core/batch.py`, `core/batch_setup.py`, and existing `ui/batch*` adapters | Persist the run request, reuse the core execution service per item, checkpoint decisions in manifests, cancel safely, and clean runtime state at item/run boundaries. |
| Workflow and generated Python | `core/workflow.py`, `core/export.py` | Persist portable global mode and per-node preferences plus non-scientific optimizer-lock UI metadata, not machine timings or resolved hardware; migrate v3 safely, expose explicit runtime overrides, and return/write provenance. |

`core/pipeline.py` is already large and has active unrelated changes. Accelerator
callables must not be added there. The preferred association is an immutable
side table in `core/compute_specs.py`, keyed by operation ID and validated
against `NODE_LIBRARY_BY_ID`. `OperationSpec` remains the source of graph and
scientific-call metadata; `compute_specs_for(operation_id)` is the source of
compute implementations. This avoids importing optional implementation code from the node
library and lets operation-family agents edit separate declaration blocks.

### 1.2 Core value objects

The public contracts should settle on these concepts before an operation is
promoted:

- `ComputeRequest`: global mode (`cpu`, `auto`, `prefer_gpu`, or `custom`), immutable
  node-ID-to-preference mapping, visible/strict fallback policy, selected
  runtime/device, precision-policy ID, workload-policy ID, accelerator-memory
  budget, and safety reserve. It has no Qt types.
- `NodeComputePreference`: `auto`, `cpu`, `best_gpu`, an implementation-library
  preference such as `cupyx`/`cucim`, or an advanced stable implementation ID.
  Preferences are retained while another global mode is active but ignored
  outside Custom.
- Optimizer locks are deliberately not part of `ComputeRequest` or
  `NodeComputePreference`. They control which alternatives `Find fastest` may
  investigate; they do not alter normal execution, scientific cache identity,
  or the meaning of an authored backend preference.
- `ComputeEnvironment`: detected runtimes and implementation libraries,
  versions, driver/runtime, Python implementation/minor/ABI tag, device
  identity/class, discrete/unified memory topology, OS execution mode, and probe
  status.
- `OperationComputeSpec`: one immutable declared implementation and its runtime,
  public/input/internal dtypes, conversion policy, support/parity/memory/
  progress constraints, and stable implementation/library IDs.
- `ArrayFacts`: revision-keyed per-port facts needed by support/performance
  policy, including finite counts, min/max/range, label maximum/count,
  foreground density where relevant, strides/contiguity, and guarantees
  propagated by upstream contracts. Facts are complete for any scientific
  exclusion; sampled facts may inform performance only and are labelled.
- `WorkloadDescriptor`: operation, resolved dimensions, shapes, dtypes,
  parameters, `ArrayFacts`, topology/transfer facts, device/host tiers, and
  available memory, including the cost of any required fact scan.
- `BenchmarkRecord`: environment/workload fingerprints, candidates, cold and
  warm synchronized samples, parity result, memory observations, expiry rules,
  and artifact/policy versions. Raw timings are local environment data.
- `NodeExecutionDecision`: global mode, node preference, resolved runtime and
  implementation, policy/benchmark record, reason code/text, estimate, and
  whether a fallback occurred.
- `ExecutionSegment`: a host node or a maximal connected same-runtime device
  sub-DAG with
  entry/exit ports, retained ports, liveness counts, and a memory estimate.
- `ExecutionPlan`: immutable ordered segments plus environment and decisions.
- `ExecutionProvenance`: run-level environment and per-node records, cache-key
  digests, fallback events, warnings, and cleanup outcome.

`PipelineRunRequest` gains a `compute_request` value. `PipelineRunResult` gains
an `execution_report`; errors become typed internally but retain the current
user-facing `error` string for compatibility. No request or result contains a
device array.

### 1.3 Control and data flow

```mermaid
flowchart TD
    U["User selects CPU, Auto, Prefer GPU, or Custom"] --> UI["Qt presentation captures ComputeRequest"]
    S["Custom node preferences or benchmark actions"] --> UI
    UI --> R["PipelineRunRequest with detached workflow and host snapshots"]
    R --> V["Qt-free workflow, source, axis, grid, and manual/dirty validation"]
    V --> C["Lazy capability and environment snapshot"]
    C --> P["Reviewed policy plus whole-graph planner"]
    HST["Auto: exact completed history isolated by execution surface"] --> P
    BENCH["Custom: explicit node and pipeline benchmark evidence"] --> P
    P --> M["Accelerator and host-memory preflight"]
    M --> E["Segment executor"]
    E -->|"host segment"| CPU["Existing CPU operation functions"]
    E -->|"device segment"| GPU["Lazy runtime plus CuPyX/cuCIM implementation"]
    GPU --> B["Host materialization only at declared boundaries"]
    CPU --> H["Host-only result store"]
    B --> H
    H --> O["PrototypePipeline host caches, napari presentation, save/export"]
    E --> PR["ExecutionProvenance and visible decisions/fallbacks"]
    PR --> O
```

Preflight is complete before a scientific or side-effecting node runs. A strict
Custom preference that cannot be honored therefore fails without partial
graph execution. Device segments are execution transactions: device values and
provisional host copies are committed to the public host result store only
after the whole segment succeeds. This makes one-time CPU retry after a GPU OOM
safe and prevents duplicate side effects.

## 2. Backend semantics

### 2.1 User-visible meanings

| Toolbar mode | Exact behavior |
| --- | --- |
| `CPU` | Do not discover, probe, or import an accelerator runtime for execution. Every scientific operation uses the current CPU implementation. This is the compatibility/reproduction mode and the migration result for workflow v3. |
| `Auto` | Default for new interactive sessions. With no exact compatible history, choose from reviewed GPU defaults wherever the scientific, environment, and memory gates pass. If history is accelerated-only, the next global Auto run measures the authoritative CPU assignment once on the same execution surface. Once both observations exist, a later matching run uses acceleration only when it clears the reviewed 1.20x/20-ms benefit margin; otherwise CPU wins. Auto never silently benchmarks multiple implementations. |
| `Custom` | Show `Auto for this node`, `CPU`, and one library choice such as `GPU · CuPy` or `GPU · cuCIM` for every implemented node. `Auto for this node` authors no backend pin and uses the reviewed Auto default; completed-run history is consulted only when the global mode is Auto. Show `Best GPU` only when at least two distinct libraries compete. Exact pins are advanced-only, although a loaded pin remains visibly represented until replaced. Unimplemented nodes remain visibly CPU. Preferences are planned together, so the UI may explain that a locally faster node would make the complete pipeline slower by forcing a transfer/runtime boundary. |

New sessions default to Auto even when no accelerator package is installed. In
that environment Auto runs normally on CPU, the toolbar status says that GPU
acceleration is not installed, and diagnostics offer one copyable command for
the compatible optional extra. Absence of an optional package is not an error.

Only successful, fallback-free completed full-pipeline wall times enter Auto
history. Interactive, batch, and registry-lifecycle execution surfaces have
separate keys and are never combined into a timing pair. The one CPU
exploration run is reported as Auto performance exploration; node/provider and
whole-pipeline multi-implementation comparisons remain explicit Custom actions.

`Auto` CPU selection and runtime fallback have different machine-readable
states. Use `decision_kind=policy_cpu` for the former and
`decision_kind=fallback_cpu` plus `fallback_reason` for the latter. The spike's
current `BackendSelection.fell_back=True` for an unsupported Auto operation and
its eager capability detection before an explicit CPU decision are corrected
during Pass 0.

### 2.2 Custom choices and fallback

`ComputeRequest.node_preferences` is a validated immutable mapping keyed by
stable node ID. It is active only in Custom mode but retained when the user
temporarily changes global mode. A node preference may be:

- `auto`: use the reviewed Auto default without authoring a backend pin (shown
  as `Auto for this node` in the node dropdown); this Custom preference does
  not consume raw benchmark records or Auto's completed-run history;
- `cpu`: require the scientific reference implementation;
- `best_gpu`: require the best supported GPU assignment under whole-graph
  planning without forcing the user to choose a library;
- `library:<id>`: choose the best validated implementation from that library,
  for example `cupyx` or `cucim`; or
- `implementation:<stable-id>`: advanced exact pin used for reproduction or an
  accepted benchmark result.

The ordinary node dropdown deliberately compresses this full contract. It
shows one option per library and exposes `Best GPU` only when multiple libraries
are meaningful alternatives. Exact implementation IDs are authored only by
advanced/developer tooling. If a workflow already contains an exact pin, the
dropdown shows that current value as an `Advanced pin · <library>` entry. A
known pin excluded by the active admission setting is marked `unavailable`, as
is an unknown ID, until the user selects a normal replacement; the control must
never silently display `Auto for this node` while retaining the pin.

Interactive Custom mode defaults to **visible fallback** for usability. If a
forced `best_gpu`, library, or exact-implementation choice is unavailable,
unsupported for the actual dtype/parameters, or encounters a classified OOM,
the planner may choose CPU once and
must show a persistent node badge and run-level warning. An advanced **Fail if
a selected GPU cannot run** switch changes the same request to fail complete
preflight instead. Batch, headless, and generated callers can select either
policy explicitly. Invalid parameters, axis/grid errors, parity failures,
unclassified runtime errors, and writer errors never become fallbacks.
Custom `auto` choosing CPU is a normal policy result, not fallback.

### 2.3 Mixed graphs and preflight

- `CPU`: never mixed.
- `Auto`: mixed graphs are expected and partitioned at operation, support,
  benefit, runtime, or memory boundaries.
- `Custom`: authored node preferences are constraints on a whole-graph plan,
  not independent wrappers. Compatible CuPyX and cuCIM implementations may stay
  in one CUDA/CuPy segment only after their zero-copy interoperability contract
  passes; otherwise a runtime/library transition is costed as a boundary.
- That constraint describes ordinary Custom **execution** after preferences
  have been authored. During a `Find fastest` analysis, an unlocked node's
  current preference is only the baseline assignment and every scientifically
  eligible alternative is considered. The optimizer proposes preference edits;
  only separate optimizer-lock metadata limits its search.
- Skipped manual nodes are not scheduled. A cached skipped-manual output is a
  host boundary. A manual node explicitly selected for calculation is included
  in preflight and benchmarking.
- A side-effecting `Save Image` or batch publication node is never part of a
  device segment and cannot run until all required compute preflight succeeds.

Fallback is permitted only for enumerated cases: implementation unavailable
before launch, unsupported declared region, memory preflight, and runtime-
classified OOM. A fallback never silently changes the stored preference and one
item's fallback never changes later batch items.

### 2.4 Persistence and portability

Compute intent belongs in workflow JSON once the headless execution, UI, cache,
and provenance contracts are stable. Pass 4 introduces the minimal workflow v4
compute block together with the public controls so accepted benchmark choices
survive reopening; Pass 8 extends the same frozen schema into generated Python,
CLI, batch/hash integration, and export sidecars. Workflow v4 adds:

```json
"execution": {
  "compute": {
    "mode": "auto",
    "fallback_policy": "visible",
    "node_preferences": {
      "median_filter_1": "implementation:vipp.cupy.median_filter"
    },
    "precision_policy": "scientific-default-v1",
    "workload_policy": "vipp-best-available-v1"
  }
}
```

Optimizer locks are stored separately as non-scientific workflow UI metadata,
for example:

```json
"metadata": {
  "vipp": {
    "compute_optimizer": {
      "locked_node_ids": ["background_1"]
    }
  }
}
```

The list is validated against current node IDs. Its absence means every node is
unlocked, preserving older workflow behavior. Lock state is portable so the
user's instruction survives reopening, but it is not copied into
`ComputeRequest`, normal execution provenance, candidate timing identity, or
scientific cache keys merely because it changes the optimizer's search space.
The review/apply transaction does include the exact lock set in its own analysis
identity so a lock edit makes an unaccepted proposal stale.

Device index, exact device identity, memory cap, driver, and resolved decisions
are runtime environment, not authored workflow intent, and are not stored here.
Benchmark samples, predicted times, and a machine-derived winner are also not
portable workflow data. When a user accepts a benchmark result, only the stable
node preference is saved; the local benchmark record remains in a fingerprinted
machine cache and becomes stale when inputs/parameters, VIPP, implementation,
runtime, Python implementation/minor/ABI, driver, or device identity changes,
with the separate node/pipeline
scopes defined in section 5.3.
Workflow v3 is safely migrated to the v4 CPU default because v3 could only run
the current CPU path; v1/v2 remain rejected for the existing scientific-schema
reasons.

Opening a workflow with CUDA preferences on CPU-only or Apple-only hardware
preserves the authored preferences and shows their unavailable status. Auto runs
on CPU. Custom visible-fallback mode runs on CPU with explicit warnings;
strict mode fails preflight. A session override does not mutate the document;
"Use CPU and save" or accepting new benchmark choices does. Generated Python
uses embedded intent unless the caller supplies an explicit override. Batch
config records the effective request so replay cannot inherit a different UI
preference.

Workflow loading validates node IDs and preference syntax without importing an
optional library. Unknown or currently unavailable stable implementation IDs
are preserved so a portable workflow does not lose intent; preflight, not
deserialization, explains availability. An exact implementation pin is stronger
intent, not a guarantee of bit-for-bit reproduction without the recorded
version/environment/provenance.

Deleting a node removes its authored preference. Graph duplicate/copy-paste
remaps and copies the authored preference to each new node ID, but never local
benchmark evidence, planned/used decisions, caches, or hardware state. `Paste
parameters` remains scientific-parameter-only and does not replace the target's
compute preference; a future explicitly named `Paste all node settings` action
would be required to do that.

The canonical scientific workflow hash includes authored compute intent after
v4. Non-scientific optimizer-lock metadata does not change that hash. It never
includes a resolved implementation, device, or policy result. Those belong to
execution provenance.

### 2.5 Consequences across execution surfaces

| Surface | Consequence of the compute contract |
| --- | --- |
| Interactive run | The session request is captured in every synchronous or background `PipelineRunRequest`; stale-run rejection compares graph/source plus compute mode, fallback, node preferences, and policy fingerprints, and discards provenance with a stale result. |
| Cache | Every output-port record carries scientific result identity plus separate request/decision provenance. Only the actual implementation and result-affecting semantics key values; global mode, preference, fallback, and benchmark evidence explain the decision but do not invalidate an otherwise identical result. An exact pin cannot consume another implementation's entry. |
| Batch | `BatchConfig` records the effective override; each item is replanned against current free memory but the request is unchanged. A fallback on one item does not silently rewrite later items to CPU. |
| Generated Python | Embedded workflow intent is the default. Function/CLI override is explicit, returned in provenance, and does not mutate `_WORKFLOW_JSON`. |
| Reproducibility | `CPU` is portable and stable. Auto is intentionally hardware-dependent. Prefer GPU expresses placement preference rather than a speed claim. Custom exact pins express stronger intent but may be unavailable elsewhere. Policy, environment, actual decisions, fallback records, and implementation versions are required to reproduce a result. |

## 3. Operation capability model

### 3.1 Immutable declaration

The declaration should be data, not code. A representative contract is:

```python
@dataclass(frozen=True, slots=True)
class OperationComputeSpec:
    operation_id: str
    runtime_id: str                         # "cpu-numpy" or "cuda-cupy"
    implementation_library_id: str          # "cpu", "cupyx", or "cucim"
    implementation_id: str                  # "vipp.cupy.median_filter"
    implementation_version: str             # "1"
    supported_spatial_dims: frozenset[int]  # {2} or {2, 3}
    input_contract_ids: tuple[tuple[str, str], ...]
    output_contract_ids: tuple[tuple[str, str], ...]
    array_domain: str                        # host-numpy/cuda-cupy/...
    admission_tier: str                      # developer_hidden/public_custom/public_auto_candidate
    validated_environment_policy_id: str     # OS/Python/runtime/library ranges
    parameter_policy_id: str                # exact range/mode validator
    parity_policy_id: str
    progress_granularity: str               # segment/kernel/iteration
    cancellation_granularity: str           # before-after-kernel/iteration
    supports_device_residency: bool
    temporary_memory_model_id: str
    workload_policy_id: str
    boundary_policy_id: str
    precision_policy_ids: tuple[str, ...]
    limitations: tuple[str, ...] = ()
```

Each referenced `ComputePortContract` declares the port name and value kind
(`array`, `scalar`, `table`, or another typed state), accepted public dtypes/
schema, value-domain facts, internal and accumulation dtype, output dtype/schema
rule, conversion/rounding/overflow policy, and non-finite policy. This per-port
model is mandatory for multi-input, dtype-changing, and table-producing nodes;
a coarse operation-wide dtype tuple is prohibited.

Admission is an explicit, region-specific lifecycle. `developer_hidden` is
reserved for incomplete or unvalidated work, is available only to an explicit
headless developer request, and is excluded from ordinary UI, workflow, Auto,
and Custom discovery. `public_custom` has passed scientific parity plus
the required memory, progress, cancellation, cleanup, and runtime gates and is
therefore visible in normal pipelines. `public_auto_candidate` has additionally
met the reviewed packaged-evidence requirements to be an Auto default, but only
inside its validated environment/workload policy. This admission evidence is
not machine-local learning and raw node benchmarks never teach Auto. A provider
is not hidden merely because some of its dtype/parameter/platform subregions
are unsupported: those calls receive a visible CPU decision or fallback. An
internal
`allow_experimental=True` test/developer flag is never serialized into workflow
JSON or exposed as a normal user setting.

Parameter support is resolved by a named pure validator in the policy layer,
not an arbitrary callable stored in `pipeline.py`. The validator returns a
typed support result with a stable reason code. Examples include odd median
sizes within the validated range, exact Gaussian sigma handling, RL spatial
rank, PSF rank, iteration range, normalization/clipping modes, and RL-TV
regularization parameters.

Declarations are narrower than library API availability but the contract is
not float32-specific:

- Subtract Background/Rolling-Ball Background: preserve current smoothing,
  light-background inversion, 2D/3D block/channel semantics, clipping, and
  public dtype restoration. Target `uint8`, `uint16`, `float32`, and `float64`
  independently; bool and non-finite cases remain CPU until proved.
- Median: current slice-wise YX semantics, odd validated kernel sizes,
  explicit/scalar channel behavior, reflect boundary, and separate declarations
  for each validated integer/float dtype region.
- Gaussian: separate 2D slice-wise and 3D declarations, exact sigma tuples,
  reflect boundary, zero-sigma behavior, excluded channel axes, and explicit
  native/public dtype behavior.
- RL/RL-TV: current float32 working/output semantics, 2D and 3D spatial blocks,
  matching PSF rank/grid, and all current public parameter modes only after each
  is tested.

Bool, additional integer/float regions, non-finite behavior, new boundary modes,
fast math, and dynamic/multi-output device implementations stay unsupported
until separately promoted. Architecture, registry, benchmark, memory, cache,
and provenance tests must nevertheless exercise declarations for all dtype
classes so later coverage does not require a redesign. An implementation may
not advertise a dtype or parameter merely because an optional library accepts
it.

### 3.2 Lazy implementation registry

`core/compute_registry.py` contains separate runtime and implementation-library
descriptors: stable ID, display name, import string, supported OS family,
array/device domain, interoperability claims, and optional entry-point origin.
It does not import an implementation module at registry construction. Built-in
implementation mappings use import strings:

```text
vipp.cupy.median_filter -> napari_vipp.core.gpu.cupy_median:median_filter
vipp.cucim.subtract_background -> napari_vipp.core.gpu.cucim_background:subtract_background
```

The registry loads a runtime/library only after request resolution says it may
be used. A CPU request never calls accelerator discovery or
`importlib.import_module` for CuPy/cuCIM. Third-party discovery uses separate
entry-point groups such as `napari_vipp.compute_runtimes` and
`napari_vipp.compute_implementations`; discovery failures become diagnostic records,
not plugin-import failures. Runtime, library, and implementation IDs are stable
cache/provenance identifiers; display names are not. cuCIM may participate in a
CUDA/CuPy-resident segment without a transfer only after tests prove it accepts
the runtime-owned array on the same device/stream and obeys allocator/lifetime
rules; a library name alone never authorizes zero-copy sharing.

Registry startup validates these invariants without loading optional runtimes or
implementations:

1. every declaration references an existing operation ID;
2. `(runtime_id, implementation_id, implementation_version)` is unique;
3. every named parity, parameter, memory, boundary, precision, and workload
   policy exists;
4. no production declaration exists without its promotion tests and policy
   record;
5. no accelerator module is imported while `napari_vipp` or the npe2 manifest is
   imported.

### 3.3 Capability detection lifecycle

Detection has two phases. Descriptor discovery lists built-in and entry-point
runtimes/libraries without importing them. A probe runs only when the UI asks
for accelerator status or an Auto/Prefer-GPU/Custom request needs it. The
CuPy probe records every visible device, then creates a context for the selected
device, checks driver/runtime identity and free/total memory, imports each required CuPyX submodule, executes
and synchronizes a real one-element kernel, and tears down the probe allocation.
Import success alone is never availability.

Reports distinguish runtime/library availability from operation promotion. A usable
CuPy installation can report zero supported VIPP operations. Supported IDs are
derived from validated `OperationComputeSpec` records, not supplied by the UI.
Probe results are cached by process/environment fingerprint, including Python
implementation/minor/ABI, and can be refreshed
after installation or a provider failure. A CPU request neither discovers entry
points nor probes/imports CuPy during execution.

Implementation-library health is probed separately and lazily. The developer
cuCIM probe verifies the pinned build digest, required submodules, one real
operation kernel, same-device/stream behavior, allocator/lifetime compatibility,
and zero-copy exchange with runtime-owned CuPy arrays. A healthy CuPy runtime
does not imply healthy cuCIM, and a failed library probe cannot poison CuPyX
implementations.

## 4. Device-resident graph execution

### 4.1 Planning model

Planning starts after source snapshots, graph restoration, dirty-closure, and
manual-node selection are known. It plans only nodes that this run will execute.
Each output port is the unit of storage and liveness.

A GPU segment is the maximal connected sub-DAG of scheduled nodes that:

- resolved to the same runtime, array/device domain, and device;
- have compatible validated implementations and precision policies;
- can exchange resident arrays;
- fit as a unit under the memory policy; and
- contain no source, writer, manual-skip, or other declared host/side-effect
  boundary. Host table inputs are boundaries. A validated device-side
  measurement may terminate a segment by converting its private device result
  through a typed host-table finalizer; public `TableState` never resides on the
  accelerator.

"Maximal" is topological, not merely a consecutive list. A segment can contain
a branch and later join. A CPU node or different runtime/device splits GPU
regions. Different implementation libraries do not split a region when their
declared and tested array-domain interoperability allows zero-copy sharing;
otherwise the planner inserts and costs the required boundary.

### 4.2 Internal array ownership

`DeviceValue` is an internal, non-public record owned by the segment executor:

```text
(runtime instance, device id, opaque runtime array, shape, dtype,
 output-port key, remaining device consumers)
```

The opaque array is never placed in `PrototypePipeline`. The segment's private
`ExecutionValueStore` may hold:

- a host value;
- a device value;
- both temporarily when a branch or retained output needs host materialization;
- the existing `ImageState`/`TableState`; and
- the cache/provenance record.

Public commit stores only the host value and state. All accepted interactive
cache entries remain host-owned in passes 0-10. Device arrays are not retained
across separate interactive runs because the current detached worker may not
reuse a stable execution thread/context and stale-run rejection must remain
simple. Dirty partial recalculation can reuse a clean cached host output; that
output is transferred once if it enters a new GPU segment.

### 4.3 Boundary rules and edge cases

- **Source:** one H2D copy per distinct source/output entering a segment. Aliased
  fan-out reuses the same device value.
- **Consecutive supported operations:** keep each output resident until its last
  device consumer completes.
- **CPU boundary:** materialize the required port once, synchronize it, and keep
  the device copy only if another device consumer remains.
- **Multi-input:** a GPU node can join resident inputs; host/cached inputs are
  copied at segment entry. All inputs must pass operation support and grid
  validation before any transfer. The image and PSF are distinct live inputs.
- **Branch/fan-out:** consumer counts cover every source port. A GPU-to-CPU and
  GPU-to-GPU fork creates one host copy and one continuing resident reference.
- **Multiple outputs:** plan, count, transfer, retain, and release each output
  port independently. Initial promoted operations are single-output; generic
  multi-output support is nevertheless tested with a fake runtime.
- **Scalar/table output:** a device implementation may compute these values but
  must terminate or materialize at a typed host finalizer that restores the
  declared scalar/table schema, units, ordering, and overflow policy before
  public commit. Downstream host consumers see only the existing host state.
- **Cached upstream output:** it is a host entry boundary. Its cache record is
  validated before transfer. Device identity from an older run is irrelevant.
- **Skipped manual node:** its valid/stale cached host output is a boundary. A
  manual node selected to calculate can join a GPU segment if promoted.
- **Retained/selected/pinned/preview output:** add its output port to the host
  materialization set. Copy it without breaking downstream residency. napari
  receives the existing detached host copy behavior.
- **Batch output/save:** always host-materialize before the existing staging and
  source-reverification sequence. Publication remains outside GPU scope.
- **Across batch items:** per-item sources, intermediates, and outputs are never
  retained. A fixed immutable source such as a shared PSF may remain resident in
  a batch-scope constant cache only when its exact source revision/cache key,
  runtime/device, dtype/shape, and read-only contract are unchanged, its bytes
  are reserved in every item's memory estimate, and no implementation mutates
  inputs. Release all such constants on cancel, runtime/library error, source refresh,
  or batch end.
- **Cancellation:** stop launching new work, synchronize already submitted work,
  release all private values, and return only after cleanup. Never commit a
  canceled segment.
- **Dirty recalculation:** clean cached host ports remain committed; only the
  planned dirty closure gets new decisions/cache keys. A canceled or failed
  clone never replaces live host caches.

### 4.4 Segment-planning pseudocode

```python
def plan_run(pipeline, run_scope, compute_request, environment, cached_records):
    scheduled = resolve_dirty_and_manual_scope(pipeline, run_scope, cached_records)
    descriptors = infer_workloads_without_executing_nodes(pipeline, scheduled)

    decisions = {}
    for node_id in topological_order(scheduled):
        node = pipeline.nodes[node_id]
        if is_host_infrastructure(node):
            decisions[node_id] = host_boundary("declared_host_boundary")
            continue

        candidates = validate_declared_candidates(
            node, descriptors[node_id], environment, compute_request
        )
        decisions[node_id] = candidate_decision_set(
            compute_request.mode,
            compute_request.preference_for(node_id),
            candidates,
        )

    # Auto and Custom are graph-global: include transfers, residency,
    # branches/joins, host materialization, memory, and authored constraints.
    decisions = solve_graph_assignment(
        pipeline, scheduled, descriptors, decisions, environment
    )

    failures = unhonored_strict_custom_preferences(decisions, compute_request)
    if failures:
        raise ComputePreflightError(all_failures=failures)

    regions = maximal_connected_gpu_subdags(pipeline, scheduled, decisions)
    segments = split_regions_until_memory_estimates_fit(regions, descriptors)
    ordered = topologically_order_host_nodes_and_segments(segments, scheduled)
    return ExecutionPlan(ordered, decisions, environment)
```

`infer_workloads_without_executing_nodes` uses source/cached shapes and pure
shape/dtype propagation. If an output shape cannot be known until execution,
Auto resolves that node to CPU. A Custom GPU requirement reports an
unsupported-dynamic-shape reason unless the declaration provides a safe
upper-bound model.

Support policies may also require complete `ArrayFacts`. Compute them lazily
from revision-keyed source/cached host arrays, or propagate a guarantee from a
validated upstream implementation. Their scan/transfer cost enters Auto and
benchmark estimates. A sampled scan may tune a cost model but never proves a
scientific value-domain restriction. If a required complete fact for an
intermediate cannot be known safely before execution, Auto chooses CPU and a
forced Custom choice reports the typed unsupported reason; it does not launch
speculatively and silently fall back after observing the values.

### 4.5 Execution pseudocode

```python
def execute_plan(plan, pipeline, host_cache, cancel):
    store = ExecutionValueStore.from_host_cache(host_cache)
    provenance = ExecutionProvenance.start(plan)
    try:
        for unit in plan.units:
            cancel.check()
            if unit.is_host:
                prepared = prepare_node_call(pipeline, unit.node_id, store.host_inputs)
                output = call_existing_cpu_function(prepared)
                store.commit_host(finalize_node_result(prepared, output))
                provenance.complete_host(unit)
                continue

            try:
                provisional = execute_gpu_segment_transaction(
                    unit, pipeline, store, cancel, provenance
                )
            except RuntimeOutOfMemory as oom:
                cleanup_and_synchronize_runtime(unit.runtime)
                if not may_retry_once(plan.request, unit, oom, provenance):
                    raise
                provisional = execute_segment_on_cpu_transaction(
                    unit, pipeline, store, cancel
                )
                provenance.record_fallback(unit, "runtime_oom")
            store.commit_host_only(provisional)
    finally:
        store.release_all_device_values()
        synchronize_and_apply_pool_policy(plan)
    return materialize_pipeline_host_result(store), provenance.finish()


def execute_gpu_segment_transaction(segment, pipeline, committed_store, cancel, prov):
    private = SegmentStore()
    runtime = registry.runtime(segment.runtime_id)
    with runtime.execution_scope(segment.device, segment.memory_policy):
        preflight_live_memory(segment, runtime.memory_snapshot())
        for entry_port in segment.host_entries:
            cancel.check()
            private.put_device(entry_port, runtime.to_device(committed_store.host(entry_port)))
            prov.transfer(entry_port, "host_to_device")

        for node_id in segment.topological_nodes:
            cancel.check()
            prepared = prepare_node_call(pipeline, node_id, private.inputs(node_id))
            device_output = registry.call(segment.implementation(node_id), prepared)
            runtime.operation_checkpoint(segment.progress_granularity(node_id), cancel)
            private.put_outputs(node_id, device_output)
            private.release_dead_values()

        for port in segment.host_outputs | segment.retained_outputs:
            cancel.check()
            private.put_provisional_host(port, runtime.to_host(private.device(port)))
            prov.transfer(port, "device_to_host")
        runtime.synchronize()
        cancel.check()
        return private.finalize_host_results_and_states()
```

## 5. Workload selection policy

### 5.1 Policy inputs

Every Auto decision uses a `WorkloadDescriptor` containing at least:

- operation and implementation ID/version;
- resolved 2D/3D spatial mode and complete array rank;
- input/output shape, element count, and dtype;
- strides/contiguity plus complete or explicitly sampled finite fraction,
  min/max/range, foreground density, and label count/maximum where the
  operation's support or cost depends on them;
- kernel size, sigma tuple, PSF shape, and iterations where relevant;
- all parameters that change work or memory;
- number and identity of resident GPU predecessors/successors;
- distinct H2D and D2H boundaries predicted for the candidate segment;
- whether an output must be retained, previewed, pinned, or saved;
- selected device tier and exact available memory snapshot;
- runtime, implementation-library/version, CUDA-major family, OS mode, Python
  implementation/minor/ABI tag, and driver;
- host tier or local host calibration, because CPU speed changes crossover;
- policy record ID/version and whether the descriptor is inside its validated
  interpolation bounds; and
- the measured/predicted cost of required host/device fact scans.

### 5.2 Threshold generation and storage

Extend `scripts/benchmark_gpu.py` into a matrix harness rather than converting
its RTX 4050 values into constants. For each promoted implementation, collect
warm, synchronized CPU time, resident GPU time, H2D/D2H time, end-to-end time,
peak live/pool memory, and parity over a factorial or space-filling matrix of:

- small through large 2D and 3D shapes;
- supported dtypes;
- kernel/PSF sizes and iteration counts;
- isolated node and representative resident chains;
- Windows native, Linux, and WSL2 where supported;
- CUDA 12 and CUDA 13 environments;
- multiple host/device performance tiers, including the minimum supported VRAM.

Use held-out workloads to fit and validate a simple explainable model or
piecewise threshold. Store reviewed records as versioned package data, not in
Python conditionals. A record includes runtime/library/implementation version
ranges, admitted Python implementation/minor/ABI, OS/runtime class, host/device
tier, feature bounds, predicted CPU/GPU/transfer cost coefficients or
breakpoints, required minimum speedup, confidence, and
benchmark artifact digests. Raw results remain in `docs/benchmarks/`; shipped
policy records are the reviewed derivative.

These shipped records define reviewed Auto defaults; they are not Auto's local
history. With no exact compatible history, Auto uses the reviewed default. An
accelerated-only observation causes the next global Auto run to measure CPU
once on the same execution surface; a later matching run applies the
1.20x/20-ms gate to the completed pair. Interactive, batch, and
registry-lifecycle timing surfaces are never mixed. Isolated node and **Find
fastest** records remain explicit Custom evidence, and Auto never silently
benchmarks multiple implementations.

Device tiers should be based on a short, deterministic startup/transfer/compute
microprofile plus compute capability and VRAM class, not model-name matching.
The microprofile contains no user image data and its local result is cached by
runtime/device fingerprint. If the profile is missing, stale, or
outside a shipped policy's validated bounds, Auto chooses CPU. Timing, tier,
policy ID, and decision enter benchmark identity/provenance, not scientific
result identity.

### 5.3 On-demand node and pipeline benchmarking

Custom mode exposes `Benchmark node` only when the selected node has at
least two validated candidates for its resolved inputs and parameters. The
main-toolbar `Find fastest pipeline` action is visible only in Custom mode and
only after enough source/cached metadata exists to construct workload
descriptors and at least one unlocked node has multiple eligible
implementations. For this analysis, current preferences form the baseline but
do not narrow any unlocked node's candidate set. A node is excluded from
comparison only by a separate explicit optimizer lock or by scientific/runtime
ineligibility.

Benchmarking follows these rules:

1. Use the exact current parameters and resolved input shape/dtype/axis/grid
   contract. Benchmarking a representative crop is allowed only as an explicitly
   labelled quick estimate and never silently replaces an exact full-workload
   record.
2. Run the existing CPU implementation once outside timed rounds as the
   scientific reference, then validate each admissible implementation in an
   isolated transaction before accepting any timing. A failed alternative is
   excluded and explained for that fingerprint; it does not abort comparison of
   other parity-qualified alternatives. The captured current implementation must
   remain qualified. Benchmarking may only evaluate an already promoted
   dtype/parameter support region; passing on one user's input never expands a
   declaration.
3. Record cold-start/JIT time separately and randomize paired CPU/candidate
   order after warmup. The manual `Benchmark node` surface begins with seven
   synchronized rounds. `Find fastest pipeline` uses progressive 3 -> 7 -> 15
   screening: a materially and consistently separated result stops early,
   whereas a close or mixed result receives more evidence. Use absolute paired
   duration/saving uncertainty for escalation; speedup-ratio variance alone can
   amplify a small GPU denominator and must not make an obvious winner run 15
   rounds. All parity-qualified candidates remain available to the graph solver,
   including a locally slower candidate that may save transfers globally.
   Report the paired median ratio and a versioned 95% paired-bootstrap lower
   confidence bound, plus end-to-end, resident, transfer, fact-scan, and
   peak-memory measures. The bootstrap method/seed and outlier policy are
   benchmark-policy data, not ad hoc UI behavior.
4. Do not execute source readers twice unnecessarily, writers, `Batch Output`,
   publication, or any other side effect. Do not replace live caches/history or
   node preferences until the user accepts the result. Support cancellation, a
   visible/configurable time budget, and
   release every device value on all exits. Before presenting or applying a
   result, recheck the detached graph/source/compute-intent fingerprint and
   discard it if the live state changed.
   The budget is one absolute elapsed wall-clock deadline for the complete
   analysis, not a host- or device-memory limit. A deadline expiry is
   inconclusive: no fastest assignment was determined, the current assignment
   was not proven optimal, and no preference may change. The result must name
   the active stage/node, elapsed time, completed evidence, and the action to
   retry with a longer limit. Preserve complete exact-key records for reuse,
   but discard every partial timing set for the node active at expiry.
   The current single-widget surface queues normal calculation while optimizer
   evidence owns the GPU window. Before headless or multi-window concurrency is
   promoted, replace that UI-only exclusion with a process-wide lease keyed by
   runtime and physical device. Lease waiting must honor cancellation and the
   same absolute deadline, permit different devices to proceed independently,
   and release on every success, error, cancellation, and cleanup path.
5. Before measuring, look up a complete record by the exact node input
   bytes/shape/dtype/strides, operation/parameters, VIPP/NumPy/SciPy/scikit-image
   and implementation/runtime versions, Python implementation/minor/ABI tag,
   driver, device, memory scope/topology, and measurement-policy fingerprint.
   Reuse complete timings plus explicitly typed deterministic scientific-parity
   rejections. Never reuse a transient runtime, OOM, cleanup, or timing failure;
   retry it and atomically replace the record instead. This prevents one
   temporary CUDA failure from blacklisting a viable implementation while
   avoiding repeated work for a reproducible parity mismatch.
   A cancellation time budget is not part of that scientific/performance key
   because it does not change a successfully completed measurement. Mark the
   record stale after any result- or timing-affecting identity change. A hit skips
   node timing only: fresh whole-pipeline parity remains mandatory before a
   changed assignment is offered.
6. `Use fastest` writes only a stable node preference. The UI retains the full
   local evidence for explanation but workflow JSON never stores raw timings,
   exact hardware, or an automatically resolved implementation.

Staleness has two scopes. A node record is invalidated by its operation,
parameters, resolved input revision/shape/dtype/axes/content facts, relevant
layout, implementation/dependency/runtime versions, Python implementation/
minor/ABI, driver/device, or memory-topology changes. A pipeline proposal also
invalidates on graph topology,
scheduled/manual scope, retained/selected/pinned/preview host materializations,
memory cap/reserve, global mode/fallback, current baseline assignment, or
optimizer-lock set. An
accepted implementation preference remains authored when evidence becomes
stale, but loses every `fastest`/`optimal` claim and offers `Rebenchmark`.

Whole-pipeline optimization is not a loop that independently chooses the
fastest implementation for each node. It reuses or collects candidate timings,
then solves a constrained graph assignment that includes H2D/D2H transfers,
same-runtime residency, CuPyX/cuCIM interoperability, branches/joins, required
host materializations, memory/liveness, and side-effect boundaries. It may
therefore choose a slightly slower implementation for one node to make the
complete pipeline faster. The proposed assignment and expected total are shown
before `Apply choices`; mandatory fresh parity and end-to-end timing validate the
winner without publishing outputs. Timing advances through 5 -> 7 -> 15 paired
rounds, stopping when the proposed winner or current assignment is decisively
separated by the versioned 5%/10-ms and confidence gates. A still-close result
at 15 rounds is inconclusive and cannot be applied. If the current assignment
wins, the analysis completes successfully and says so rather than presenting
“not beneficial” as an error.
The review reports the reverse confidence bound when final validation overturns
the graph model; it must not label the rejected alternative as validated.
Locked GPU nodes need no comparative timings, but they still contribute a
conservative current-workload memory estimate to graph-wide VRAM feasibility.

When measured candidates are within the noise floor (initially the greater of
5% or 10 ms for the measured unit), prefer the current valid choice; otherwise
prefer CPU for an isolated node or the choice that preserves a faster resident
pipeline segment. These thresholds are versioned policy, not UI literals.

### 5.4 Deterministic conservative algorithm

1. Reject GPU for unsupported dimension, dtype, parameter, precision, provider,
   boundary, or finite-value requirement.
2. Reject GPU for an absent/mismatched policy record or an out-of-domain
   descriptor.
3. Predict total segment cost, including distinct transfers and required host
   materializations—not only kernel time.
4. For automatic, non-benchmarked selection, require the lower confidence bound
   to predict at least 1.20x end-to-end speedup and at least 20 ms absolute
   savings for the complete candidate segment. These conservative defaults are
   versioned and may change only with reviewed cross-device evidence.
5. Form maximal candidate segments, recompute transfer counts, and demote
   segments whose end-to-end prediction no longer meets the gate. Repeat until
   stable; ties resolve to CPU.
6. Run memory preflight. Auto may demote a segment or split it at the least
   costly boundary; it must record that memory caused the CPU decision.

Custom provider/implementation pins ignore the Auto performance threshold
but never ignore support, scientific parity, or memory constraints. An accepted
local benchmark may choose a smaller but statistically clear win because it
measured the actual workload and environment.

## 6. GPU memory management

### 6.1 Discrete and unified memory topology

The existing host-RAM cache guard continues to own host arrays. On a discrete
CUDA device, accelerator policy is a separate `AcceleratorMemoryPolicy` shown
as VRAM beside RAM in the UI and never folded into `_pipeline_cache_nbytes`.
Host output materialization still counts toward the RAM guard; a run can pass
one guard and fail the other.

A future Apple Metal/MPS or MLX runtime must declare `memory_topology=unified`.
CPU and GPU then share one physical pool: VIPP must not display RAM plus VRAM as
if they were additive, and must not apply two independent caps. The accelerator
runtime reports its allocations/working-set recommendation while the platform
memory service reports system pressure; the
planner coordinates host cache, device working set, and safety reserve under a
single budget. Until an Apple provider passes its own gates, macOS shows RAM and
`Apple GPU acceleration not enabled`, not invented VRAM.

Recommended initial defaults, subject to empirical tuning before public release, are:

- set the configured runtime-managed allocation cap to 80% of device memory;
- keep `max(512 MiB, 10% of total device memory)` free as a device-wide safety
  reserve;
- also honor current device-wide free bytes, which accounts for other
  processes; and
- refuse preflight unless the conservative incremental peak fits both the
  runtime-pool headroom and device-wide headroom.

For discrete memory, take one synchronized snapshot immediately before
preflight and compute two independent headrooms:

```text
pool_headroom = max(0, configured_pool_cap - current_runtime_managed_bytes)
device_headroom = max(0, device_wide_free - safety_reserve)
```

The candidate's **incremental runtime-managed peak** must fit `pool_headroom`;
its **incremental total-device peak**, including new out-of-pool library/context
workspace and transfer staging, must fit `device_headroom`. Already-live VIPP
pool bytes and already-live out-of-pool bytes are already reflected in the
current counters/free snapshot and must not be subtracted a second time. A
total-at-peak model is acceptable only if it is first converted to these
incremental quantities against the same snapshot.

The UI permits a percentage or absolute cap but not less than the runtime's
documented minimum. The effective cap, reserve, topology, and pressure snapshot
are provenance fields. Discrete and unified defaults are separate reviewed
policies; the CUDA values above must not be copied blindly to unified memory.

### 6.2 Estimation and liveness

`MemoryEstimate` includes unique entry inputs, each live output, retained
device values at branches, operation temporaries, library workspace, transfer
staging, host materializations, and a calibrated uncertainty/fragmentation
margin. The segment planner computes port-level last use and a peak over the
topological schedule. Unified estimates additionally count host/device aliases
and copy-on-write behavior without double-counting a proven shared allocation.

Named implementation memory models cover at least:

- median/Gaussian: input, output, CuPyX workspace upper bound by rank/dtype/
  kernel, and branch-retained values;
- RL: observed image, normalized image, estimate, blur, ratio, correction, PSF,
  mirrored PSF, output, versioned padded real/complex FFT arrays, cuFFT plan
  workspaces, and a calibrated first-use/out-of-pool allowance;
- RL-TV: all RL buffers plus per-axis gradients, norm, normalized components,
  divergence, denominator, and stack/workspace behavior.

Unknown implementation workspace is measured across the declared support matrix
and stored with a safety multiplier. A model that cannot establish a safe upper
bound cannot be offered as a Custom GPU choice and causes Auto to choose CPU.

### 6.3 CuPy pool policy

Use a VIPP-owned CuPy `MemoryPool` within the CUDA runtime execution scope,
prefer a thread-local `cupy.cuda.using_allocator(...)` scope where supported,
and restore prior allocator state on exit. Set its hard limit to the effective
pool cap. Do not empty a process-global pool used by unrelated code. Initial
transfers are synchronous and do not introduce a long-lived pinned-host pool;
an optimized pinned pool requires separate host-memory accounting later.

The CuPy pool limit is not a complete VRAM limit. CUDA contexts, cuFFT/cuDNN or
other library handles, JIT modules, staging buffers, and native cuCIM allocations
may be outside it. Capture device-wide free/total before and after probe/run,
record VIPP pool live/reserved bytes, estimate a calibrated out-of-pool delta,
and include that delta plus margin in preflight and post-run leak checks.

Within a successful batch, free arrays at each item's liveness boundaries but
allow the VIPP pool to retain free blocks for reuse while it remains below the
cap. At item end, synchronize and prove that no live per-item device values
remain. Only the explicitly accounted immutable batch constants described above
may survive. Free cached pool blocks and batch constants on classified OOM,
when the cap is exceeded or reduced, on runtime teardown, and at batch end—not
after every node.

### 6.4 OOM and cleanup

- Catch only runtime-classified allocation OOM. Do not substring-match every
  CUDA error.
- Best-effort synchronize, drop all segment references, free VIPP-pool blocks,
  and capture estimate/free/cap/pool state.
- `Auto` retries the failed segment once on CPU. A Custom GPU requirement
  retries only under visible fallback; strict selection fails. Mark the segment
  as already retried to prevent loops.
- Retry uses committed host boundary inputs and commits no failed GPU output.
- If cleanup or synchronization reports a second runtime error, fail the run
  and mark the runtime unhealthy until capability refresh.
- Batch `finally` cleanup runs before staging/publication status is finalized.
  One failed item cannot leave device references reachable by the next item.

## 7. Cache identity and provenance

### 7.1 Cache records

The current cache is structural and host-only. Add a record beside every cached
output port so dirty-run hydration can prove scientific identity and request
admissibility. Keep three concepts separate:

1. `ScientificResultKey`: what actually produced these values;
2. `CacheAdmissibility`: whether the current request may reuse that actual
   implementation; and
3. `BenchmarkRecordKey`: machine/workload timing evidence from section 5.3.

The scientific result key is a canonical digest of:

- upstream source revision or upstream output-port cache keys;
- operation ID and canonical public parameters;
- actual runtime/library and implementation ID/version;
- normalized result-affecting dependency fingerprint, including relevant
  NumPy/SciPy/scikit-image/CuPy/cuCIM versions and any patched-wheel build
  digest; alternatively an implementation contract must bump its version for
  every supported dependency-range change;
- public/internal dtype, conversion, non-finite, precision, and accumulation
  policy IDs;
- parity-policy ID and any authorized equivalence group;
- only runtime/device properties declared scientifically result-affecting by the
  implementation contract; and
- relevant axis/grid metadata and existing operation/cache identity inputs.

Global mode, node preference, fallback reason, workload policy, benchmark
timings, transfer count, exact device, and driver normally belong to decision
provenance and the separate benchmark key—not result identity. Thus an Auto→CPU
result and explicit-CPU result may reuse the same scientific cache entry. A
current exact implementation pin rejects a cache produced by another
implementation even when both passed tolerance parity; automatic Auto,
Prefer-GPU, or Custom planning may reuse only an implementation its current
plan independently admits.

`CacheAdmissibility` is evaluated after current planning, never inferred from
the old cache entry: a `cpu` preference accepts CPU; `best_gpu` accepts a GPU
implementation chosen by the current graph plan and cannot consume CPU merely
because it is cached; a library pin accepts that library only; and an exact pin
accepts only that implementation or an authorized equivalence group. A cached
CPU fallback becomes reusable only after the current run independently reaches
the same valid visible-fallback decision. Strict selection never consumes it.

Human reason text is never hashed. Cached host arrays from the current pre-GPU
runtime receive CPU records when hydrated, or are conservatively invalidated if
their source/parameter identity cannot be reconstructed. Duplicate nodes copy
authored preferences but never copy local benchmark evidence.

CPU and GPU cache entries are separate by default, including median. A parity
policy may later publish a `cache_equivalence_group`, but only after bitwise
equivalence is proved over every supported dtype/parameter/boundary region and
reviewed explicitly. Tolerance-based parity never authorizes cache sharing.

### 7.2 Provenance schema

`ExecutionProvenance` records:

- request intent, effective session override, and fallback permission;
- capability/environment snapshot, runtime/library/driver, Python
  implementation/minor/ABI tag, device class, discrete/unified memory topology,
  memory policy, and policy IDs;
- per node: implementation, precision, support/policy decision, reason,
  transfers, estimate, actual runtime/library/implementation, fallback,
  progress granularity, cache hit/miss/key digest, and parity-policy ID;
- per-port public/internal/output dtype or table schema, conversion/rounding/
  overflow and non-finite policies, and observed/propagated `ArrayFacts`;
- result-affecting dependency/build fingerprint, estimated versus measured peak
  memory (pool and out-of-pool), and candidate quarantine details;
- benchmark source (`shipped-policy`, `local-node`, `local-pipeline`, or none),
  fingerprint/digest, staleness state, and whether the user accepted the choice;
- segment boundaries and materializations;
- OOM/runtime/library errors, retry count, cancellation point, and cleanup result;
- VIPP and CPU scientific dependency versions.

Integration behavior is exact:

- `PipelineRunResult.execution_report` carries the report to the widget.
- `PrototypePipeline` gains host-only per-port cache/provenance mappings when a
  result is accepted. Stale worker results remain discarded wholesale.
- `ImageState.history` keeps its current scientific operation descriptions;
  structured compute provenance is associated separately so display history is
  not polluted with hardware text.
- Undo/redo and workflow JSON store authored intent, not the last resolved run.
- Batch manifests add run-level compute environment/policy and item-level node
  decisions/fallbacks; each output records the provenance digest that produced
  it.
- Generated `PipelineResults` exposes `.execution_report` and
  `.output_provenance`.
- Standalone exported/saved results write an atomic
  `<output-name>.vipp-provenance.json` sidecar by default; formats with a stable
  metadata field may mirror a digest, but the JSON sidecar is authoritative.
- A diagnostic export contains the capability report, installation diagnosis,
  policy records used, memory snapshot, and recent decision/error codes, but no
  array data or raw device serial number.

## 8. Progress and cancellation

Extend progress from `(node, current, total, message)` to a backward-compatible
typed update that can also carry phase (`preflight`, `transfer`, `kernel`,
`iteration`, `materialize`, `cleanup`), segment ID, and indeterminate state. The
Qt worker continues to forward only data.

Exact operation behavior:

- **RL and RL-TV:** check cancellation before each iteration, launch the
  iteration's kernels, synchronize once at the iteration checkpoint, check
  again, then report completed progress. For leading non-spatial blocks, total
  remains `block_count * iterations`, matching current CPU behavior. A reported
  iteration is complete on the device, not merely queued.
- **Gaussian and median:** check before transfer, immediately before kernel
  launch, and after synchronization. The provider kernels are monolithic. Do
  not claim or display mid-kernel cancellation.
- **Transfers:** report start/completion by distinct boundary value. A transfer
  is complete only after the required stream synchronization.
- **Multi-node segment:** display overall node/segment position plus the active
  node phase. Determinate iteration progress takes precedence; monolithic work
  remains indeterminate.
- **Batch:** item progress contains nested segment/node/iteration detail. Batch
  cancellation stops starting new items and signals the current item.
- **Asynchronous cancellation:** changing the graph sets the existing event.
  The UI says "Cancel requested—waiting for current GPU step" while a
  monolithic kernel is outstanding. The result is not returned until streams
  are synchronized and device references are released.

Cancellation is not an error and does not trigger CPU fallback. A canceled
segment commits no new host output. Existing clean caches remain intact.

## 9. Packaging and environment support

### 9.1 Optional extras

Add no GPU dependency to base requirements. Add mutually exclusive extras:

```toml
gpu-cuda12 = [
    "cupy-cuda12x[ctk]>=14,<15; platform_system == 'Windows' or platform_system == 'Linux'",
]
gpu-cuda13 = [
    "cupy-cuda13x[ctk]>=14,<15; platform_system == 'Windows' or platform_system == 'Linux'",
]
```

VIPP's base package metadata accepts Python 3.12 and newer, but the initial
**admitted GPU matrix is CPython 3.12 only**. The setup scripts default to 3.12
and refuse another minor unless a developer explicitly opts into an unsupported
probe. Diagnostics report that state, and no public GPU implementation appears
as available in Auto, Prefer GPU, or Custom—including library or exact
pins—on an unvalidated Python implementation/minor/ABI. Visible fallback uses
CPU with the specific matrix reason; strict selection fails preflight. Only the explicit
headless developer-experimental path may probe such an environment, and it
cannot create public admission evidence by itself. Expand the advertised GPU
matrix one Python minor at a time only after clean resolution, import, kernel,
parity, memory, and cleanup tests on each claimed OS/CUDA pair. The extras may
remain mechanically resolvable on another base-compatible Python, but
resolution alone is not a support claim.

Documentation must say never to install both CuPy CUDA-major distributions in
one environment. The supported path uses `[ctk]` CUDA component wheels and
still requires a compatible NVIDIA driver. A driver-only wheel can work for a
subset but is not the supported VIPP install because missing component DLLs/
shared libraries can surface only when another CuPyX module loads.

The environment markers keep CUDA distributions out of macOS resolution. They
do not imply that `pip install napari-vipp[gpu-cuda12]` enables a GPU on macOS;
the UI and user guide must identify CUDA as unavailable and use CPU until a
separate Apple provider is admitted, without offering a CUDA install/repair
command there. Before each release, build an isolated environment
for every supported OS/Python pair, resolve the base package, and prove that the
macOS dependency graph contains no `cupy`, `cupyx`, or NVIDIA CUDA component.
For advertised Windows/Linux GPU targets, resolve the selected extra from a
clean environment rather than relying on a developer machine's CUDA state.

### 9.2 Reproducible development and user installation

Phase 1 adds supported setup scripts rather than a single cross-platform
`requirements.txt`, because CUDA-major and OS packages are intentionally
different:

- `scripts/setup_gpu_dev.ps1 -CudaMajor 13` for the primary native-Windows RTX
  5090 development environment;
- `scripts/setup_gpu_dev.ps1 -CudaMajor 12` and a shell equivalent for the
  compatibility tracks;
- `scripts/setup_gpu_dev.sh --cuda-major 13` and `--cuda-major 12` for each
  validated native-Linux track, using distinct environments;
- versioned constraint/lock inputs for each tested CUDA major plus the project
  `dev` dependencies; and
- a real-kernel/provider diagnostic command that exits nonzero when the
  environment is incomplete.

The scripts create a dedicated Python 3.12 virtual environment, never modify the
global interpreter, never install both CuPy CUDA-major distributions, and print
the exact environment and probe result. CUDA 13 is the first local development
track; CUDA 12 compatibility is required before public release. README commands
remain available for users who prefer manual setup.

For nontechnical users, the first public UX is guided installation: diagnostics
select exactly one compatible extra and present a copy button, short explanation,
restart requirement, and a verification action. A future `Install acceleration`
button may automate this only in a VIPP-owned/managed environment with explicit
confirmation, progress/logging, rollback guidance, and application restart. It
must not run `pip` silently inside an arbitrary active napari environment.

No Apple accelerator extra is named until an MPS/Metal or MLX provider passes
operation-level parity, packaging, memory, and performance gates on the M1 Max.

### 9.3 Platform behavior

- **CPU-only Windows/Linux/macOS:** base install, plugin import, workflow load,
  generated Python, and CPU execution work. Capability says GPU unavailable
  without an import exception.
- **macOS:** always reports `platform_unsupported` for the NVIDIA provider.
  `Auto` selects CPU as a normal policy decision. An unavailable CUDA choice in
  Custom mode follows visible/strict fallback policy; diagnostics do not show
  a CUDA install command or attempt a source build. Apple acceleration remains a
  separately labelled research path, not an implied CUDA replacement.
- **Native Windows x86-64:** supported for CuPy after import, device enumeration,
  context creation, one real kernel, and required CuPyX module probes pass.
  Diagnostics distinguish missing package, incompatible wheel, driver failure,
  missing runtime component, no device, and kernel-probe failure.
- **Linux:** supported only for the named NVIDIA/CUDA-compatible glibc
  distributions and architectures in the release matrix, using the same probe
  and both CUDA-major CI tracks. Container use must expose a compatible
  driver/device. An unvalidated distribution remains CPU-only; local source
  compilation does not expand the published support matrix.
- **WSL2:** report WSL explicitly in provenance and diagnostics; use Linux
  wheels and document host-driver prerequisites. WSL2 is useful secondary
  coverage, not a substitute for the required native-Windows CuPy path.
- **Incompatible environments:** never surface a raw import traceback as the
  primary message. Show the classified cause, selected extra, installed package
  versions, and a copyable install/repair command. Raw CUDA details remain in
  expandable diagnostics.
- **CI without GPUs:** do not install GPU extras for required CPU jobs. Mock
  provider tests exercise selection, segmentation, cleanup, and import safety.
- **Plugin discovery:** npe2 validation and `import napari_vipp` must prove that
  `cupy`, `cupyx`, and optional implementation modules are absent from
  `sys.modules`.

Phase 1 may register and exercise a hidden/developer-only cuCIM implementation
against the pinned source build so Subtract Background can prove the
multi-library architecture. Do not publish a cuCIM user extra or advertise its
operations before the packaging/admission pass. The
[native-Windows source evaluation](cucim-windows-source-evaluation.md) now
provides a reproducible `v26.06.00` CPython 3.12 `win_amd64` skimage wheel,
selected upstream tests, and primitive-level operation benchmarks. Rolling ball,
Canny, labeling, Otsu, and region properties showed promising primitive speed;
the measurements did not reproduce every VIPP wrapper behavior and therefore do
not constitute production-node parity or promotion. Gaussian, ordinary median,
Sobel, binary morphology, and cuCIM Richardson-Lucy did not justify replacing
their CuPy paths in the measured fixtures. The build required small downstream
packaging/NumPy adaptations and excludes native Clara I/O. The admission pass
must still
validate supported Linux targets, another Windows tier, memory, cancellation,
clean-install/JIT cost, packaging/CI maintenance, and region-table schema
adaptation. cuCIM implementations use the CUDA/CuPy runtime array domain when
verified; their optional package lifecycle remains independently removable.
macOS remains ineligible for cuCIM CUDA because source compilation cannot supply
the missing runtime.

The [cuCIM Windows port plan](cucim-windows-port-plan.md) is the delivery plan
for making those Windows artifacts maintainable and for attempting native
Clara/`CuImage` support. It deliberately runs alongside this provider-admission
plan: successful packaging does not waive operation parity, memory,
cancellation, provenance, or benefit gates. A feature-completeness/Clara review
starts soon after the first headless compute slice and records continue/defer/
stop evidence; implementation remains outside Phase 1 unless it becomes a
prerequisite for a VIPP I/O requirement.

## 10. UI/UX plan

### 10.1 Global controls

Add one global `Compute` selector (`CPU`, `Auto`, `Prefer GPU`,
`Custom`) to the main toolbar/settings and batch setup summary. New sessions
default to Auto; v3 workflows initially restore their historical CPU intent
until the user chooses otherwise. Add adjacent compact status such as
`Auto · RTX 5090`, `Prefer GPU · 3 GPU / 2 CPU`, or
`Custom · 2 GPU / 2 CPU`. If multiple usable devices exist, an advanced
device selector may choose the run device, but its index is session state rather
than portable workflow intent.

In the current responsive toolbar the compact selector belongs immediately
before Settings and is mirrored in the Settings overflow menu so it remains
reachable at narrow widths. Compute controls must not displace the existing
preview/zoom state or make the main toolbar wrap unpredictably.

`Find fastest pipeline` appears beside the selector only in Custom mode. It
is disabled with an explanation until the scheduled graph and workload
descriptors are known. An advanced `Fail if a selected GPU cannot run` switch
changes visible fallback into fail-closed preflight for Custom intent.
Prefer GPU always uses visible fallback; strict Prefer-GPU requests are invalid.
The ordinary interactive default keeps fallback enabled and conspicuous.

### 10.2 Node and run explanations

The inspector Compute section is read-only in CPU, Auto, and Prefer GPU modes.
In Custom mode, implemented nodes gain a preference dropdown and eligible
nodes gain `Benchmark node`. Unimplemented scientific nodes show CPU without a fake selector;
source/writer infrastructure is marked `Host` or left unbadged.
The dropdown offers `Auto for this node`, `CPU`, and one `GPU · <library>`
choice per validated library. It adds `Best GPU` only when at least two distinct
libraries are declared. Exact pins are not normal choices; a loaded current pin
is shown as a temporary `Advanced pin · <library>` entry. Pins excluded by the
active admission setting and unknown IDs are visibly marked `unavailable`.
Selecting a normal choice removes that temporary entry.
The section can report:

- `Will run with CuPy (GPU)` or `cuCIM (GPU)` with runtime/device and predicted
  benefit;
- `Will run on CPU` with one primary reason such as small workload, unsupported
  dtype, no validated threshold, memory cap, unsupported operation, or missing
  dependency;
- `Choice pending; input not available yet` before shape/dtype are known;
- `Selected implementation unavailable` with exact dtype/parameter/package/
  platform reason and fallback/repair action;
- `Fallback used` with reason and a link to run details.

Every calculated graph card may use one small header pill beside its title:
`CPU`, `GPU · CuPy`, `GPU · cuCIM`, or `Fallback · CPU`. The existing preview
corner remains available for processing state. Before calculation the pill is visually
distinct and says `Planned`; after an accepted run it says/represents `Used`.
Tooltips expose the device, implementation version, policy/benchmark source,
and reason. Eligibility alone must never look like actual execution. Unsupported
graph regions can be highlighted in a preflight explanation view; the default
graph remains uncluttered.

The run summary must let a user answer directly:

- **Which implementation will this node use?** Show CPU/CuPy/cuCIM, runtime,
  device, preference source, and segment.
- **Why did it run on CPU?** Show the stable reason translated to plain text.
- **Was fallback used?** Show a persistent run-level warning and affected nodes.
- **Is the result still valid?** State the promoted parity policy and that CPU
  fallback used the reference implementation; distinguish this from a failed or
  canceled result.
- **What should I install?** Recommend exactly one matching extra and explain
  that a supported NVIDIA driver is still required.

### 10.3 Benchmark presentation

`Benchmark node` shows candidates, validation status, cold time, warm robust
median/range, peak accelerator memory, and whether transfers were included. The
result has `Use fastest`, `Keep current`, and `View details`; it never changes a
preference merely because the benchmark finished. `Use fastest` is one undoable
authored-preference edit. If its local evidence later becomes stale, the choice
remains active but the `fastest` label is removed and `Rebenchmark` is offered.

The current experimental node dialog shows the candidate implementation, warm
median, CPU speedup, parity result, peak memory, and explicit `Use fastest`
action. Cold/range/transfer-resident detail, a dedicated `View details` surface,
and visible stale-evidence/rebenchmark state remain presentation hardening.

Benchmark progress must distinguish the whole analysis from the provider call
currently executing. Operations with real internal boundaries should report
those boundaries: CPU and cuCIM background subtraction report completed spatial
planes, and a GPU plane is complete only after its output assignment has been
synchronized. The per-operation indicator resets for every parity, cold,
warmup, or paired-timing invocation. A monolithic provider call must not invent
percentage progress; its indicator may pause at the call boundary until the
synchronized call returns, with text explaining that work is still continuing.

`Find fastest pipeline` shows the current and proposed implementation assignment,
estimated/measured total, transfer/runtime boundaries, peak memory, stale or
estimated nodes, and any excluded candidate. `Apply choices` is a separate
confirmation and one undoable action. The current CPU/Best GPU/library/exact
choice is displayed as the starting assignment, not treated as a constraint.
A separate per-node lock is the sole user-authored instruction to preserve a
backend during this search, and an applied winner remains unlocked unless the
user locks it separately. An `Auto for this node` choice has no explicit backend to
preserve and cannot be locked until the user selects a per-node choice. Before
analysis, the UI summarizes how many nodes are
unlocked, locked, or have only one scientifically eligible implementation. If
the globally optimal plan chooses a slower isolated node to preserve residency,
the explanation states that plainly.
Benchmark progress is cancellable and never publishes a writer/batch output.
The analysis dialog has two determinate levels where the provider permits it:
an overall bar across pipeline stages/nodes and a current-operation bar naming
the node, implementation, measurement phase, and round. A monolithic
synchronized CPU/GPU library call can only emit progress before and after the
call; the UI must describe that limitation rather than imply that an unchanged
percentage is a hang.

The current experimental update replaces the ambiguous whole-analysis override
with explicit per-node optimizer locks. It provides current/proposed rows with
the portable preference that would be authored, locked/excluded status,
validated end-to-end totals and the paired lower confidence bound when an
assignment changes, plus a successful current-assignment result when it already
wins. Transfer direction,
current free/total VRAM, cap/reserve, peak candidate memory, and graph liveness
are enforced by the coordinator but are not yet exposed as detailed review-table
columns. Per-node refusal drill-down, saved local
evidence inspection, and residency-boundary explanations remain Pass 4 UX
hardening; absence of those explanations does not relax any admission gate.

When a candidate is excluded only because of dtype, the review should name the
exact unsupported region. For native-`uint16` Gaussian it may suggest reviewing
an explicit **Convert Dtype** to `float32`, with a link to the dtype warning and
an explanation of `Preserve` versus the node's default `Rescale`; it must not
imply that conversion is scientifically neutral or silently edit the graph.

### 10.4 Errors, memory, and progress

- Preflight OOM: show estimate, usable memory under cap/reserve, and options to
  reduce workload, lower retained outputs, adjust cap, use Auto/CPU, or enable
  visible fallback.
- Runtime OOM with fallback: show `GPU ran out of memory; this segment was
  recomputed on CPU` and record it. Without fallback, no partial result is
  presented as current.
- On discrete CUDA hardware, host cache/RAM and accelerator pool/VRAM appear as
  separate rows in one Execution/Memory panel. Show device-wide free/total,
  VIPP live and pool-reserved bytes, effective cap/reserve, predicted peak, and
  measured out-of-pool delta. On unified-memory hardware,
  show one shared/unified-memory row plus the provider allocation/working-set
  budget; never add RAM and nominal VRAM together.
- During monolithic kernels, progress remains indeterminate and cancellation
  wording is honest. RL/RL-TV show iteration progress.
- Pipeline-optimizer timeout messages say that the analysis is inconclusive,
  identify the active stage/node and elapsed wall-clock limit, and state that no
  settings changed. They distinguish reusable complete exact records from the
  discarded partial timings of the interrupted node and recommend a longer
  limit when the user wants to finish the comparison. They never describe
  budget exhaustion as an optimal result.
- Batch workspace shows requested mode, device, current item/node, fallback
  count, and manifest link. One item fallback does not silently change later
  items; each is planned from the same request and current memory snapshot.

## 11. Scientific parity framework

### 11.1 Common rules and metrics

The CPU pipeline path—not an independently simplified benchmark function—is
the reference. Tests call through the same prepared-node and metadata/grid
contracts used by production. GPU acceleration must preserve shape, dtype,
axis/channel semantics, output state, and all public defaults.

For arrays `reference` and `candidate`:

```text
max_abs = max(abs(candidate - reference))
NRMSE = ||candidate - reference||2 /
        max(||reference||2, sqrt(N) * absolute_floor)
```

Also report mean absolute difference, finite/non-finite masks, min/max, and
operation-specific scientific metrics. Tolerance checks require both NRMSE and
maximum-absolute gates; a good aggregate cannot hide a local defect.

Common fixture requirements:

- deterministic seeds and read-only inputs;
- zeros, constants, impulses, ramps, high dynamic range, boundary-touching
  structures, signed values, and finite-value extremes;
- NaN, +Inf, and -Inf behavior matching the CPU path or explicit exclusion from
  GPU support after a complete (not sampled) preflight check;
- scalar arrays whose trailing dimension is 3/4, explicit RGB/RGBA channel
  axes, CZYX/TCZYX, and reordered explicit axes;
- 2D/3D minimum, representative, and extreme supported shapes/parameters;
- deterministic synthetic ground truth plus representative calibrated real
  scientific datasets;
- repeatability across two runs and all supported provider/runtime tracks.

The only initial precision policy is `scientific-default-v1`: no TF32, reduced
precision, mixed precision, fast-math algorithm switch, or changed accumulation
policy. RL/RL-TV retain their current float32 working/output and float64 PSF-sum
normalization behavior. Additional precision modes require separate visible
intent, cache identity, parity policy, and reviewed scientific evidence.

Every implementation declares scientific dtype/value contracts per input and
output port, not one coarse `supported_dtypes` tuple. The declaration includes
accepted public dtypes, value-domain constraints, internal/accumulation dtype,
output-dtype rule, conversion/rounding/overflow policy, and a non-finite policy
(`preserve`, `clean exactly`, `reject`, or `CPU decision`). This is required for
multi-input RL, labels-plus-intensity operations, and dtype-changing outputs.

### 11.2 Operation policies

#### Rolling-Ball/Subtract Background — `background-dtype-parity-v2`

- Implement and test both nodes because they share the expensive background
  estimator. The production adapter—not raw `skimage.restoration.rolling_ball`
  or a benchmark-only approximation—is the CPU reference.
- Preserve the default 3× smoothing and `disable_smoothing`, light-background
  inversion, `clip_negative`, 2D/3D spatial blocks, leading/channel axes, bool
  identity/zero behavior, non-finite behavior, and `_restore_numeric_dtype`
  rounding/clipping exactly.
- Target `uint8`, `uint16`, `float32`, and `float64` independently. The current
  v2 admission requires bitwise-exact public output for `uint8`/`uint16`.
  `float32` requires exact dtype/shape, finite/NaN/+Inf/-Inf masks, the zero
  mask, and sign bits at zero, with NRMSE `<= 2e-6` and
  `max_abs <= 1e-6 + 2 * eps(float32) * max(1, input_peak, reference_peak)`.
  Float64 remains CPU-only until a separately versioned policy is evidenced.
- The bounded float32 policy is explicit because SciPy and CuPyX use different
  accumulator order in the smoothing stage; a few-ULP intermediate difference
  can become a large ULP count near zero after subtraction. Global ULP remains
  diagnostic, while the aggregate and maximum-absolute gates are normative.
- The existing cuCIM results are primitive feasibility/performance evidence
  only: they omit important VIPP wrapper semantics. Promotion requires the
  prepared production-node path and complete adapter benchmark/parity matrix.
- Prefer a cuCIM implementation when admitted; retain an extension point for a
  custom CuPy implementation if it is easier to package or wins on a validated
  workload. The CUDA/CuPy array runtime remains common to both.

#### Median — dtype-specific `median-cupy-*-v1`

- Initial targets: `uint8`, `uint16`, finite `float32`, and `float64`, current
  slice-wise YX semantics, odd declared sizes in the validated range, current
  channel-axis handling, and current reflect boundary convention. Promote each
  dtype independently.
- Required parity: identical dtype/shape and exact finite values for every
  promoted integer or float case because median selects existing samples; a
  tolerance may not hide a different rank/boundary result. Check signed-zero
  bits explicitly. Do not treat the spike's two exact results as proof for all
  cases.
- Explicitly test repeated values/ties, signed zero, extrema, each boundary,
  leading stack axes, channels, and every promoted odd kernel size.
- Float non-finite inputs and bool resolve to CPU until exact behavior is
  validated.

#### Gaussian — dtype-specific `gaussian-cupy-*-v1`

- Preserve `scipy.ndimage.gaussian_filter` default reflect boundary, sigma=0
  copy behavior, slice-wise YX axes, 3D active-axis selection, excluded channel
  axis, output dtype, and current bool conversion (bool is not initially
  promoted).
- The float32 gates are NRMSE `<= 2e-6` and
  `max_abs <= 1e-6 + 5e-6 * max(abs(reference))`, plus exact shape/dtype/state.
- Target `uint8`, `uint16`, `float32`, and `float64` independently. Integer/bool
  promotion requires exact public output after the current dtype behavior;
  float64 gets its own tighter evidence-based tolerance and never inherits the
  float32 gate by default.
- Test zero/anisotropic/maximum sigma, narrow axes, impulses at every boundary,
  leading blocks, explicit channel axes, and scalar trailing 3/4 axes.
- Non-finite and unpromoted dtype regions remain CPU until separately validated.

#### Sigma Filter — `sigma-dtype-parity-v1`

- Preserve the clean-room CPU operation, not merely a similarly named denoiser:
  Fiji's radius plateaus and circular footprint, fixed row-major offset order,
  nearest/clamped YX borders, immutable source, independent channels/leading
  planes, float32 sample/square boundaries, ordered float64 accumulation,
  center-relative inclusive selection, exact minimum-count ceiling, both
  fallback modes, and dtype restoration are all scientific behavior.
- Initial CPU and GPU types are finite native-endian `uint8`, `uint16`, and
  `float32` only. Non-native byte order fails closed before device transfer.
  Radius is 0.5–10, sigma width is finite/non-negative, minimum fraction is
  0–1, outlier awareness is boolean, and the optional channel axis must leave
  two resolved YX axes. Float32 requires complete finite extrema and a
  square-safe magnitude. ROI/mask behavior is absent in version 1.
- Require identical shape/dtype. Unsigned output must be bitwise exact after
  the node's float32 cast, clip, and half-up restoration. Float32 requires
  equal finite/zero/sign masks, NRMSE `<= 2e-6`, and
  `max_abs <= 1e-6 + 4*eps(float32)*max(1,input_peak,CPU_peak)`. ULP remains
  diagnostic. Adversarial fixtures must separately prove the same inclusive
  selection and fallback branches; an aggregate tolerance cannot excuse a
  branch change.
- Freeze the deliberate numerical stabilizations: use exact
  `ceil(footprint_count * minimum_fraction)` and clamp every negative computed
  population variance to positive zero. Do not copy the plugin's approximate
  ceiling or NaN-dependent negative-variance behavior merely to call the result
  exact Fiji parity.
- The CuPy provider uses a fused two-pass `RawKernel` with `--fmad=false`,
  precise divide/square-root, explicit subnormal-safe float conversion, bounded
  synchronized row tiles, and no image-sized sliding-window tensor. Attribute
  it to CuPy, not CuPyX. Include any contiguous input/axis-restoration staging in
  the memory estimate and timing.
- Pin independently executed official-plugin fixtures and their source/class,
  ImageJ jar, generator, and harness hashes. The source-current real-device
  parity/lifecycle record passed, so the exact region is now public in Custom
  and is a reviewed Auto default where its current gates pass. Raw matrix
  timings do not teach Auto. Completed-run history follows the reviewed-default,
  same-surface CPU-exploration, then 1.20x/20-ms selection sequence. Future
  widened regions repeat this gate; they do not inherit admission from the
  current matrix.

#### Otsu — `cupy-otsu-threshold-exact-v1`

- The implemented CuPy adapter reproduces VIPP's complete finite-value
  histogram, bin/count policy, integer
  offset/range preservation, bool identity, slice/stack scope, and final mask;
  raw `cucim.skimage.filters.threshold_otsu` scalar parity is insufficient.
- Compare the final public dtype/mask exactly and record the threshold only as
  intermediate provenance. Empty/all-non-finite and constant regions follow the
  CPU error/result contract exactly.

#### Canny — `cupyx-canny-edges-exact-v1`

- The implemented CuPy/CuPyX adapter preserves VIPP's BT.601 channel reduction,
  plane-wise spatial semantics,
  ordered low/high quantile thresholds with `use_quantiles=True`, sigma/boundary
  behavior, leading blocks/channels, and exact boolean output. Adding an
  absolute-threshold mode would be separate scientific and schema work.
- Only complete parameter regions whose final edge mask is exactly equal are
  public candidates. Raw cuCIM Canny was evaluated and rejected because
  adversarial final masks diverged; a raw default fixture cannot establish this
  contract.

#### Connected components — `cupyx-connected-components-v1`

- Implemented as an exact-profile public candidate in Phase 5. The CPU
  operation remains authoritative: nonzero foreground, SciPy rank-one/full-rank
  connectivity, independent leading-block labeling with IDs restarted at one,
  native shape-preserving `int32`, and existing overflow/error behavior.
- The admitted GPU region is boolean 2D/3D, face/full connectivity, fewer than
  2,147,483,646 elements in each spatial block, and the pinned CuPyX/runtime/
  CPU-reference environment. Numeric mask conversion, 1D, oversized blocks,
  and unqualified environments retain CPU without an implicit cast or axis
  reinterpretation.
- `labels-bitwise-int32-v1` requires identical public label IDs/order. The
  CuPyX adapter passes directly and therefore adds no canonicalizer. Any future
  cuCIM or other provider with an equivalent partition but different IDs must
  canonicalize deterministically and include that work in parity, memory, and
  timing.
- `cupyx-connected-components-memory-v1` counts the full one-byte bool input
  and four-byte output plus seven bytes of workspace for one active block. A
  single-block workload is therefore 12 bytes per element. Progress is truthful
  only after a synchronized leading block; a single plane/volume remains one
  atomic call with cancel-after-return behavior.

#### Measurements — `measurements-*-v1`

- Implemented for the basic morphology and basic morphology-plus-intensity
  schemas as `cucim-measure-objects-basic-v1` and
  `cucim-measure-objects-intensity-basic-v1`. The promoted region preserves
  exact `TableData` column names/order, public scalar/storage types, row order,
  calibration/units, empty tables, sparse IDs, and label/count overflow checks.
- The provider terminates its device segment at a mandatory typed host-table
  finalizer. The packed result crosses to the host and the private CUDA scope is
  cleaned before public `TableData` construction; finalizer timing and staging
  memory are included in the optimizer cost.
- Extended-property schemas remain authoritative CPU regions. Value parity for
  a convenient `regionprops_table` subset is not sufficient to promote them;
  every advertised column and downstream table contract requires evidence.

#### Ordinary Richardson–Lucy — `rl-cupy-f32-v1`

- Mirror the production prepared pipeline path: constant 0.5 initialization,
  zero-extension `same` convolution, PSF cleaning and optional float64-sum
  normalization, `filter_epsilon`, input/output clipping, input-scale
  preservation, per-block 2D/3D behavior, and float32 output.
- Freeze the prepared pipeline/native loop as the authoritative reference for
  UI, headless workflow, batch, generated Python, and GPU parity. Preserve the
  legacy direct helper's scikit-image behavior when `progress is None` as a
  separately tested API until a versioned migration deliberately unifies it;
  Pass 6 must not silently move a caller between the two semantics.
- The implemented exact-workload benchmark gate requires matching shape and
  `float32` dtype, identical finite masks with completely finite output, NRMSE
  `<= 2e-6`, and
  `max_abs <= 1e-6 + 5e-6 * max(abs(reference))`. Maximum float32 ULP distance
  is recorded as a diagnostic but is not a separate pass gate. The exact
  validated region is publicly visible on this branch; wider MSE, flux, and
  point/line/dim-feature recovery evidence is still required to broaden the
  region or make stronger release/platform claims.
- Test 2D/3D, leading blocks, iterations 1/2/25 and long-run boundaries on a
  small fixture, PSF sizes/support, normalized and deliberately unnormalized
  valid PSFs, parameter extremes, zeros, negative clipping modes, and cleaned
  NaN/Inf inputs.
- The public-candidate initial GPU region is deliberately narrower than the
  public CPU surface: finite float32 Image/PSF, odd PSF extents, the default-safe
  options, `filter_epsilon == 1e-8`, and no more than 25 iterations. The
  unchanged `1e-12` CPU default, other epsilon values, and longer runs remain
  CPU-only until a versioned numerical study clears the same parity policy.

#### RL-TV — `rl-tv-cupy-f32-v1`

- Implemented as an exact-profile public candidate in Phase 2C. Preserve the
  current formula, minus
  sign, denominator placement/floor,
  central-difference `np.gradient` convention, zero-extension convolution,
  constant initialization, epsilon behavior, clipping, and lack of physical
  spacing in the TV stencil. Acceleration may not introduce reflect padding,
  observed initialization, an adjoint stencil, implicit PSF preparation, or new
  defaults.
- With `tv_regularization=0`, GPU RL-TV meets the same GPU/CPU RL gate and must
  remain exactly equivalent to the corresponding ordinary GPU RL provider for
  the same admitted call.
- Use the existing deterministic phantom harness and preserve its production
  checks: finite/non-negative output, denominator-floor diagnostics, feature
  retention, PSF centering/sampling sensitivity, 2D/3D boundary structures, and
  current default behavior.
- Positive TV uses the versioned development screen NRMSE `<= 0.005` and
  `max_abs <= 1e-6 + 0.005 * CPU_peak`, plus no more than 0.5 percentage point
  change in point, thin-line, or dim-line recovery and no more than 0.5%
  relative change in MSE/border MSE/flux versus CPU for maintained promotion
  fixtures. This separate gate is required by the nonlinear recurrence and
  must not be applied to ordinary RL or lambda-zero RL-TV.
- Real-data gate: bead data and at least three calibrated biological datasets
  spanning sparse points, dim structures near bright signal, anisotropic 3D,
  and boundary objects, with blinded review and no systematic loss.

Operation tolerances are versioned policy data. Relaxing one is a scientific
change requiring evidence and review, not a test-maintenance edit.

## 12. Testing and CI matrix

### 12.1 Required checks on every PR

- Full CPU-only test suite with no GPU packages installed on native Windows,
  macOS, and Linux. Each OS builds the wheel, installs it into a clean
  environment, validates the napari manifest, imports the plugin and generated
  Python, and runs the tests.
- Dependency-resolution assertions prove that the macOS base package and
  platform-marked GPU extras resolve without installing CuPy or NVIDIA CUDA
  components. Windows/Linux GPU-extra resolution is checked in isolated
  scheduled environments for every supported Python/CUDA-major pair.
- `test_plugin_contract.py`, npe2 validation, generated script import, and a
  subprocess assertion that CuPy/CuPyX/optional modules were not imported.
- Compute-contract parsing/JSON round trips for
  CPU/Auto/Prefer-GPU/Custom, retained dormant per-node preferences,
  visible/strict validation, benchmark fingerprints, and stable reason-code
  tests.
- Mock capability/runtime/library tests for absent package, failed import, no device,
  driver/runtime mismatch, failed real-kernel probe, multiple devices, and
  unhealthy provider refresh.
- Fake-device planner tests for linear, mixed, branch, fan-out, join,
  multi-input, multi-output, cached boundary, skipped/manual, retained/pinned,
  dirty-subgraph, same-runtime CuPyX/cuCIM residency, cross-runtime boundaries,
  and unavailable strict Custom choices. The fake array is opaque and raises
  if ordinary NumPy code coerces it outside the runtime.
- Revision-keyed `ArrayFacts` tests cover complete versus sampled facts,
  propagation, invalidation, scan-cost accounting, non-finite exclusion,
  content-sensitive policies, and conservative CPU decisions when facts cannot
  be proven before execution.
- Benchmark tests use an injected deterministic clock/cost model, not sleeps.
  Cover parity-failing candidate quarantine, randomized paired samples and the
  versioned confidence calculation, node/pipeline staleness scopes,
  cancellation/time budget, no writer/cache mutation, local node choice, and a
  whole-graph case where independent node winners lose to a resident plan.
- Deterministic memory/liveness estimates, cap/reserve failures, classified OOM,
  one-time retry, no retry loop, unclassified-error failure, and cleanup in
  success/error/cancel paths, for discrete and simulated unified memory.
- CPU regression tests proving the prepared-node/executor refactor produces the
  same arrays, metadata, progress, cache pruning, manual state, and errors.
- Workflow/generated-code/batch tests appropriate to the active pass, including
  workflow-v4 migration in Pass 4 and generated/batch provenance hashes in
  Pass 8.
- Ruff and architecture-boundary checks.

These tests use a fake runtime or NumPy-backed device adapter. They validate
control flow, not scientific CuPy parity.

### 12.2 Scheduled/manual real-GPU validation

| Dimension | Minimum matrix |
| --- | --- |
| CPU/package OS | Native Windows, macOS, and Linux required on every PR; include the supported macOS architectures in release CI. |
| Real NVIDIA GPU OS | Native Windows and supported Linux scheduled; WSL2 manual/release candidate until a stable runner exists; macOS explicitly excluded because current CUDA has no macOS target. |
| CUDA major | Separate CUDA 12 and CUDA 13 environments; never both CuPy wheels together. |
| Runtime/library | Supported CuPy major and lockfile endpoints; every advertised cuCIM build digest and required module/interoperability probe. |
| Packaging | Clean base and wheel install on all three OS families; clean platform-marked extra resolution on all three; real CuPy import/kernel/submodule probe on every advertised Windows/Linux target. |
| Device tier | Minimum supported VRAM/tier, a mid-tier device, and a higher-tier device; include one laptop/WDDM system. |
| Initial available hosts | Native Windows/CUDA 13 RTX 5090 primary; at least one Windows RTX 40-series laptop secondary; Ubuntu 22.04/24.04 x86-64 native before Linux advertisement; WSL2 secondary; M1 Max CPU/package and later Metal feasibility. |
| Scientific parity | Every promoted operation, dtype, dimension, parameter boundary, deterministic fixture, and real-data gate. |
| Performance | Cold diagnostic separately; randomized warm synchronized end-to-end and resident-chain runs with robust medians/confidence; production-node adapters rather than raw optional-library primitives. |
| Memory | Peak measurements, near-cap runs, branch/fan-out, intentional preflight failure, real OOM recovery, and post-item leak checks. |
| Cancellation | RL/RL-TV iteration stop, monolithic wait-and-cleanup, transfer boundary, segment, and batch item. |

Every PR remains mergeable from CPU-only required checks. Scheduled GPU jobs
gate operation promotion and releases, not ordinary documentation/UI PRs.
Stable dedicated hosts compare robust medians against a rolling reviewed
baseline; performance regressions alert at 15% and block promotion/release only
after rerun excludes thermal or shared-host noise. Scientific parity failures
always block.

## 13. Implementation passes

Each pass is a separately reviewable integration unit made from the coherent
commits in section 14; a PR is optional when requested. Every pass has a rollback
that leaves the CPU path usable. File lists are ownership boundaries, not blanket
permission to rewrite adjacent code.

### Delivery phase map

- **Phase 1 — headless foundation and first vertical slice:** Passes 0-3.
  Freeze the then-current CPU/Auto/Custom and benchmark contracts (before
  Prefer GPU was added); unify headless/device
  execution behind one Qt-free service; build the fake-tested device-resident
  runtime; create the
  reproducible CUDA 13 development environment/doctor path; and implement
  Rolling-Ball/Subtract Background, median, and 2D/3D Gaussian. No toolbar,
  workflow-schema, batch, or generated-Python behavior changes yet.
- **Phase 2 — interactive use and deconvolution:** Pass 4 plus the RL/RL-TV
  operation work in Passes 6-7. Ordinary RL and RL-TV are implemented
  headlessly. Add toolbar mode, Custom node choices,
  badges, review-first node and whole-pipeline benchmark UI,
  diagnostics/install guidance, RAM/VRAM presentation, the minimal workflow v4
  compute-intent block plus canonical hash and atomic reader/writer preservation
  so accepted choices persist, and
  a small explicitly scoped wave of inexpensive residency-bridge nodes in
  parallel where file ownership is disjoint.
- **Phase 3 — segmentation/measurement wave and cuCIM completeness review:**
  Otsu, Canny, Connected Components, and the basic production-schema
  Measurements slice are implemented. Time-box the full cuCIM/Clara
  Windows investigation and Apple M1 Max provider
  feasibility.
- **Phase 4 — durable execution surfaces (implemented 2026-08-04):** Passes 5
  and 8 add batch, generated Python/CLI overrides, effective-config/artifact
  hash and provenance integration, export sidecars, nested progress,
  cooperative cancellation,
  structured OOM/fallback evidence, and cleanup-gated publication using the
  already frozen workflow v4 compute block and canonical hash.
- **Phase 5 — broad reasonable-node coverage and release hardening:** Passes
  9-10 promote remaining filtering, pointwise operations, morphology,
  segmentation, label cleanup, colocalization, and other families where a
  runtime/library implementation is scientifically faithful and operationally useful.

Phase 1 is complete only when all of these historically scoped conditions are
true:

- the phase-era headless contracts support CPU/Auto/Custom, `best_gpu`,
  library/exact per-node preferences, visible/strict fallback, multiple
  implementation candidates, and same-runtime CuPyX/cuCIM interoperability;
- old callers remain CPU-compatible and CPU execution imports no accelerator;
- every real headless/device run uses one Qt-free execution service, and the
  existing synchronous and worker application paths have explicit CPU
  behavioral-parity coverage; Pass 4, not Phase 1, routes those interactive
  application paths through the service;
- an opaque fake runtime executes transactional segments across linear,
  branching, joining, cached/manual, cancel, OOM, and fallback cases without a
  device value escaping;
- initial production adapters have explicit per-port dtype/parameter policies,
  with `uint8`, `uint16`, and float32 treated as first-class microscopy targets
  and float64 promoted only under its own evidence;
- every advertised region passes production-CPU parity, read-only input,
  memory, cleanup, cancellation, and provenance tests; unsupported regions make
  typed CPU decisions with no silent lossy cast;
- real RTX 5090 execution passes at least one finite-float32 region for each of
  `rolling_ball_background`, `subtract_background`, `median_filter`,
  `gaussian_blur`, and `gaussian_blur_3d`; Phase 1 cannot complete by advertising
  zero regions, while integer/float64 promotion remains independently gated;
- headless node benchmarking validates parity before timing, and the pipeline
  optimizer demonstrates one H2D and one D2H transfer for a compatible
  Background → Gaussian → Median CUDA/CuPy chain under the explicit Phase 1
  developer-experimental flag;
- result-cache identity and benchmark-profile identity are separate, and
  benchmarking mutates neither live output nor scientific caches;
- a dedicated CUDA 13/Python 3.12 environment can be recreated and passes the
  doctor/probe plus RTX 5090 parity/smoke/benchmark suite, including measured
  peak/pool/out-of-pool memory, post-run live-allocation checks, and one real
  classified-OOM cleanup/recovery smoke; the checksum-recorded cuCIM wheel is
  installed/probed in this same VIPP environment; at least one RTX
  40-series laptop supplies secondary evidence before Phase 2 Auto policy is
  presented as broadly calibrated; and
- cuCIM background remains explicitly experimental/unavailable wherever its
  package is not validated; no capability is falsely advertised.

### Pass 0 — Contract stabilization

**Depends on:** nothing.
**Owns:** `core/compute.py`; new `core/compute_specs.py` and
`core/compute_policy.py` contract shells; `test_compute.py`; new focused
`_tests/test_compute_specs.py` and `_tests/test_compute_policy.py`; this
plan/spike documentation updates. Avoid `pipeline.py`
unless a tiny read-only association helper is indispensable, because it has
active unrelated edits.

**Public contracts:** `ComputeRequest`, `ComputeEnvironment`, support/decision
reason enums, `OperationComputeSpec`, `WorkloadDescriptor`, `MemoryEstimate`,
`NodeComputePreference`, `BenchmarkRecord`, `NodeExecutionDecision`,
`ExecutionPlan`, `ScientificResultKey`, `CacheAdmissibility`,
`BenchmarkRecordKey`, and transient `ExecutionReport` data shells, JSON
serialization, and the then-current CPU/Auto/Custom semantics, before Prefer
GPU was added. Separate runtime/array domain
from implementation library and model per-port public/internal dtype/non-finite
policies. Add no accelerator callables and advertise zero production GPU
operations.

**Tests/documentation:** strict parsing, immutable/JSON-safe values, retained
node preferences, Auto CPU is not fallback, visible/strict Custom behavior,
stale benchmark fingerprints, declaration validation, and import safety. Fix
the spike so CPU resolves before capability detection and imports no optional
package. Update architecture docs to point to the new contracts.

**Migration:** preserve compatibility helpers around the spike's
`BackendCapability`/`select_compute_backend` only if callers/tests need them;
mark them as operation-local spike APIs. No workflow or batch schema change.

**Acceptance/rollback:** full CPU suite and plugin import pass without CuPy;
capability list remains empty. Rollback removes new unused contracts without
affecting execution. **Still disabled:** all GPU execution and UI controls.

**Parallelism:** after its contracts are merged/frozen, Pass 1 substrate, Pass
2 background science fixtures/adapter work, and Pass 9 platform validation can
start in parallel with disjoint files.

### Pass 1 — Headless execution, benchmarking, and development substrate

**Depends on:** Pass 0.
**Owns:** new `core/compute_registry.py`, `core/device_execution.py`,
`core/node_execution.py`, `core/compute_benchmark.py`,
`core/compute_diagnostics.py`, `core/gpu/__init__.py`, and
`core/gpu/cupy_runtime.py`; new focused registry/device/node/benchmark/
diagnostic tests; minimal coordinated changes in `core/pipeline.py` and
`core/execution.py`; and reproducible developer setup/doctor scripts plus
Python-3.12 CUDA-13/CUDA-12 constraint files. No production operation is owned
by this pass.

**Public contracts:** a runtime/array-domain protocol distinct from
implementation-library adapters; lazy CUDA/CuPy discovery; opaque device
values; segment/liveness and memory plans; private pool scope; transfer,
synchronization, cleanup, and prepared-node call/finalization seams; a Qt-free
node benchmark service; a graph-global assignment optimizer; transient per-node
decision/fallback/provenance reports; and headless result-cache admissibility.
Make every real device execution use this service in Pass 1 and prove
`PrototypePipeline.run()` CPU behavioral parity; Pass 4 then routes every
interactive path through it. The setup tool
has an explicit developer-only option to install a locally built, checksum-
recorded cuCIM wheel into the same VIPP environment; it never mistakes the
build script's temporary environment for the application environment.

**Tests/documentation:** an opaque fake runtime covers linear graphs, branches,
joins, multi-input/output nodes, cached/manual/retained values, cancellation,
OOM, one-retry fallback, and cleanup. Deterministic injected clocks and cost
models test node and pipeline selection without sleeps. Benchmark transactions
cannot publish output, mutate scientific caches, execute writers, or retain
device values. CPU regression covers arrays, metadata, dirty-cache hydration,
progress, pruning, and import without any optional accelerator package.

**Migration:** `PipelineRunRequest.compute_request` defaults to CPU so existing
headless callers behave identically. Auto is the default only for new
interactive sessions once Pass 4 supplies the UI. No workflow or batch schema
changes in this pass.

**Acceptance/rollback:** fake-runtime execution and benchmark/optimizer tests
pass, no device object reaches a public cache, and the dedicated CUDA 13/Python
3.12 environment can run a real CuPy kernel and diagnostic probe on the RTX
5090. The setup refuses mixed CuPy CUDA-major packages and provides exact repair
commands. Rollback selects the established CPU executor. **Still disabled:**
production GPU operations, toolbar controls, workflow/batch persistence, and
managed package installation.

**Parallelism:** setup/diagnostic work and fake-runtime work may overlap Pass 2
science-fixture work after Pass 0 with disjoint ownership. Only one agent edits
`pipeline.py` or `execution.py`.

### Pass 2 — Rolling-Ball and Subtract Background

**Depends on:** Pass 0 contracts; integration depends on Pass 1.
**Owns:** new `core/gpu/cucim_background.py`; background declarations in
`core/compute_specs.py`; operation-specific dtype, value, memory, and parity
policies; new `test_gpu_background.py`; and production-adapter benchmark
evidence. The CPU operations and base dependencies remain unchanged. The cuCIM
adapter stays developer-only until its packaging gate passes. This pass owns the
integration between the pinned builder's emitted wheel and Pass 1's explicit
experimental-wheel setup/doctor path.

**Public contracts:** separately versioned implementations for
`rolling_ball_background` and `subtract_background`, both using the pinned
cuCIM primitive through the common CuPy array domain. The adapter must preserve
VIPP's complete public behavior: optional 3x smoothing, light-background
inversion, clipping/subtraction, block and channel iteration, non-finite policy,
progress, dtype restoration, and metadata—not merely match a raw cuCIM call.
The support contract is per input/output port and initially targets validated
`uint8`, `uint16`, and float32 regions; float64 is admitted only under separate
evidence.

**Tests/documentation:** production CPU parity across dark/light background,
smoothing on/off, 2D/3D leading blocks, RGB/channel handling, radius boundaries,
non-finite values, read-only input, public dtype/shape, cancellation boundaries,
memory estimates, missing-cuCIM decisions, and one wider real microscopy set.
Raw primitive benchmark results are labelled feasibility evidence and cannot
stand in for these adapter tests.

**Migration:** existing workflow-v3 files load in CPU mode until the user opts
in. The scientifically and lifecycle-validated cuCIM background region is now a
normal public candidate on this branch. Its exact environment policy still
fails closed when the reviewed wheel/provenance is absent, and unsupported
workloads visibly remain on CPU.

**Acceptance/rollback:** every advertised dtype/parameter region clears the
scientific, memory, cleanup, cancellation, and provenance gates. That exact
region is now `public_auto_candidate`; packaging and broader platform support
remain separate claims, and Auto-performance evidence never bypasses the exact
environment gate. The current public gate is the recorded native-Windows CUDA
runtime API 13.2 (`13020`), driver API 13.3 (`13030`), and RTX 5090 (compute
capability 12.0) host. CUDA 12 is qualification-only and outside public
admission; secondary NVIDIA models remain provider-level qualification targets.
Rollback removes only the
background declarations/adapter. The production-adapter tests must run from the
same dedicated VIPP environment into which the recorded wheel was installed,
not only the builder's temporary venv. **Still pending:** a public cuCIM
installation extra and wider-platform qualification.

**Parallelism:** fixtures and adapter code can overlap Pass 1 in new files;
declaration/integration waits for Pass 1. One owner edits `compute_specs.py`.

### Pass 3 — Median, Gaussian, and the first headless optimizer

> **Historical pass gate:** this section records the constraints in force when
> Pass 3 was completed. Later passes added public toolbar controls and promoted
> scientifically and lifecycle-validated implementations within their exact
> admitted environment regions. The current status and region tables above
> supersede the historical visibility statements below.

**Depends on:** Pass 1. Median/Gaussian implementation proceeds independently
of experimental cuCIM packaging; only the combined resident-chain test waits for
Pass 2.
**Owns:** new `core/gpu/cupy_median.py` and `core/gpu/cupy_gaussian.py`;
median/Gaussian declaration and policy blocks; `core/compute_policy.py`;
versioned packaged policy records; generalized benchmark matrices; and focused
GPU filter and optimizer tests.

**Public contracts:** CuPyX implementations for `median_filter`,
`gaussian_blur`, and `gaussian_blur_3d`; distinct 2D slice-wise and true-3D
Gaussian IDs; per-port dtype/value policies; versioned host/device tier records;
and deterministic Auto/Custom planning. `uint8`, `uint16`, and float32 are
first-class validation targets. Median requires exact production parity in each
advertised region; Gaussian uses its reviewed operation-specific tolerances.
Float64 and non-finite behavior remain explicit per-operation regions, never an
implicit cast.

**Tests/documentation:** median kernel/footprint, channel/axis/boundary and dtype
matrices; Gaussian zero/anisotropic sigma, 2D/3D axes, dtype and tolerance
matrices; small-workload CPU selection; resident-neighbor GPU selection;
unknown-policy CPU selection; stale local benchmark handling; policy artifact
digests; and real-device cold, warm, transfer, resident, and peak-memory data.
The pipeline optimizer must prefer a single-transfer
Background → Gaussian → Median CUDA segment when that is globally fastest, even
when an isolated-node winner differs.

**Migration:** none. Auto/Custom can be passed only as headless runtime
requests until Pass 4; raw benchmark evidence is local and no workflow schema
changes.

**Acceptance/rollback (historical):** scientifically valid implementations
appeared as Custom candidates within the explicit Phase 1 developer request
regardless of the Auto threshold; at that pass boundary, public exposure still
required the packaging tier.
Developer Auto calibration considers only validated regions whose complete
segment clears the lower-confidence 1.20x and 20-ms gate, or a valid local
benchmark shows a statistically clear win. Small,
unknown, or out-of-domain work remains CPU. Rollback removes filter declarations
and policy records while leaving the substrate/background adapter intact.
**Still disabled:** public toolbar/node controls, batch GPU, RL, and RL-TV.

**Parallelism:** median and Gaussian science/adapter work can proceed in
parallel in separate modules; one integrator owns shared declarations/policy and
benchmark collection can run independently on named hardware.

### Pass 4 — Interactive compute controls, cache identity, and provenance

**Depends on:** Passes 1-3.
**Owns:** `core/execution.py`, persistent interactive cache/provenance mappings
on top of the transient headless records from Passes 0-1,
`ui/compute.py`, `ui/workers.py`, composition-only edits in `_widget.py` and
`_graph.py`; the compute-intent-only workflow v4 change in `core/workflow.py`
and its canonical workflow-hash/goldens/tests; minimal compatibility edits in
every existing workflow reader/writer (including batch, generated Python, and
export paths) so they parse and preserve the v4 block while still forcing CPU;
and focused execution/diagnostic/widget/graph tests. Coordinate carefully with
unrelated UI work and give each large composition file one owner.

**Public contracts and UI:** `PipelineRunResult.execution_report`, actual-
implementation host cache records, a session `ComputeRequest`, and one execution
service for formerly synchronous and background paths. At this historical pass,
add the then-current compact toolbar selector immediately before Settings with
`CPU`, `Auto`, and `Custom`; Prefer GPU is added by the later product update. New
interactive sessions default to Auto, and the selector is mirrored in Settings
when the toolbar collapses. `Find fastest pipeline…` exists only in Custom mode.
The inspector Compute group offers `Auto for this node`, `CPU`, one choice
per declared GPU library, and `Best GPU` only where multiple libraries compete,
plus `Benchmark node…`. Exact preferences remain an advanced/developer contract;
a loaded current pin remains visibly represented until replaced.
`Use fastest` and `Apply choices` each create one undoable authored-intent edit.
The current CPU/Best GPU/library/exact preference is an optimizer starting point,
not a constraint. Add a distinct portable per-node optimizer lock; only that lock
prevents `Find fastest` from comparing and replacing an otherwise eligible
implementation. Applying a winning preference never creates a lock implicitly.

**Presentation:** each processing-node header has a compact planned/used pill
such as `CPU`, `GPU · CuPy`, `GPU · cuCIM`, or amber `CPU fallback`; host-only
infrastructure is labelled `Host` or left unbadged. Planned and actual states
are visually distinct and become stale on relevant edits. Run details explain
the selected runtime/library/device, benchmark source, fallback, and repair
action. Diagnostics provide safe copyable install commands, never silently run
pip, and show system RAM plus dedicated VRAM on discrete devices or one shared
memory budget on unified-memory devices.

**Tests/documentation:** scientific result identity is based on the actual
implementation/version and scientific semantics, not global mode, fallback
policy, device index, or raw benchmark timing. Benchmark-profile identity is
separate. Test Auto ↔ Custom reuse of identical actual CPU results, stale-run
rejection after preference changes; CPU, `best_gpu`, library, exact-pin, and
independently re-resolved fallback cache admissibility; node deletion/
duplication; one-step optimizer
and node-benchmark undo, workflow v4 round trips/migration, canonical workflow
hash changes when authored compute intent changes, preservation through every
existing reader/writer while external execution remains CPU-only, narrow-toolbar
behavior, selected/pinned/preview host materialization,
visible unavailable/OOM fallback, strict preflight, install guidance, CPU-only
UI, and honest cancellation.

**Migration:** bump workflow to v4 here. Missing/v3 execution intent becomes CPU
to preserve historical behavior; newly authored workflows/sessions default to
Auto. Store only global intent, fallback, and accepted stable node preferences;
local benchmark evidence and resolved hardware remain local. Existing headless
callers retain CPU defaults and host caches without reconstructable records are
invalidated once. Land the v4 parser, serializer, canonical hash, and round-trip
preservation in all existing consumers atomically. Until their later integration
passes, batch/generated/export consumers preserve the block but explicitly force
CPU and do not reinterpret or discard the authored intent.

**Acceptance/rollback:** the user can see what was planned, what actually ran,
why, what was benchmarked, whether evidence is stale, and how to repair a missing
dependency; accepted choices survive save/reopen while raw evidence does not
enter the workflow. No device value reaches napari. Rollback hides controls and
forces CPU through the same execution service, but retains the v4 parser,
canonical hash, and cross-consumer round-trip so already-saved workflows remain
readable. **Still disabled:** batch GPU, generated/CLI compute overrides, export
sidecars, and unimplemented node families.

**Parallelism:** UI tests can begin against fake reports while provenance is
built. Only one agent edits `_widget.py`; only one edits `_graph.py`.

### Pass 5 — Batch execution

**Status (2026-08-04): implemented.** Batch config/manifest schema 2, full
saved/effective compute requests, shared CPU/GPU execution, exact per-item/node
provenance, structured OOM records, nested progress, cooperative cancellation,
cancelled status, item cleanup, and fail-closed publication are integrated.
Schema-1 configs migrate to CPU. The saved runner supports one-run CLI
overrides, both progress levels, SIGINT cleanup, and exit code 130.

**Superseded schema note (2026-08-06):** release 0.13.0a1 advances batch configs
and manifests to schema 3 for guarded per-source axis declarations and raw versus
effective axis provenance. Schema-1 and schema-2 configs remain readable.

**Depends on:** Pass 4.
**Owns:** `core/batch.py`, `core/batch_setup.py`, batch manifest/config versions,
`ui/batch.py`, `ui/batch_controller.py`, `ui/batch_navigator.py`, and focused
changes in `_tests/test_batch.py`, `_tests/test_batch_setup.py`,
`_tests/test_batch_controller.py`, and `_tests/test_batch_navigator.py`, plus
batch docs. It calls `core/execution.py` but does not redesign it.

**Public contracts:** effective global mode, fallback policy, and authored
per-node preferences in `BatchConfig`; run/item/node actual compute provenance
in `BatchManifest`; cancel token; nested progress; runtime batch scope and item
cleanup.

**Tests/documentation:** multi-item residency/pool reuse without cross-item
arrays; item cleanup on success/error/cancel/OOM; one-time visible CPU retry;
manifest fallback records; source identity re-verification immediately before
atomic publication; attached batch-config/hash replay; partial-output and
continue-on-error behavior; and CPU-only replay. A fully skipped item must not
discover, import, initialize, or probe an accelerator. Representative benchmark
choices are labelled estimates because per-item shape/dtype may differ; every
item still preflights its actual workload.

**Migration:** bump batch config/manifest schema deliberately. Old configs map
to CPU because that preserves their only former behavior; old manifests remain
read-only artifacts. Authored compute intent participates in the attached
configuration/history hash, but local timings and resolved hardware do not. Do
not change workflow schema.

**Acceptance/rollback:** every item has zero live device values after cleanup;
all v0.12 staged-write, source-identity, skip, attached-config/hash, and atomic
promotion guarantees still pass; and the runner replays its recorded request.
Rollback forces CPU and continues producing the new manifest fields. Generated
Python/CLI/export integration is now supplied by completed Pass 8.

**Parallelism:** batch UI can be developed against fake core records with
separate owners, but `core/batch.py` has one owner. Can overlap Pass 6's CuPy RL
algorithm tests after Pass 4 if no shared files are edited.

### Pass 6 — Ordinary Richardson–Lucy

**Status (2026-07-29):** the headless implementation, exact node/optimizer
benchmark substrate, and exact-region public Auto/Custom visibility are
implemented. Durable batch/generated/CLI/export exposure was completed on
2026-08-04; broader real-data evidence and cross-platform qualification remain
gated.

**Depends on:** Passes 1 and 4; durable exposure also depends on Pass 5's batch
cleanup contract.
**Owns:** new `core/gpu/cupy_rl.py`; RL declaration/policy blocks; RL-specific
memory model; new `_tests/test_gpu_rl.py`; focused additions to
`_tests/test_operations.py` and `_tests/test_execution.py`, plus
`_tests/test_batch.py` only after Pass 5; benchmark artifacts. CPU algorithm edits are prohibited
unless a separate reviewed contract test exposes an existing inconsistency.

**Implementation contracts:** operation `richardson_lucy_deconvolution`,
implementation `rl-cupy-f32-v1` version 1, callable
`napari_vipp.core.gpu.cupy_rl:richardson_lucy_deconvolution`, block/iteration
checkpoint protocol, conservative fixed-output RL memory estimate, and
`rl-float32-tolerance-v1` parity policy.

**Tests/documentation:** focused fake and real-CUDA coverage now exercises current
parameters, PSF/grid checks, 2D/3D and leading blocks, negative/non-finite
handling, scale preservation, typed output planning, per-iteration progress/
cancel/sync, exact two-input evidence, cleanup, and optimizer reuse. The
implemented exact parity gate is NRMSE `<= 2e-6` plus
`max_abs <= 1e-6 + 5e-6 * reference_peak`, with maximum ULP distance retained
as a diagnostic. Iteration-500 stress outside the initial admitted region,
wider calibrated real-data performance, RL-specific real-OOM evidence, and
cross-platform evidence remain promotion work beyond the common durable
substrate tests.

**Migration:** none; declarations only widen runtime capability.

**Acceptance/rollback:** the exact region passes its focused scientific,
memory-model, cleanup, exact-benchmark, and iteration-progress gates and is
visible as a public candidate. Auto selection still requires the section 5.4
end-to-end benefit rule using the production adapter; unvalidated environments
and parameter regions remain on CPU. No parameter is hard-coded from the spike.
Rollback removes the RL declaration. **Still disabled:** any new RL
initialization/boundary/default outside the accepted CPU contract.

**Parallelism:** algorithm/parity work can overlap Pass 5 batch work after Pass
4 with disjoint files; final batch tests wait for Pass 5. One owner controls
shared RL provider primitives.

### Pass 7 — RL-TV

**Status (2026-07-29):** the headless implementation, exact lambda-zero and
positive-TV regions, ordered-input benchmarking, iteration
progress/cancellation, conservative memory model, fixed/holdout evidence, and
exact-region public Auto/Custom visibility are implemented. Durable
batch/generated/CLI/export exposure was completed on 2026-08-04; calibrated
biological data and cross-platform qualification remain gated.

**Depends on:** Pass 6 and the existing RL-TV validation baseline.
**Owns:** new `core/gpu/cupy_rl_tv.py` (or an RL-TV-only extension of a shared
RL provider module owned by this agent); RL-TV declarations/policies/memory
model; new `_tests/test_gpu_rl_tv.py`; focused additions to
`_tests/test_rl_tv_validation.py`, `scripts/validate_rl_tv_phantoms.py`, its
results, and documentation.

**Public contracts:** `vipp.cupy.richardson_lucy_tv` version 1,
`rl-tv-cupy-f32-v1`, floor-activation diagnostics, and the existing iteration
checkpoint protocol.

**Tests/documentation:** lambda-zero equivalence, current TV sign/stencil/
epsilon/floor semantics, phantom feature retention, PSF/grid behavior,
2D/3D/iteration extremes, real calibrated datasets, progress/cancel, memory,
OOM, and durable batch cleanup.

**Migration:** none. Do not change shipped examples, defaults, formula,
initialization, padding, PSF preparation, or TV spacing.

**Acceptance/rollback:** the exact public-candidate profiles pass numerical,
feature, memory, cleanup, and truthful iteration gates. Auto selection still
requires the section 5.4 end-to-end benefit rule; broader restoration,
release, and platform claims still require calibrated real-data and
cross-platform review. Rollback removes only RL-TV capability. **Still
disabled:** alternative TV stencils, observed
initialization, reflect padding, and fast precision.

**Parallelism:** dataset preparation and blinded review can begin earlier; code
integration waits for Pass 6 and has one owner for shared RL files.

### Pass 8 — Generated Python and cross-surface persistence

**Status (2026-08-04): implemented.** Generated callables and CLIs now honor
the embedded or explicit compute request, preserve manual-node/full-pipeline
semantics, report the formal execution result, serialize exact actual node
identities and structured fallback/failure records, bind provenance to each
output, and write atomic sidecars. CLI overrides are non-mutating, progress and
SIGINT cancellation are wired, cancellation exits 130, and publication fails
closed when cleanup is not proven. The generated folder loop is explicitly
labelled non-durable; production collection replay delegates to Pass 5's saved
runner.

**Depends on:** Passes 4-7 and stable batch request/provenance contracts.
**Owns:** integration-only changes around the frozen Pass 4 workflow-v4 compute
block in `core/workflow.py`; `core/export.py`; export/generated tests; focused
workflow and batch tests; effective batch-config/generated-artifact hash and
provenance integration; and examples/docs that explicitly choose intent. Do not
add unrelated graph schema changes.

**Public contracts:** `run_pipeline(..., compute_request=None)`; CLI
mode/fallback/node-preference
overrides; `PipelineResults.execution_report`; and a provenance-sidecar helper.

**Tests/documentation:** reuse Pass 4's v4 round-trip/migration goldens;
unavailable implementation IDs remain import-safe; Auto remains portable on
CPU-only hosts; strict versus visible fallback is identical across surfaces;
session/CLI overrides do not mutate embedded JSON; Pass 4's canonical workflow
hash remains authoritative while batch-config/generated-artifact hashes respond
deliberately to effective overrides; generated version lock;
CPU-only generated import; and cache/provenance parity between UI, headless,
batch, and generated runs.

**Migration:** no new workflow-schema bump. Pass 4 already migrates v3 to v4 and
stores accepted stable preferences only. Batch config records its effective
override and own config hash; generated Python embeds the same portable v4 block.

**Acceptance/rollback:** exported and batch runs reproduce authored request
semantics and record actual resolution; old migrated CPU workflows execute
unchanged; and a stale benchmark never remains labelled optimal. Rollback keeps
the Pass 4 v4 parser/UI persistence and forces CPU only in the disabled external
surfaces.

**Parallelism:** generated-Python work can begin against the frozen Pass 4 v4
fixture, but one owner coordinates integration-only `workflow.py` and hash edits.

### Pass 9 — Cross-platform packaging and provider-completeness gates

**Progress evidence:** the native-Windows cuCIM skimage sub-gate has a
repeatable pinned build procedure, selected upstream tests, primitive
benchmarks, and a checksum-aware application-environment install in
[the source evaluation](cucim-windows-source-evaluation.md). The dedicated
CUDA-13 environment passed its probes, `pip check`, and all 98 background tests,
including 45 real RTX cases (integer exact, float32 bounded v2). This is strong
Phase 1 local evidence, not public
distribution or cross-platform admission. Pass 9 remains open for
Linux/multi-device evidence, deterministic artifact production, distribution,
and feature completeness. The
separate [Windows port plan](cucim-windows-port-plan.md) owns the upstream-
tracking fork and eventual native C++/Clara work; this pass decides what VIPP
can honestly install and advertise.

**Depends on:** Pass 0 contracts and the proposed optional-extra metadata;
otherwise independent of production operation promotion.
**Owns:** GPU-extra environment markers in `pyproject.toml`, cross-platform CI
jobs, provider/package probe scripts, supported-platform documentation,
isolated cuCIM build/benchmark artifacts, and machine-readable clean-environment
resolution/probe evidence. It does not own workflow schemas or enable a
scientific operation merely because cuCIM builds.

**Public contracts:** Python 3.12/CUDA 13 as the primary native-Windows
development track; a separately validated CUDA 12 track before public release;
named native-Linux distribution/architecture coverage; WSL2 documented as a
separate Linux deployment rather than a repair path for native napari; no CUDA
dependency resolution on macOS; provider-neutral diagnostics; and explicit
promote/defer/reject results for each CuPyX/cuCIM implementation region.

**Tests/documentation:** clean base-wheel build/install/import/npe2/generated-
Python and CPU suites on native Windows, macOS, and Linux; CUDA-13 RTX 5090 and
RTX 40-series laptop probes; CUDA-12 native Windows/Linux probes; Ubuntu
22.04/24.04 x86-64 evidence before Linux advertisement; and WSL2 secondary
evidence. Each CUDA track installs exactly one compatible CuPy distribution and
runs required CuPyX modules. macOS packaging resolves no CUDA packages. A
time-boxed M1 Max Metal/MPS/MLX feasibility study reports candidate operations,
array bridges, unified-memory accounting, packaging, and parity requirements;
until a provider passes them, macOS uses the CPU path without being described as
intrinsically GPU-incapable.

Clone cuCIM at a pinned revision on every advertised target, build installable
artifacts, run relevant upstream and VIPP-adapter tests, and benchmark only
justified candidates. Soon after Phase 1, perform a named full-feature review of
`cucim.clara/libcucim` rather than treating a skimage-only build as the desired
end state. Prefer a feature-complete, maintainable cuCIM integration; keep the
Clara work outside Phase 1 while its scope and upstream path are established.

**Migration:** none; packaging metadata changes only optional dependency
resolution.

**Acceptance/rollback:** all three base OS jobs pass and every advertised CUDA
environment installs from a clean environment and passes its doctor. A CUDA
target without a wheel or reproducible supported build is removed from the
published matrix. Each cuCIM operation requires a reproducible package plus the
common scientific, memory, cancellation, maintenance, and Custom/Auto gates;
one fast primitive cannot admit the whole library. A narrower OS/provider matrix
must be explicit. Rollback removes cuCIM independently and can remove CUDA
extras/install UX while retaining the portable CPU base. **Still disabled:**
every unvalidated OS/distribution/architecture/provider, including Apple GPU
execution until its own provider gate passes.

**Parallelism:** CI and packaging files have one owner. cuCIM build/benchmark
evidence and other clean-environment probes can run in parallel with Passes 1-8
after Pass 0 when they use disjoint artifacts.

### Pass 10 — Segmentation, measurement, and broad node promotion

**Depends on:** Passes 1-4 plus operation-specific scientific evidence for
implementation work. A family may be researched and validated before Pass 9 is
complete, but public advertisement requires its relevant Pass 9 runtime/library/
platform packaging gate. Batch/workflow/generated integration waits for Passes
5 and 8 rather than blocking the core operation adapter.
**Owns:** one operation family per sub-pass/PR, with its own provider module,
declaration, memory/workload/parity policy, tests, benchmark artifacts, and docs.

**Public contracts:** no new generic contract unless an operation proves the
existing one insufficient. A small Phase 2 bridge wave covers only inexpensive
pointwise/arithmetic/mask operations needed to preserve useful residency. The
Phase 3 scientific wave has completed Otsu, Canny, Connected Components, and
the basic production-schema Measurements slice. The later broad-coverage
phase adds remaining filters, morphology,
segmentation, label cleanup, colocalization, and other reasonable nodes. For
each node, benchmark all scientifically admitted CuPyX, cuCIM, and future
runtime/library implementations rather than assuming one library is universally best.
No family is promoted by provider API similarity alone.

**Tests/documentation:** full common promotion gate plus operation-specific
scientific fixtures, mixed-graph/batch/export integration, performance,
cancellation granularity, and memory.

**Migration:** normally none. A new parameter, algorithm, or precision mode is
separate scientific/schema work.

**Acceptance/rollback:** each Custom candidate meets parity, bounded-memory,
cleanup, provenance, packaging, and CI requirements. Auto additionally requires
section 5.4 evidence for the complete segment. A GPU residency bridge may remain
a Custom candidate even when its isolated kernel is slower, because the
whole-pipeline optimizer may prove that avoiding transfers is globally faster.
Rollback deletes only that declaration/policy. **Still disabled:** all
unpromoted dtype/parameter regions and operations.

**Parallelism:** multiple operation-family agents may work concurrently only
after declaration/policy files are split into family-owned modules or a single
registry integrator serializes their small shared-map edits.

### 13.1 Critical path and safe parallel work

```text
Pass 0 contracts
  -> Pass 1 headless execution/benchmark/dev substrate
      -> Pass 2 Background/Subtract Background
          -> Pass 3 Median/Gaussian/headless optimizer
              -> Pass 4 toolbar + Custom node/pipeline UX
                  -> Pass 6 RL [implemented] -> Pass 7 RL-TV [implemented]
                  -> Pass 10 Otsu/Canny/labels/measurements and bridges
                  -> Pass 5 batch
                      -> Pass 8 generated Python/cross-surface persistence
                          -> Pass 10 remaining reasonable-node families

Pass 9 packaging/provider validation starts after Pass 0 and runs alongside the
feature passes. Its CUDA matrix must pass before public CUDA extras ship. Its
cuCIM completeness and Apple-provider investigations independently promote,
defer, or reject those surfaces without blocking the portable CPU application.
```

Safe parallelism is evidence/UI-fixture work with disjoint files. Shared core
registries, `pipeline.py`, `execution.py`, `_widget.py`, `batch.py`, and
`workflow.py` each have one active owner. A registry integrator should merge
family declarations after parallel agents finish rather than allowing several
agents to edit the same tuple/map.

## 14. Implementation execution rules

The numbered passes above are the authoritative work orders. Generate a fresh,
pass-specific task from those contracts when implementation starts; do not reuse
the pre-2026-07-27 global-only prompts because they predate Custom mode,
per-node choices, and the runtime/library split.

For every implementation pass:

1. Inspect the branch status, current diffs, architecture tests, and directly
   affected CPU functions before editing. Preserve unrelated user work.
2. Assign one owner to each shared registry and each large composition file.
   Parallel agents may own disjoint implementation modules, fixtures, benchmarks, and
   platform evidence; the designated integrator serializes shared-map edits.
3. Preserve the CPU function's parameters, defaults, axes, boundaries, dtype,
   scaling, metadata, progress, and errors. A matching optional-library primitive
   is evidence only until the complete VIPP adapter passes parity.
4. Keep optional accelerator discovery lazy. CPU mode and fully skipped batch
   work must not import or initialize an accelerator.
5. Admit a candidate to Custom only after scientific, memory, cleanup, and
   cancellation gates. Make it a reviewed Auto default only after the separate
   packaged end-to-end policy gate. Raw node or optimizer evidence never admits
   it automatically. With accelerated-only exact compatible history, the next
   global Auto run measures CPU once on the same execution surface; later
   matching runs apply the 1.20x/20-ms gate to the completed pair. Never mix
   interactive, batch, or registry-lifecycle timing surfaces.
6. Use deterministic fake runtimes/clocks for required tests and named real GPU
   hosts for promotion evidence. Report unsupported dtype/parameter regions
   explicitly.
7. Benchmark transactionally: no writer, publication, history, live cache, or
   preference mutation before user acceptance. Store stable choices in authored
   configuration; keep raw timings and hardware fingerprints local.
8. Run focused tests, CPU-only import/architecture checks, Ruff, and the feasible
   full suite before integration. Run real-device parity/benchmark jobs whenever
   the pass advertises hardware support.
9. Commit each coherent reviewed feature on
   `codex/gpu-cross-platform-support` with a narrow message, then push the branch
   after checks pass. The designated integrator alone commits/pushes shared
   multi-agent work; never force-push or mix unrelated changes.

### Phase 1 integration sequence

- **Commit A — phase-era contracts:** CPU/Auto/Custom, before Prefer GPU; node
  preferences, fallback,
  runtime/library/implementation, dtype/value, decision, and benchmark records.
- **Commit B — CPU execution seam:** unified Qt-free prepared-node/execution
  service with byte/state-for-state CPU compatibility.
- **Commit C — fake runtime/planner:** opaque device values, segmentation,
  liveness, memory, cancellation, OOM/fallback, the generic transactional node-
  benchmark and graph-optimizer services, and deterministic fake tests.
- **Commit D — CUDA development runtime:** lazy CuPy runtime, diagnostics/doctor,
  private-pool/out-of-pool accounting, and reproducible CUDA 13/12 setup.
- **Commit E — background:** production-faithful Rolling-Ball and Subtract
  Background adapter with experimental cuCIM admission.
- **Commit F — median:** CuPyX median declarations, exact parity matrix, memory,
  and real-device evidence.
- **Commit G — Gaussian:** CuPyX 2D/3D Gaussian declarations, tolerance matrix,
  memory, and real-device evidence.
- **Commit H — operation-candidate benchmarking:** wire the production adapters
  into the already fake-tested parity-before-timing service, then collect local
  fingerprints/staleness and real-candidate evidence.
- **Commit I — pipeline optimizer/integration:** graph-global assignment,
  one-transfer Background → Gaussian → Median acceptance, and complete Phase 1
  headless provenance/cache-admissibility tests.

Stop after each commit if CPU behavior changes, a device value escapes, parity
is unexplained, cleanup is incomplete, or the environment cannot be reproduced.

## 15. Recorded product decisions and remaining gates

The 2026-07-27 product direction resolves the interaction model. The remaining
items below are empirical release gates or deliberately deferred product choices,
not reasons to redesign Phase 1.

| ID | Decision | Recorded direction | Remaining gate |
| --- | --- | --- | --- |
| D1 | Compute intent | Global modes are CPU, Auto, Prefer GPU, and Custom; Auto is the new-session default. Prefer GPU selects an admitted reviewed GPU regardless of CPU speed and keeps per-node preferences dormant. Custom provides per-node `Auto for this node`/CPU choices, one choice per GPU library, and Best GPU only when multiple libraries compete; exact pins are advanced-only. `Find fastest` is Custom-only and treats these as the current assignment unless a distinct optimizer lock is set. | Keep lock state separate from execution intent and scientific identity. |
| D2 | Fallback | Auto or Prefer GPU choosing CPU normally is not fallback. Prefer GPU requires visible fallback; Custom uses visible CPU fallback by default and may opt into strict fail-closed behavior. | Retain typed retryability tests across all execution surfaces. |
| D3 | OOM | Auto or Prefer GPU may clean and retry one affected transactional segment once on CPU and must report it. Custom follows visible/strict policy. | Validate no partial commit, leak, or duplicate side effect. |
| D4 | Persistence | Workflow v4 stores global intent, fallback, and authored node preferences; v3 migrates to CPU. Separate validated VIPP UI metadata stores optimizer-locked node IDs without changing the scientific workflow hash. Pass 4 atomically updates the canonical workflow hash for compute intent. Batch config v2 stores its full effective request and v1 migrates to CPU; generated/batch/export surfaces execute that intent and record configured/effective hashes plus actual provenance. Resolved hardware and timings remain outside portable workflow JSON. | Implemented across interactive, batch, generated Python/CLI, and export; cross-platform replay evidence remains a release gate. |
| D5 | Installation UX | Start with provider-aware diagnostics and a safe copyable command. A later in-app installer requires explicit consent, an isolated supported environment, progress, verification, and restart; never mutate an arbitrary napari environment silently. | Validate CUDA-13 and CUDA-12 packages before publishing extras or commands. |
| D6 | Result caching | Key results by actual implementation/version and scientific semantics. Identical actual CPU execution may be reused across Auto/Prefer-GPU/Custom; different implementations remain separate unless a reviewed bitwise-equivalence group exists. | Prove stale-run and exact-pin behavior in Pass 4. |
| D7 | Initial hardware | CUDA acceleration targets validated native Windows and Linux first, including RTX 5090 and RTX 40-series laptops. WSL2 is a separate Linux deployment. macOS uses CPU initially while M1 Max Metal/MPS/MLX support is investigated. | Name public OS/Python/CUDA/device tiers only after clean-host evidence. |
| D8 | Performance | Custom admission is scientific/operational. A reviewed Auto default requires a lower-confidence 1.20x end-to-end prediction and 20-ms saving. With no compatible history, Auto uses that default; accelerated-only history schedules one same-surface CPU measurement; a later matching run applies the same 1.20x/20-ms gate to the pair. Incompatible interactive, batch, and registry-lifecycle surfaces are never mixed. Raw Custom benchmark winners follow their separate review threshold and never teach Auto. | Recalibrate only from reviewed multi-device production-adapter evidence. |
| D9 | CI | Every PR keeps CPU/package checks on Windows, macOS, and Linux. Scheduled real-GPU jobs cover Windows CUDA 13 and native Linux CUDA 12/13; releases expand the matrix. | Secure stable GPU hosts and define maintenance ownership before public promotion. |
| D10 | Device identity | Exact device/driver/runtime belongs in local benchmark identity and run provenance, not scientific result identity unless it changes semantics. Portable artifacts use descriptive tiers and privacy-preserving identifiers. | Review the provenance schema in Pass 4. |
| D11 | Memory | Discrete CUDA starts with an 80% cap and `max(512 MiB, 10%)` reserve, separate from host RAM. Unified-memory providers use one shared budget and never add RAM plus nominal VRAM. | Tune on the 5090, laptop GPUs, and M1 Max before broad defaults ship. |
| D12 | Sidecars | Atomic `.vipp-provenance.json` is the default beside standalone generated exports; batch keeps equivalent per-item execution documents and per-output digest links in its manifest/checkpoints. | Implemented; retain normalized-path, failure-sidecar, and atomic-write regression tests. |
| D13 | Precision | Ship strict scientific-default behavior only; no global fast/mixed-precision control. | Add any relaxed precision only as operation-specific, versioned evidence-backed work. |
| D14 | Cross-platform meaning | VIPP remains supported on Windows, macOS, and Linux. CUDA is only one provider; lack of CUDA on macOS does not preclude a later Apple GPU provider. | Apple feasibility and packaging have their own gate; never imply NVIDIA code runs on macOS. |
| D15 | cuCIM/Clara | Use cuCIM operation-by-operation and keep CuPy independent. Clara is outside Phase 1 but must receive a named near-term feature-completeness/upstream review; the desired end state is not a permanently hobbled skimage-only fork. | Reproducible target packages, VIPP-adapter parity, Linux evidence, and a maintainable Clara decision. |
| D16 | Benchmark persistence | Benchmarking proposes choices transactionally. Workflows store only user-accepted stable preferences; raw timings/hardware remain local and visibly become stale. | Finalize invalidation fingerprints and local record migration in Passes 0-4. |
| D17 | Phase 1 scope | Headless contracts/substrate/setup plus Background, Subtract Background, median, and 2D/3D Gaussian; no production toolbar, batch, workflow, or managed installer yet. | Implemented on the GPU development branch; Phase 2 owns interactive exposure and persistence. |

### Principal risks and mitigations

- **Execution refactor changes CPU behavior.** Make CPU parity the Pass 1 gate,
  retain a CPU-only rollback, and extract call preparation/finalization before
  provider logic.
- **Asynchronous work appears complete too early.** Synchronize at timing,
  iteration, transfer, commit, cancellation, and cleanup boundaries.
- **Memory estimates undercount hidden workspace.** Use provider-specific
  measured upper bounds with margin, a private hard-capped pool, and actual-OOM
  recovery.
- **Auto decisions vary across machines.** Ship versioned validated records,
  use conservative out-of-domain CPU behavior, and record every descriptor and
  policy decision.
- **An isolated-node winner makes the pipeline slower.** Optimize the whole
  graph with transfer, residency, branch, host-materialization, and memory costs;
  never greedily concatenate node winners.
- **Benchmarking perturbs results or measures thermal noise.** Run candidates in
  side-effect-free transactions, synchronize timed boundaries, separate cold and
  warm samples, randomize order, expose variance, and keep the current choice
  inside the versioned noise floor.
- **Accepted evidence becomes stale.** Retain the authored preference but stop
  claiming it is optimal after graph, parameter, input, device, driver, runtime,
  or implementation changes; offer a clear rebenchmark action.
- **Fallback masks defects.** Retry only classified unavailable/support/OOM
  cases; unclassified provider or scientific errors fail.
- **Device arrays leak through branching or stale workers.** Keep them in a
  transactional private store, host-materialize public outputs, and assert zero
  device values at every segment/item/run exit.
- **Tolerance drift hides scientific changes.** Version parity policies, require
  local and aggregate gates, preserve real-data fixtures, and review any relaxed
  threshold as a scientific change.
- **A fast library primitive is not the VIPP node.** Benchmark the complete
  adapter, including dtype restoration, non-finite handling, blocks/channels,
  metadata, smoothing, clipping, and transfers.
- **Installation guidance damages an existing environment.** Detect the current
  platform/runtime first, prefer a dedicated environment, never install two CuPy
  CUDA-major distributions together, and reserve future managed installs for
  explicit consent plus verification/restart.
- **Concurrent agents collide in large files.** Assign one owner to shared core
  registries and each of `pipeline.py`, `_widget.py`, `batch.py`, and
  `workflow.py`; split operation-family declarations before parallel promotion.

## Ordered next suggested steps (maintained 2026-08-04)

This is the implementation queue after the basic Measurements vertical slice
and the completed durable batch/generated-Python/CLI/export integration.
Update this section when a wave lands so the branch and handoff report retain
one explicit order. The completed public Canny/Otsu contracts and their separate evidence
protocol are recorded in
the [Phase 3A report](gpu-phase3-canny-otsu-implementation-report.md); Sigma's
contract, external Fiji evidence, fused public CuPy provider, and canonical
timing crossovers are recorded in the
[Phase 4 report](gpu-phase4-sigma-filter-implementation-report.md). Connected
Components' exact IDs, resident CuPyX path, lifecycle/memory limits, and
machine-local screening are recorded in the
[Phase 5 report](gpu-phase5-connected-components-implementation-report.md).
Measurements' typed host boundary, exact basic schemas, public regions, and
workload-dependent timings are recorded in the
[Phase 6 report](gpu-phase6-measurements-implementation-report.md).

1. **Native platform evidence:** validate supported native Linux targets and the
   available Windows RTX 40-series laptops, including clean setup, real kernels,
   parity, memory, cancellation, cleanup, and end-to-end selection. WSL2 is
   secondary evidence, not a substitute for native Windows/Linux claims.
2. **CPU-only and optional-GPU packaging evidence:** keep base wheel import,
   workflow, batch, generated-Python, CLI, and export checks green on native
   Windows/macOS/Linux without accelerator packages. Produce reproducible,
   mutually exclusive CUDA 12/13 installation and clean-environment evidence
   before broad distribution claims.
3. **cuCIM/Clara feature-complete investigation:** perform the named near-term,
   time-boxed packaging/upstream review. Prefer a maintainable feature-complete
   cuCIM route; do not normalize a permanently hobbled skimage-only fork.
4. **Apple M1 Max study:** evaluate Metal/MPS/MLX through the provider contracts
   with unified-memory accounting. Keep the CPU path as the honest fallback
   unless an operation family passes scientific and performance gates.
5. **Convert Dtype, inexpensive residency bridges, and broader reasonable-node
   coverage:** support only explicit authored conversions and scientifically
   faithful operations that can keep useful segments resident. Never insert a
   cast or bridge merely to improve a benchmark. Promotion still requires the
   same parity, memory, progress, cancellation, cleanup, and durable-surface
   tests.

The UI immutable-snapshot/lifecycle hardening listed above remains a prerequisite
for broad optimizer claims and should be completed alongside the early operation
waves; it is not marked complete by Phase 2B. Lazy resource-backed axis/channel
selection is also now tracked before sustained real-acquisition benchmarking:
Extract Channel and Select Axis Slice must slice first and materialize only the
selected workload rather than eagerly decoding a complete ND2 acquisition.

## Handoff summary

1. **Product model:** CPU, Auto, Prefer GPU, and Custom are distinct global
   modes. Auto is the default. Prefer GPU uses every eligible reviewed GPU
   region without the CPU-speed gate, requires visible fallback, and leaves
   saved per-node choices dormant. Custom adds per-node preferences, node
   benchmarking, and a graph-global `Find fastest pipeline…` action with
   distinct per-node locks.
2. **Architecture:** one Qt-free execution service plans immutable decisions,
   executes transactional host/device segments through lazy runtimes and
   implementation libraries, keeps public caches host-only, and returns actual
   implementation provenance.
3. **Phase 1:** implemented on the GPU development branch: contracts ->
   fake/lazy-CUDA substrate and reproducible dev setup -> production-faithful
   Background/Subtract Background -> CuPyX median and Gaussian -> headless node
   benchmark, scientific cache identity, and whole-pipeline optimizer.
4. **Next wave:** the initial toolbar/inspector/badge slice, workflow-v4 compute
   intent, setup/memory diagnostics, selected-node benchmark review, the
   conservative Custom whole-pipeline optimizer, and ordinary GPU RL are
   implemented. RL-TV, exact-mask Canny/Otsu, and the Sigma Filter vertical
   slice are now implemented too. Connected Components is also complete with
   exact native-`int32` IDs, resident CuPyX execution, block-boundary lifecycle,
   and packaged policy artifact v5. Basic morphology and intensity Measurements
   are complete with a cuCIM resident provider, mandatory typed host-table
   finalizer, visible exact-region fallback, and RTX workload evidence.
   Durable batch/generated-Python/CLI/export execution is now complete for
   supported node regions, including exact provenance, structured fallback,
   nested progress, cancellation, and cleanup-gated publication. Continue in
   the maintained order above: native/packaging evidence, provider-completeness
   review, Apple feasibility, then explicit bridges and broader node coverage.
5. **Admission rule:** scientific validity, memory, progress, cancellation,
   cleanup, and runtime evidence make a region a normal public Custom and
   Prefer-GPU candidate; incomplete/unvalidated work alone remains
   `developer_hidden`. A reviewed Auto default additionally needs conservative
   packaged end-to-end performance evidence. Auto uses that default without
   compatible history, performs one same-surface CPU measurement after an
   accelerated-only observation, then applies the 1.20x/20-ms gate once the
   pair exists. It never silently benchmarks multiple implementations or mixes
   incompatible execution surfaces. Prefer GPU bypasses only that
   performance requirement;
   unsupported subregions visibly use CPU, and primitive benchmarks alone admit
   nothing.
6. **Platform direction:** portable CPU support remains Windows/macOS/Linux;
   CUDA targets validated Windows/Linux first. M1 Max Metal/MPS/MLX feasibility
   is a named near-term investigation with unified-memory semantics. cuCIM's
   skimage work continues, while Clara/full feature completeness is reviewed
   soon after Phase 1.
7. **Delivery discipline:** implement reviewable coherent commits on
   `codex/gpu-cross-platform-support`, validate before each push, and keep the
   remote development branch current without mixing unrelated work.
