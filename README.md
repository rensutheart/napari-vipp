<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/assets/branding/vipp-logo-dark.svg">
    <img src="docs/assets/branding/vipp-logo.svg" alt="VIPP" width="420">
  </picture>
</p>

# VIPP — Visual Image Processing Platform

**Visual workflows for reproducible bioimage analysis.**

[![CI](https://github.com/rensutheart/napari-vipp/actions/workflows/ci.yml/badge.svg)](https://github.com/rensutheart/napari-vipp/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/napari-vipp.svg)](https://pypi.org/project/napari-vipp/)
[![Python](https://img.shields.io/pypi/pyversions/napari-vipp.svg)](https://pypi.org/project/napari-vipp/)
[![License](https://img.shields.io/pypi/l/napari-vipp.svg)](LICENSE)

`napari-vipp` is the napari-native implementation of **VIPP, the Visual Image
Processing Platform**. Build typed node graphs, inspect intermediate images and
tables, tune parameters, save workflows, and repeat the same operations without
hiding axis or physical-scale metadata.

> **Alpha software:** expect breaking workflow and parameter changes. Validate
> outputs on representative data before scientific interpretation or
> publication.

VIPP's implemented safeguards include stable source revisions, physical-grid
checks, exact unsampled diagnostics, detached viewer layers, atomic artifacts,
and batch publication only after source reverification. See the
[scientific integrity boundaries](docs/architecture.md#scientific-integrity-boundaries)
and the contributor [scientific behavior requirements](CONTRIBUTING.md#scientific-behavior-requirements).

## Install And Open

VIPP 0.13.0a1 supports CPython 3.12 and 3.13. If napari is not already
installed, install it with a Qt backend at the same time:

```bash
python -m pip install "napari[pyqt6]"
python -m pip install --pre napari-vipp
vipp
```

The `--pre` flag is required while VIPP is published as an alpha release. It is
kept on the VIPP command so napari itself can continue to resolve to a stable
release.

The optional CUDA providers are currently qualified on CPython 3.12 only. On a
native Windows machine with a compatible NVIDIA driver, the self-contained
CUDA 13 route is:

```powershell
py -3.12 -m venv ".venv-vipp-gpu-cu13"
& ".\.venv-vipp-gpu-cu13\Scripts\python.exe" -m pip install --upgrade pip
& ".\.venv-vipp-gpu-cu13\Scripts\python.exe" -m pip install --pre "napari[pyqt6]>=0.6" "napari-vipp[gpu-cuda13]==0.13.0a1"
& ".\.venv-vipp-gpu-cu13\Scripts\vipp-compute-doctor.exe" --track cuda13
& ".\.venv-vipp-gpu-cu13\Scripts\vipp.exe"
```

This installs the exact NumPy, SciPy, scikit-image, CuPy, and CUDA package
versions used by the public admission policy. Public GPU admission in this
alpha is intentionally narrower than package installation: the reviewed
native-Windows evidence is for the recorded CUDA 13 / RTX 5090 environment.
Other hardware, Linux, unsupported dtypes or parameters, and missing optional
providers remain on the scientifically authoritative CPU path with a visible
reason. macOS is CPU-only in this alpha. See
[GPU scope and setup](#gpu-execution-and-development-environment) before using
accelerated results.

In napari, open:

```text
Plugins > VIPP Workflow (napari-vipp)
```

Use `Open example...` for a runnable workflow with synthetic data. A good first
choice is `Red-Channel Label Cleanup`; select nodes from left to right to review
their parameters, thumbnails, metadata, and outputs. To explore collection
processing, open `Deterministic Batch & Provenance`; VIPP prepares a small
self-contained working copy and opens it already configured and previewed.

![VIPP example workflow chooser](docs/assets/user-guide/vipp-example-chooser.png)

## What It Supports

| Area | Current alpha capabilities |
| --- | --- |
| Graph authoring | Searchable node palette, typed ports, dynamic outputs, cycle prevention, undo/redo, graph notes, draggable named tunnels, insert-on-wire, live source subtitles, auto-layout, and saved positions. |
| Images and metadata | Semantic T/C/Z/Y/X axes, scale/units/origin, channel and acquisition metadata, source identity, and operation history. |
| Image processing | Intensity transforms, filters, background correction, thresholding, watershed, binary/label morphology, channels, axes, masks, and composites. |
| Measurements | Object and intensity tables, calibrated morphology, 3D mesh morphology, skeleton/network analysis, colocalization, object association, and table composition. |
| Restoration | Born-Wolf PSF generation, measured-PSF preparation, and manual/cached 2D or 3D Richardson-Lucy and RL-TV deconvolution. |
| Reuse and automation | Independent workflow tabs, workflow JSON, generated headless Python, explicit batch outputs, background collection runs, reviewed plans, representative navigation, retained batch results, and workflow/config/manifest artifacts. |
| I/O | OME-TIFF, ImageJ TIFF, TIFF, local OME-Zarr 0.4/0.5, NPY/NPZ, common 2D raster formats, and optional microscope readers. |

Most graph operations are still eager. Large z-stacks and OME-Zarr datasets
therefore need deliberate cache, preview, and output choices; see the
[cache and memory guide](docs/cache-and-memory.md).

## Optional Microscope Readers

Install only the reader family you need, then restart napari:

| Format family | Install command |
| --- | --- |
| Nikon ND2 | `python -m pip install --pre "napari-vipp[nd2]"` |
| Zeiss CZI | `python -m pip install --pre "napari-vipp[czi]"` |
| Mixed microscope formats | `python -m pip install --pre "napari-vipp[microscope]"` |
| BioIO/Bio-Formats fallback | `python -m pip install --pre "napari-vipp[bioformats]"` |

These routes are an experimental foundation: axes and common metadata are
normalized where the source reader exposes them, but format-specific coverage
still needs validation against a broader corpus of real acquisition files.

## Workflow Basics

1. Add or select an `Image Source` for a napari layer, file, or bundled sample.
2. Add nodes from the palette and connect compatible output and input ports.
3. Select a node to tune parameters and inspect its output metadata.
4. Click `Calculate` for manual/cached nodes such as measurements and
   deconvolution.
5. Pin important image outputs into napari for full-resolution comparison.
6. Save the graph with `Save workflow...`.
7. Add `Batch Output` nodes before `Batch workspace...` when exact saved outputs
   matter.
8. Optionally click `Preview batch` to inspect the complete plan and use the
   representative slider or a preview-table row without running or saving the
   full batch. Preview is not required: `Run batch` performs its own preflight.
9. Run the collection from the retained workspace with one click, where
   overall-item and current-operation progress, cancellation, final statuses,
   validation, and the
   `vipp_batch_manifest.json` path remain available for inspection.
10. To validate the complete batch path without your own files, choose
   `Open example...` -> `Deterministic Batch & Provenance` -> `Open batch
   demo...`. Choose where to save its small working copy, review the populated
   graph, move through all three paired fields with the representative slider,
   review the three-item/nine-output batch preview, then click `Run demo batch`. VIPP
   checks the finished outputs and provenance against exact ground truth
   automatically.

Workflow JSON stores the graph and optional VIPP UI state, not cached pixels or
tables. When Batch workspace is active, Save workflow can optionally attach its
versioned config so the same workspace reopens from that one JSON file; local
paths are included, but source pixels are not. Workflow schema 4 also stores
portable compute intent under `execution.compute`: mode, fallback policy,
per-node preferences, precision policy, and workload policy. Machine-local
runtime/device choices, memory limits, experimental admission, and benchmark
evidence are deliberately excluded. Schema-3 workflows load with an explicit
CPU policy. Collection batch config version 2 and generated Python now execute
that intent through the same CPU/GPU service as interactive VIPP, support
explicit run overrides, and record the exact implementation actually used.
Version-1 batch configs migrate to CPU. See the
[durable GPU execution guide](docs/durable-gpu-execution.md) for request
precedence, provenance, OOM fallback, progress, cancellation, CLI commands, and
current limitations.

## Documentation

- [Published VIPP documentation](https://rensutheart.github.io/vipp-mkdocs/)
- [Categorized 0.13.0a1 release notes](CHANGELOG.md#0130a1---2026-08-04)
- [Documentation index](docs/README.md)
- [User guide](docs/user-guide.md)
- [Image import and export](docs/io-user-guide.md)
- [Example workflow index](examples/README.md)
- [Measurement workflows](docs/measurement-workflows.md)
- [Operator tips](docs/operator-tips.md)
- [Developer notes](docs/developer-notes.md)
- [Current planning and roadmap](docs/planning.md)

## Development

Create a local environment and install the development dependencies:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install -e ".[dev]"
```

### GPU Execution And Development Environment

VIPP 0.13.0a1 is the first alpha to package evidence-gated GPU execution. GPU
coverage is deliberately incomplete and every accelerated region retains a
visible, scientifically authoritative CPU path. Phase 1 provides
Rolling-Ball/Subtract Background, median, and
2D/3D Gaussian. Phase 2B adds ordinary CuPy/CuPyX
Richardson-Lucy, Phase 2C adds Richardson-Lucy TV for 2D/3D spatial data and
leading blocks while preserving the existing CPU formula and defaults, and
Phase 3A adds exact-mask CuPy/CuPyX Canny and CuPy Otsu providers. Phase 4 adds
the public CPU Sigma Filter node and a clean-room CuPy RawKernel provider.
Phase 5 adds exact CuPyX Connected Components for boolean 2D/3D masks, including
SciPy-identical `int32` label IDs and independent leading-block resets. Their
validated regions are normal public GPU candidates on this alpha; unsupported
regions visibly use CPU. Phase 6 adds cuCIM candidates for
the basic schemas of **Measure Objects** and **Measure Objects + Intensity**,
with an exact typed-table finalizer after the mandatory GPU-to-host boundary.
Both deconvolution paths use exact ordered-multi-input benchmarking. The alpha
includes
CPU/Auto/Prefer-GPU/Custom execution contracts, visible or strict fallback,
transactional device execution, scientific cache identity, and per-node/
whole-pipeline benchmark services. The toolbar controls
now provide the first Phase 2 interactive slice: new sessions default to
`Auto`, the main toolbar lists `Auto`/`CPU`/`Prefer GPU`/`Custom`, and
Custom mode
shows `Auto for this node`, `CPU`, and one choice per declared GPU library
where implemented. `Best GPU` appears only when multiple libraries compete;
accepted runs add compact CPU/CuPy/cuCIM badges, and CPU fallback is shown in
amber. `Prefer GPU` considers every reviewed public GPU implementation,
including `public_custom` providers that Auto does not consider. It skips
only the CPU-versus-GPU speed gate: scientific parity, dtype, parameter,
shape, environment, dependency, and memory admission remain mandatory, and
VIPP never inserts a cast or changes an authored parameter to make GPU eligible.
When all eligible GPU choices have complete comparable timing evidence, the
fastest GPU is selected; otherwise the stable implementation ID provides a
deterministic choice without implying that it is fastest. A node with no
eligible GPU receives an explained ordinary CPU decision.

`Auto` starts from reviewed GPU defaults rather than treating missing timing as
proof that CPU is faster. Successful, fallback-free completed full-pipeline
runs—whether CPU, GPU, or mixed—add only their wall time to machine-local
history. If the exact compatible history contains an accelerated observation
but no CPU observation, the next global Auto run measures the authoritative CPU
assignment once on the same execution surface. A later matching Auto run uses
the accelerated assignment only when it beats CPU by at least 1.20x and 20 ms;
otherwise it uses CPU. Interactive, batch, and registry-lifecycle timing
surfaces are never mixed. Auto never silently benchmarks multiple
implementations. The optional CPU comparison is preflighted against host-memory
headroom. On Windows that includes both available physical RAM and remaining
system commit; if either reserve would be unsafe, Auto keeps its reviewed safe
assignment, explains that the comparison was skipped, and can collect the
missing CPU evidence on a later run.

`Prefer GPU` always uses visible fallback; a strict Prefer-GPU request is
invalid because the policy explicitly means “GPU wherever possible, CPU
everywhere else.” Saved per-node preferences remain intact but dormant outside
Custom mode. Switch back to Custom to reactivate them and to use
`Benchmark node…` or `Find fastest pipeline…`; the whole-pipeline optimizer is
intentionally Custom-only. Developer-hidden implementations remain excluded
unless experimental admission is explicitly enabled, which is not a public
support claim.

Compute intent is immutable while a calculation or benchmark is active. The
mode selector and Custom per-node controls remain disabled until the work
finishes normally. To change policy sooner, the user must explicitly choose
`Cancel calculation`/`Cancel analysis`; controls unlock only after the worker
has finished synchronizing and releasing CPU/GPU resources. Entering Custom while
idle is configuration-only: it retains the last valid output and its actual
CPU/GPU provenance. When that result does not satisfy the saved Custom choices,
VIPP marks its badges and summary as a previous result; changing a per-node
choice or calculating replaces it.

Failure handling is provenance-aware rather than all-or-nothing. A failed run
may accept a verified source boundary, and a cleanup-failed run may retain a
completed processing node only when the matching actual-implementation decision
is available for its badge and report. Uncomputed or unreported processing
values never replace an earlier valid result. Cancellation keeps the prior
coherent result. If accelerator cleanup fails during a calculation, node
benchmark, whole-pipeline analysis, or collection batch, VIPP treats the
process runtime as unsafe: all new compute and policy changes are disabled
until VIPP is restarted.

VIPP uses one message-strip component, with major and actionable paths now
severity-classified;
only actionable failures receive the full alert treatment. Workflow-v4
persistence now records portable authored compute intent, while separate
non-scientific UI metadata preserves explicit optimizer locks without changing
the scientific workflow hash. Legacy workflow-v3 files load in CPU mode and
with every node unlocked until the user explicitly opts into Auto, Prefer GPU,
or Custom.
Machine-local runtime/device selection, memory limits, provider admission,
and benchmark evidence are not copied between machines. Batch config version 2
captures the full effective run request, while generated Python embeds the
portable workflow request. Both use the shared execution service, preserve
per-node choices, report exact actual implementations, and remain import-safe
on a CPU-only installation. The saved batch runner and exported CLI add
compute/fallback/node overrides, nested or operation progress, cooperative
cancellation, exit code 130, structured OOM records, and atomic provenance.
`Settings > Compute setup and memory…` verifies optional packages and hardware
on a worker and presents system RAM plus discrete VRAM, or one shared budget on
unified-memory machines. On Windows the cache status also distinguishes
physical RAM from commit headroom because either can bound a large CPU
allocation. In Custom mode, eligible single-output nodes with
one or more ordered inputs offer `Benchmark node…`: VIPP detaches and hashes
every input, includes every transfer and input in memory accounting, compares
the exact captured workload, requires scientific parity, saves evidence
locally, previews warm timing/parity/memory results, and changes the portable
node preference only after explicit acceptance. The final UI Apply boundary
revalidates the exact input bytes and metadata, graph, compute intent, locks,
candidate assignment, and accelerator environment before making one undoable
change. Writers and multi-output nodes remain excluded.
GPU eligibility is dtype-sensitive. For example, the currently reviewed CuPyX
Gaussian implementation accepts finite `float32`; native `uint16` Gaussian is
intentionally CPU-only until its integer result semantics pass a separate
scientific admission gate. The initial ordinary GPU Richardson-Lucy region
likewise requires both the Image and PSF to be explicitly finite `float32`; its
output is shape-preserving `float32`. Its first scientifically admitted region
also requires `filter_epsilon == 1e-8`, 1 through 25 iterations, odd PSF
extents, and the default-safe normalization/clipping/scale options. The CPU
operation's existing `1e-12` default and every other epsilon are unchanged and
therefore remain on CPU: VIPP does not silently alter the threshold or shorten
an authored run to use the GPU. This is a conservative measured allowlist, not
a claim that `1e-8` is intrinsically the only valid GPU value: `1e-10` already
missed the production parity gate at 25 iterations, other tested values were
not monotonic, and `1e-8` itself had failures at 50 iterations. The exact
`1e-12` point has not yet had a complete GPU admission study. Change that
scientific parameter only when it is appropriate for the analysis, then
benchmark the exact Image/PSF workload.

GPU Richardson-Lucy TV has two separately validated profiles. With
`tv_regularization == 0`, it reduces to ordinary RL and therefore uses that
path's strict `filter_epsilon == 1e-8` policy and parity gate. Positive TV is
initially admitted only at the unchanged shipped settings:
`tv_regularization == 0.002`, `tv_epsilon == 1e-6`,
`filter_epsilon == 1e-12`, `denominator_floor == 0.05`, and exactly 10 or 25
iterations. Other positive-TV iteration counts remain on CPU until their
nonlinear trajectories are measured; lambda-zero retains ordinary RL's 1–25
range. Its nonlinear recurrence amplifies small CPU/GPU convolution and
reduction-order differences, so positive TV uses a separate, versioned 0.5%
NRMSE/peak-scaled maximum-error screen plus feature, MSE, flux, boundary, and
floor diagnostics. This is an operation-specific public candidate region backed
by fixed and holdout matrices—not permission to change an authored parameter,
and not a blanket biological-restoration or cross-platform equivalence claim.

GPU provider visibility in this alpha follows the evidence. An implementation
whose declared region has passed scientific parity and the required memory,
progress, cancellation, cleanup, and runtime checks is a normal public
`Custom` or `Prefer GPU` candidate and may participate in `Auto` where
applicable performance evidence exists. `developer_hidden` is reserved for
incomplete or unvalidated work and is excluded unless experimental admission
is explicitly enabled. Promotion is region-specific: data types, parameters,
shapes, or platforms outside a provider's reviewed region remain on CPU with a
visible CPU decision or fallback. Public visibility does not imply that every
GPU, operating system, dtype, parameter region, or workload has been qualified.

**Sigma Filter** is an edge-preserving Lee filter compatible with the documented
behavior of Fiji's
[Sigma Filter Plus](https://imagej.net/ij/plugins/sigma-filter.html). It works
slice-wise over the resolved `YX` axes, uses nearest/clamped borders, and treats
every channel and leading stack index as an independent plane.
`channel_axis=None` follows VIPP's scalar-default convention; no ROI or mask
input is part of the version-1 node. The public CPU contract accepts finite
native-endian `uint8`, `uint16`, and `float32` data, preserves shape and dtype,
and exposes radius 0.5–10, non-negative sigma width, minimum-pixel fraction
0–1, and the documented outlier-aware fallback. Unsigned results use
Fiji-compatible half-up rounding.

The CuPy implementation scans each circular footprint twice in one fused
`RawKernel`, keeps image-sized data resident, and does not build an image-by-
footprint sliding-window tensor. Its exact public region is the same native-
endian finite `uint8`/`uint16`/`float32` parameter and axis surface, with
complete finite extrema facts and a float32-square overflow guard for
`float32`. Non-native byte order fails closed before accelerator transfer.
Integer output must be bitwise equal; float32 uses a tight versioned gate plus
explicit adversarial selection/fallback tests. Kernel arithmetic disables fused
multiply-add and requests precise divide/square-root. Because NVRTC can still
force flush-to-zero behavior, explicit bit conversions preserve float32
subnormal samples, squares, and outputs rather than silently changing a
threshold decision. GPU progress advances only after each 64-row tile is
synchronized; cancellation occurs between tiles. Calls outside the reviewed
region, missing CUDA/CuPy, and unqualified platforms visibly remain on CPU.

The source-current full-profile RTX 5090 record passed all 10 exact admission
cases, all 10 matched rejection cases, cancellation/cleanup, and bitwise parity
for all 18 timed workloads. Representative transfer-inclusive speedups were
23.57x for a 512² radius-0.5 plane, 55.23x for a 512² radius-2 plane, 170.95x
for a 2048² radius-10 plane, and 93.62x for an 8×512² radius-2 stack. On this
host, radius 0.5 first cleared both Auto gates at 512²: its 20.13-ms absolute
saving just exceeded the 20-ms gate, while its paired 95% speedup lower bound
was 19.58x against the 1.20x gate. Radius 2 also cleared at 512²; radii 5 and 10
cleared at the smallest tested 256². These are machine-local observations, not
portable speed promises; see the
[canonical Sigma Filter evidence](docs/benchmarks/sigma-filter-cupy-windows-rtx5090.md).

The scientific reference is the Lee 1983 sigma-filter algorithm. Frozen
unsigned-integer fixtures were generated independently by executing the
published ImageJ plugin bytecode, rather than by reusing VIPP's Python oracle.
VIPP intentionally differs from the published plugin in two narrow, tested
places: it uses exact `ceil(footprint_count * minimum_fraction)`, and clamps a
cancellation-induced negative population variance to positive zero before the
square root. See the
[Sigma Filter implementation record](docs/gpu-phase4-sigma-filter-implementation-report.md)
for formulas, provenance, evidence, limitations, and timings.

Canny preserves VIPP's float32 plane conversion, constant-boundary Gaussian and
Sobel arithmetic, bilinear non-maximum suppression, eight-connected hysteresis,
quantile semantics, leading blocks, and explicit RGB/RGBA luma conversion. Its
initial public GPU region accepts bool, `uint8`, and `uint16` inputs with
canonical sigma 0 through 12. Authored `float32` Canny remains on CPU because
CUDA subnormal flush-to-zero can change final edge bits even for finite inputs.
Otsu preserves exact native integer
levels up to the existing 65,536-level guard, NumPy float histogram edges and
first-maximum tie breaking, finite-value handling, boolean identity, stack/slice
scope, RGB/RGBA luma conversion, and the strict `image > threshold` mask rule.
Its bounded atomic histogram avoids CuPy/CUB's device-occupancy-dependent wide-
histogram workspace while retaining exact counts.
Both providers return an exact boolean mask, report only synchronized progress,
and visibly use CPU outside their admitted regions. In the source-current schema-v3
[canonical RTX 5090 record](docs/benchmarks/canny-otsu-cupy-windows-rtx5090.md),
all 28 admission cases were bitwise exact. On the 8x1024x1024 `uint16` stack,
Canny measured 0.6812 seconds on CPU versus 0.0349 seconds GPU end-to-end
(19.51x), while Otsu measured 0.0455 versus 0.0077 seconds (5.92x). The
privacy-redacted 8.51-million-voxel ND2 volume measured 16.40x and 5.28x,
respectively. Schema v3 binds the evidence to source fingerprints and strictly
limits private-source metadata; these remain short machine-local screens, not
portable performance guarantees or saved optimizer choices.

Basic GPU **Measurements** accepts native non-negative `int32` labels in
resolved 2D/3D spatial layouts, including leading blocks and sparse positive
IDs. The intensity-aware node additionally accepts a same-shape native `bool`,
`uint8`, `uint16`, or finite `float32` image. The first promoted region covers
the existing basic morphology and basic morphology-plus-intensity schemas;
extended shape, axis, boundary, ratio, and moment columns visibly remain on CPU.
Valid non-negative integer label arrays outside native `int32`, unsupported
intensity dtypes, non-finite float32 intensity, an unqualified cuCIM runtime, or
insufficient VRAM produce a specific CPU decision or fallback rather than an
implicit cast or a changed table. Boolean/non-integer label domains, negative
labels, and invalid input shapes are errors for the CPU and GPU operations; VIPP
does not misreport those invalid inputs as fallbacks.

The cuCIM provider returns a private resident packed `float64` matrix. VIPP
copies it to the host, cleans up the CUDA scope, and only then runs the mandatory
finalizer that reconstructs the exact `TableData` schema, column/row order,
units, calibration, and public scalar types. Finalizer time and staging memory
are included in optimization, and the public table always ends device
residency. Tiny cuCIM region-properties/Euler lookup caches are primed once in
a separate module-owned pool; every transactional execution pool must still
return to zero.

The source-current RTX 5090 record passed all 11 admission cases, all 11 matched
rejections, both lifecycle cases, and clean private-pool teardown. Full-public
speedups were 1.87x for 1024² morphology, 16.37x for 2048² morphology plus
`uint16` intensity, 7.28x for a 32×256×256 intensity volume, and 23.90x for a
64×512×512 confocal-like intensity volume. CPU was correctly faster for small
256²/512² images and for the measured 6×512² float32 intensity stack. These are
machine-local screens: `Auto` and `Find fastest pipeline…` compare the exact
workload and pipeline residency context instead of assuming GPU always wins.
See the [Phase 6 Measurements implementation record](docs/gpu-phase6-measurements-implementation-report.md)
and [canonical evidence](docs/benchmarks/measurements-cucim-windows-rtx5090.md).

Explicit **Convert
Dtype** nodes can unlock these GPU candidates and may improve
acceleration across a longer GPU-resident segment. Choose
`Scaling = Preserve` when the intention is to keep the numeric values; the
node's default `Rescale` deliberately remaps the intensity range. VIPP never
inserts this cast merely to win a benchmark. With Preserve, a `float32` value
exactly represents integer values with magnitude up to 2^24 (including every
`uint8` and `uint16` value), but conversion still changes the workflow's public
data representation.
Review downstream ranges, thresholds, rounding/output semantics, file writers,
and RAM/VRAM use; `float32` also requires twice the storage of `uint16`. Benchmark
the exact converted pipeline rather than assuming conversion will be faster.

Custom mode also exposes the review-first `Find fastest pipeline…` analysis.
After cancellation cleanup, it can establish a private fresh baseline for the
current graph and parameters when the retained display is previous or stale;
the user does not have to publish an ordinary replacement run first. It works
from detached source data
and compares every scientifically eligible CPU/CuPy/cuCIM implementation for
every **unlocked** node. The implementation currently in use is the starting
assignment, not an optimizer constraint. Only a separate, explicit node lock
means “keep this implementation,” and applying a winning assignment does not lock
it automatically. The lock preserves the actual implementation captured for
that analysis; it does not silently turn a broad `Best GPU` or library preference
into a portable machine-specific exact pin. A node following pipeline policy has
no explicit choice to preserve and therefore cannot be locked until the user
selects a per-node choice. Exact complete node evidence is reused only when the
workload bytes/shape/dtype/parameters, scientific software stack, implementations,
device/environment, memory scope, and measurement policy still match. Otherwise
the analysis runs parity first, screens timing at three paired rounds, and extends
to seven or fifteen only for a close or uncertain comparison. Complete-pipeline
timing starts at five paired rounds and extends to seven or fifteen only until
the result is decisive or the analysis reports it as inconclusive. Saved node
timing never
replaces fresh whole-pipeline parity before a changed assignment can be offered.
If the current assignment wins, VIPP reports that as a successful result rather
than an optimization failure.

When a synchronized GPU candidate already has enough repeat measurements to be
a reliable incumbent, the optimizer may stop a cooperative CPU warm call after
its elapsed time exceeds the incumbent's one-sided confidence bound plus a
material margin. The report records `CPU > ...; stopped early`: this is a
**censored lower bound**, not an exact CPU timing, and it is not reusable as
durable timing history. Parity is still established independently, directional
transfers remain in the graph-wide cost model, and any changed modeled
assignment must pass final paired whole-pipeline validation before it can be
offered. If the already-current assignment remains the winner, its fresh
baseline plus parity and conservative exact-or-censored comparison evidence is
reported without inventing a redundant paired alternative run.
The analysis dialog separates **overall progress** from the **current
operation**. Overall progress follows the complete analysis, while the second
bar names the node, implementation, phase, and timing round currently being
measured. The selectable time limit is elapsed wall-clock time, not a RAM or
VRAM budget. Multi-plane background subtraction reports each completed plane
for both CPU and cuCIM; its current-operation bar advances through those planes
and starts over for each parity, warmup, or timed invocation. A cuCIM plane is
reported only after its output has been synchronized. Richardson-Lucy reports
completed iterations for each leading 2D/3D block, synchronizes before each
reported checkpoint, and checks cancellation between iterations.
Richardson-Lucy TV uses the same truthful per-block/per-iteration checkpoint
contract. Operations
implemented as one monolithic NumPy, SciPy, CuPy, or cuCIM call have no truthful
intermediate milestone, so their current-operation bar can remain unchanged until that call
returns even though work is continuing; this pause alone does not mean the
analysis is stuck.

Reaching the time limit does **not** mean the current pipeline is optimal: it
means that no fastest assignment was determined within the selected time. VIPP
does not change any node settings in that case. Complete exact-workload node
records remain available for a later identical analysis, but partial timings
from the node that was interrupted are discarded. Retry with a longer time
limit; the timeout result identifies the stage and node that consumed the
remaining time and reports which completed evidence can be reused.

The analysis measures synchronized transfer costs and usable
VRAM, solves one graph-wide CPU/GPU assignment, and then validates the complete
current and proposed assignments for changed-node plus affected
observable-boundary parity and paired end-to-end benefit. Every private
validation run must report the exact requested implementation map and
environment, no fallback, and clean accelerator teardown. It refuses to
make a proposal when evidence or identity is incomplete/stale, memory is not
admissible, the graph contains an unsafe retained writer path, or the measured
gain does not exceed the greater of 5% or 10 ms with a lower confidence bound
above 1.0. Analysis changes nothing; a reviewed proposal is rechecked against
graph, source, compute intent and locks, actual assignment, exact source
bytes/metadata/image state, and a fresh probe of the exact candidate environment
before one undoable apply,
after which only affected branches are invalidated.

GPU work for one runtime/device is serialized by a fair process-wide
accelerator lease. Execution, transfer measurement, and node/pipeline
optimization therefore cannot unknowingly contend for the same CUDA device;
cancellation and the one absolute analysis deadline also apply while waiting
for the lease. Different runtime/device keys remain independent.

Validated GPU candidates are normally visible in the core admission model and
the normal UI; only unfinished or unvalidated providers remain
`developer_hidden`. This remains release- and operation-scoped support rather
than a blanket cross-platform GPU claim. The current optimizer is deliberately limited
to a calculated, writer-free scientific subgraph, one accelerator runtime, and
single-output nodes supported by exact node benchmarking. Ordered multi-input
nodes such as Richardson-Lucy and Richardson-Lucy TV are supported;
multi-output, multi-runtime,
side-effecting, and incomplete workloads still fail closed. Unifying every UI
optimizer input into one immutable application snapshot remains a named
hardening task rather than a completed claim. See the
[production GPU plan](docs/gpu-production-implementation-plan.md) for the
CPU/Auto/Prefer-GPU/Custom design, per-node and whole-pipeline benchmarking,
fallback,
memory, and promotion rules. The
[Phase 1 implementation record](docs/gpu-phase1-implementation-report.md)
summarizes the code, exact admitted matrix, validation evidence, and deferred
gates. The
[Phase 2B Richardson-Lucy implementation record](docs/gpu-phase2b-rl-implementation-report.md)
records the new provider, benchmark/lease substrate, exact parity policy,
limitations, and ordered next work. The
[Phase 2C Richardson-Lucy TV implementation record](docs/gpu-phase2c-rl-tv-implementation-report.md)
records the preserved nonlinear contract, separate lambda-zero and positive-TV
profiles, validation evidence, and remaining promotion gates. The
[Canny and Otsu implementation record](docs/gpu-phase3-canny-otsu-implementation-report.md)
records the exact-mask contracts, initial public regions, rejected raw cuCIM
Canny route, lifecycle policies, and real-device evidence protocol. The
[Sigma Filter implementation record](docs/gpu-phase4-sigma-filter-implementation-report.md)
records its clean-room CPU contract, independently frozen Fiji evidence, fused
CuPy implementation, exact public region, lifecycle evidence, and measured
crossovers. The
[Connected Components implementation record](docs/gpu-phase5-connected-components-implementation-report.md)
records exact SciPy `int32` IDs, the resident CuPyX path, block-boundary
lifecycle, memory model, CPU fallbacks, and machine-local timing interpretation.
The
[basic Measurements implementation record](docs/gpu-phase6-measurements-implementation-report.md)
records the typed-table boundary, exact promoted schema, visible CPU regions,
cuCIM lifecycle design, and workload-dependent CPU/GPU timing evidence.
The machine-local
[large-stack Richardson-Lucy timing summary](docs/benchmarks/rl-cupy-performance-windows-rtx5090.md)
compares synchronized CPU and transfer-inclusive CuPy execution on the private
representative ND2 volume and 16.8/67.1-million-voxel 3D shape stresses, with
paired median speedups of 62.79x, 76.22x, and 88.26x, respectively. The
[Richardson-Lucy TV timing summary](docs/benchmarks/rl-tv-cupy-performance-windows-rtx5090.md)
records 59.79x and 99.80x paired median speedups for the same private
8.51-million-voxel volume and a 16.78-million-voxel shape stress at the exact
positive shipped profile. The
[Canny/Otsu timing summary](docs/benchmarks/canny-otsu-cupy-windows-rtx5090.md)
records their separate 28-case exact-mask admission, synchronized timing,
memory-bound, cancellation, and zero-residue cleanup evidence.

Structural cache reuse also fails closed on exact scientific context: source
bytes/state and revision, node parameters and incoming topology, chained
upstream result identity, and the actual versioned implementation must all
match. Changing a downstream preference does not invalidate an exact upstream
cache, while in-place source changes and stale upstream parameters do.

Use the checked-in setup helper to create a dedicated Python 3.12 environment.
It pins one CUDA major, refuses mixed CuPy distributions, installs only into the
named virtual environment, runs `pip check`, and finishes with real Gaussian,
median, and signal-convolution kernels. A successful run writes a strict
provenance record inside that environment; cuCIM remains unavailable if the
record is missing, malformed, or no longer matches the installed wheel. Inspect
the exact commands without writing first if desired:

```powershell
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_dev.ps1 --track cuda13 --plan-only
powershell -ExecutionPolicy Bypass -File scripts/setup_gpu_dev.ps1 --track cuda13
.\.venv-gpu-cu13\Scripts\python.exe -m napari_vipp.core.compute_diagnostics --track cuda13
```

On Linux, the same helper can prepare an evidence environment through the shell
wrapper:

```bash
bash scripts/setup_gpu_dev.sh --track cuda13 --plan-only
bash scripts/setup_gpu_dev.sh --track cuda13
./.venv-gpu-cu13/bin/python -m napari_vipp.core.compute_diagnostics --track cuda13
```

The current executable Phase 1/Phase 2B/Phase 2C policy admits only the validated
native-Windows matrix. Linux preparation is available for the pending clean-host
validation, but GPU execution intentionally fails closed there until that
evidence is reviewed.

The base package supports CPython 3.12 and 3.13, but the initial GPU validation
matrix is deliberately CPython 3.12 only. Installing the base package or CuPy
on a newer interpreter is not a VIPP GPU support claim; each Python minor must
pass the clean-install, real-kernel, scientific-parity, memory, and cleanup
gates first.

Exact GPU parity is also defined against the authoritative CPU scientific stack.
The current public Windows region requires NumPy 2.5.1, SciPy 1.18.0, and
scikit-image 0.26.0. VIPP records those versions in the compute-environment
fingerprint and visibly keeps nodes on CPU if any are missing or different;
broader dependency versions require their own parity matrix rather than an
implicit compatibility claim.

Public admission currently matches the recorded native-Windows host exactly:
CUDA runtime API 13.2 (`13020`), driver API 13.3 (`13030`), CuPy/CuPyX 14.1.1,
and an NVIDIA GeForce RTX 5090 with compute capability 12.0. A different CUDA
runtime, driver, model, or compute capability visibly keeps the node on CPU
until that environment has its own reviewed evidence. Provider-level developer
qualification can still run directly outside this public gate.

The machine still needs a compatible NVIDIA driver. Select `--track cuda12` for
the separate `.venv-gpu-cu12` qualification-only environment; CUDA 12 is
outside the current public admission region. The project also
publishes platform-marked `gpu-cuda12` and `gpu-cuda13` extras. Those extras pin
the scientific stack used by admission; the checked-in setup helper additionally
produces a detailed development provenance record and verifies real kernels.
Never install the CUDA 12 and
CUDA 13 CuPy distributions into the same environment. If diagnostics report an
unavailable runtime, they print a copyable setup command; VIPP's CPU path remains
usable.

The scientifically validated cuCIM background provider is a normal public
candidate in its exact admitted Windows environment. That provider status is
separate from distribution: the pinned native-Windows cuCIM skimage
source-built wheel and its installation route remain experimental. The current
reproduction path is documented in the
[cuCIM source evaluation](docs/cucim-windows-source-evaluation.md) and
[`scripts/build_cucim_windows.ps1`](scripts/build_cucim_windows.ps1); it omits
Clara I/O and is not yet a general user-facing install route. The builder uses
its own temporary environment and reports the wheel path/hash; it does not
install cuCIM into `.venv-gpu-cu13`. Install a reviewed local build explicitly with
`--cucim-wheel <path> --cucim-sha256 <digest>`; both values are required and the
helper verifies the file immediately before installing it. CUDA acceleration
targets validated Windows systems first, with native Linux next. macOS
continues to use VIPP's CPU path
while an M1 Max Metal/MPS/MLX provider is investigated; Apple unified memory
must be reported as one shared budget, not RAM plus VRAM.

For production collection replay in this alpha, let `Batch workspace...`
write `vipp_batch_pipeline.py` and its config/workflow companions, then run:

```powershell
.\.venv-gpu-cu13\Scripts\python.exe .\results\vipp_batch_pipeline.py --progress
```

The runner uses the saved compute request unless explicit CLI overrides are
provided. See [Durable GPU execution](docs/durable-gpu-execution.md) before
using `--compute-mode`, `--fallback-policy`, or `--node-preference`.

Run the required checks:

```bash
python -m npe2 validate src/napari_vipp/napari.yaml
python -m ruff check .
python -m pytest
```

Launch a development instance from the repository with `./vipp`; it uses the
project's `.venv-macos` environment directly, so shell activation is not
required. The installed `vipp` command and `python -m napari_vipp` are also
supported. To open the synthetic sample with a pipeline run already completed, use
`python scripts/launch_vipp_sample.py`. The
[architecture reference](docs/architecture.md) explains the graph, metadata,
execution, persistence, and UI boundaries.

Contributions are welcome. Read [CONTRIBUTING.md](CONTRIBUTING.md) before
opening a pull request, use [SUPPORT.md](SUPPORT.md) for help and issue-reporting
guidance, and report suspected vulnerabilities privately through
[SECURITY.md](SECURITY.md). All project interactions follow the
[Code of Conduct](CODE_OF_CONDUCT.md).

## 0.13 Alpha Highlights

`0.13.0a1` is the current alpha. It is VIPP's first usable, evidence-gated GPU
release while retaining CPU as the portable scientific reference path:

- toolbar `CPU`/`Auto`/`Prefer GPU`/`Custom` policy, per-node
  CPU/CuPy/cuCIM choices and
  actual-run badges, setup diagnostics, RAM/VRAM reporting, node benchmarking,
  and a review-before-apply whole-pipeline optimizer;
- public-candidate GPU regions for background subtraction, median, Gaussian,
  Richardson-Lucy, Richardson-Lucy TV, Canny, Otsu, Sigma Filter, connected
  components, and basic measurement profiles;
- one execution contract across interactive calculation, durable collection
  batch, generated Python/CLI, and export, including exact implementation
  provenance, nested progress, cooperative cancellation, memory admission,
  structured OOM fallback, cleanup, and atomic publication;
- workflow schema 4 and batch-config/manifest schema 2 for portable compute
  intent, with schema-3 workflows and version-1 batch configs migrating to
  explicit CPU;
- independent workflow tabs, high-resolution colocalization scatter tools,
  live source subtitles, draggable tunnel rerouting, and retained napari
  camera/slice/display state during node tuning;
- Low/Standard/High/Very High thumbnail backing detail, responsive sampled
  Slice contrast,
  exact resolution-independent Stack Percentile histograms/native Min-max
  reductions, conservative adaptive CPU/CuPy presentation routing, separate
  per-node thumbnail-contrast footers, and truthful progress/cooperative
  cancellation; and
- ND2 ordered-axis metadata correction, Crop Stack type preservation, new
  Sigma Filter and ImageJ Auto Threshold nodes, plus substantial cache,
  optimizer, progress, cancellation, and publication hardening.

GPU support is not complete or generally cross-platform in this alpha. Linux
and RTX 40-series qualification, Apple acceleration, general cuCIM packaging,
and additional node providers remain planned. Colocalization and ImageJ
compatibility work also changes some numerical results; review the scientific
compatibility notes before comparing old and new analyses. See the categorized
[0.13.0a1 release notes](CHANGELOG.md#0130a1---2026-08-04), the
[upgrade and workflow contract](docs/user-guide.md#save-workflow-json), and
[planning.md](docs/planning.md) for the remaining milestones.

## Citation, Acknowledgement, And License

If VIPP contributes to your work, acknowledge `napari-vipp` and link to the
[project repository](https://github.com/rensutheart/napari-vipp). Citation
metadata is available in [CITATION.cff](CITATION.cff); a DOI or manuscript
citation can be added when available.

napari-vipp is distributed under the BSD 3-Clause License. See
[LICENSE](LICENSE) for the full terms.
