# napari-vipp Planning And Roadmap

Last reviewed: 2026-08-14

This is the concise source of truth for VIPP's product direction, active
priorities, and intended release order. Detailed implementation contracts and
completed engineering records belong in the specialist documents linked below;
version-by-version delivered detail belongs in the [changelog](../CHANGELOG.md).

## Product Direction

VIPP is a napari-native visual workflow builder for bioimage analysis. The graph
canvas is the primary work surface: users should be able to build, inspect,
reuse, export, batch-run, and publish workflows without losing the connection
between pixels, semantic axes, calibration, acquisition metadata, tables, and
provenance.

The product is organized around complete scientific workflows:

- segmentation and label cleanup for 2D and true 3D data;
- object, intensity, mesh, and skeleton measurements;
- two-channel pixel and object colocalization;
- multi-channel, z-stack, time-lapse, and multi-scene fluorescence data;
- native PSF generation and PSF-aware restoration/deconvolution;
- microscope acquisition import with normalized axes, channels, objective, and
  scale metadata; and
- reproducible interactive, batch, generated-Python, and command-line
  execution with explicit outputs.

PSF generation, deconvolution foundations, and optional microscope-reader
routing are part of the 0.11 baseline; explicit batch configuration and
provenance are part of 0.12. Version 0.13.0a1 brought the first usable,
evidence-gated GPU regions into the ordinary product; a2 corrected fresh-graph
GPU planning, a3 enabled local qualification on compatible secondary NVIDIA
hardware, and a4 replaced the device-model allowlist with the current
architecture and environment gates. Compute intent and exact implementation
provenance now span interactive, batch, generated Python/CLI, and export
execution. Version 0.13.0a5 added branded launch profiles, the non-mutating
Windows installer planner, its transactional apply engine, and a per-user
Windows bootstrapper. Version 0.13.0a6 adds atomic insertion before a named
tunnel, multi-node selection and movement, validated graph-fragment copy/paste,
exact-operation value transfer, clearer compute diagnosis and qualification,
and the first shared multi-series source slice.

The differentiator is not the number of nodes or GPU badges. VIPP should make a
complete workflow portable across CPU and supported accelerators, preserve its
scientific meaning, explain every automatic decision or fallback, and produce
enough evidence for another person to review or reproduce the run.

Near-term work therefore has six connected goals:

1. complete and qualify the novice-first Windows installer distribution;
2. harden the 0.13 field experience and broaden the validated compute matrix;
3. complete coherent GPU-resident workflow regions rather than isolated nodes;
4. establish one first-class model for files, series, scenes, groups, wells,
   fields, and selected axis items;
5. make large OME-Zarr data deliberate through pyramids, lazy/chunked
   capabilities, and materialization safeguards; and
6. turn validation, batch recovery, graph reuse, and reproducibility export
   into ordinary product features.

Registration, model-backed segmentation, stitching, specialist surface/mesh
exchange, and AI-assisted graph authoring remain later milestones. They should
not displace the source, scale, validation, and reproducibility foundations.

## Reference Documents

- [architecture.md](architecture.md): implementation architecture, data model,
  persistence, and known seams.
- [user-guide.md](user-guide.md): current end-user workflow behavior.
- [node-roadmap.md](node-roadmap.md): implemented node families and candidate
  scientific operations.
- [io-user-guide.md](io-user-guide.md) and [ome-io-plan.md](ome-io-plan.md):
  supported I/O, source inspection, `ImageDataset`, and OME architecture.
- [cache-and-memory.md](cache-and-memory.md): cache modes, memory guard, and the
  current eager-versus-lazy boundary.
- [psf-and-deconvolution-plan.md](psf-and-deconvolution-plan.md): PSF,
  deconvolution, microscope metadata, and validation requirements.
- [gpu-production-implementation-plan.md](gpu-production-implementation-plan.md):
  CPU/Auto/Prefer-GPU/Custom behavior, implementation admission, residency,
  fallback, benchmarking, memory, packaging, and detailed GPU delivery records.
- [durable-gpu-execution.md](durable-gpu-execution.md): shared interactive,
  batch, generated-Python/CLI, and export execution semantics.
- [cucim-windows-source-evaluation.md](cucim-windows-source-evaluation.md) and
  [cucim-windows-port-plan.md](cucim-windows-port-plan.md): the pinned private
  Windows build, measured promote/defer decisions, and possible upstream/full
  Clara direction.
- [measurement-workflows.md](measurement-workflows.md),
  [skeleton-nodes.md](skeleton-nodes.md), and
  [object-mesh-morphology-plan.md](object-mesh-morphology-plan.md): measurement
  workflow guidance.
- [analytical-phantom-validation.md](analytical-phantom-validation.md),
  [colocalization-method-notes.md](colocalization-method-notes.md), and
  [research-and-publication.md](research-and-publication.md): current validation
  and publication-facing evidence.
- [mitomorph-feature-parity.md](mitomorph-feature-parity.md): MitoMorph-inspired
  feature parity tracking.
- [desktop-startup-and-installer-plan.md](desktop-startup-and-installer-plan.md):
  branded launch profiles, the in-napari loading host, the separate local-build
  cuCIM bundle, and the staged Windows/Linux/macOS installer plan.

## Current Public Baseline

Current alpha release: `0.13.0a6`.

The public baseline includes:

- workflow schema 4 with portable CPU/Auto/Prefer-GPU/Custom intent and durable
  per-node preferences;
- collection batch schema 3 with source-axis declarations, explicit outputs,
  deterministic planning, cancellation, checkpoints, manifests, and exact
  implementation provenance;
- one shared execution contract across interactive, batch, generated Python,
  CLI, and exported outputs;
- independent workflow tabs, searchable graph authoring, typed ports, tunnels,
  notes, undo/redo, manual/cached nodes, isolated tuning, and memory controls;
- OME-NGFF-inspired image/table metadata, OME-TIFF/ImageJ TIFF/common
  raster/NumPy I/O, local OME-Zarr 0.4/0.5 I/O, and initial optional microscope
  reader routing;
- PSF generation and preparation, RL and RL-TV deconvolution, segmentation,
  label cleanup, measurements, skeleton/network analysis, colocalization, and
  spatial-association workflows; and
- reviewed accelerator regions for background subtraction, median, Gaussian,
  RL/RL-TV, Canny, Otsu, Sigma Filter, connected components, and basic object
  measurements in their exact admitted environments and workloads.

Important present limits:

- CPU remains the portable scientific reference and fallback path;
- normal public GPU admission is deliberately narrow and is currently backed by
  exact native-Windows CUDA 13 / RTX 5090 evidence;
- the standard CUDA extras do not distribute or require cuCIM; Windows users
  build the pinned optional wheel locally for themselves;
- interactive sources and most operations remain eager even where a reader can
  expose lazy data;
- one physical file may contain several scientifically distinct source items,
  but that identity is not yet modeled consistently across every reader and
  batch route; and
- validation is strong for calibrated morphology but uneven across
  colocalization, watershed, skeleton/network analysis, real PSFs,
  microscope-metadata normalization, interrupted batch replay, and OME-Zarr
  round-tripping.

## Ordered Product Priorities After 0.13.0a6

The order below reflects the post-0.13.0a6 perspective. Work with disjoint
ownership may proceed in parallel, but a later feature does not bypass the
scientific, compatibility, and operational gates of an earlier foundation.

### Delivery Baseline: Desktop Startup And Installation

The `0.13.0a5` launcher provides branded Automatic, CPU-only,
and Prefer-GPU graphical entry points with real startup milestones, plus a
lightweight loading host when the plugin is opened inside napari. A separate
deterministic Windows cuCIM bundle performs the verified pinned build locally
and contains no redistributable cuCIM wheel.

The Windows planner inspects managed CPU/CUDA and
explicitly selected existing-napari routes without changing packages, files,
shortcuts, or the registry. It records the interpreter, target, conservative
disk reserve, stable validation issues, top-level release requirement,
acceptance commands, shortcuts, and rollback boundary in deterministic schema
v1 JSON. Its transactional apply engine and per-user Windows bootstrapper are
included in `0.13.0a5`. Linux and macOS installers should reuse the same
headless environment-plan contract. The Windows `.exe`
must become the first installation route in the README, quick start, release
notes, and versioned manual; existing-napari, planner, and raw pip instructions
remain advanced routes. The signed Windows filename remains reserved for a
valid Authenticode artifact; an unsigned alpha must use `-UNSIGNED` and include
checksum-first Windows-warning instructions.
Linux GPU and Apple acceleration remain gated by their independent platform
qualification. See the [quick start](quick-start.md) and
[desktop startup and installer plan](desktop-startup-and-installer-plan.md).

The primary installer persona is a physiologist with no assumed knowledge of
Python, napari environments, CUDA packages, or dependency management. The
managed path must automatically recommend the safe CPU or qualified GPU setup,
describe only its practical effect, and use one **Install VIPP** confirmation.
Interpreter paths, package-resolution details, rollback boundaries,
existing-napari mutation, and manual compute selection belong under
**Advanced details**. Internal safety checks remain strict, but they must not
be presented as a chain of technical approvals the user has to understand.

### Delivery Baseline: GPU Qualification

Phase 1 is implemented headlessly and interactively:
CPU/Auto/Prefer-GPU/Custom and per-node/benchmark contracts, unified
execution, the dedicated CUDA development/doctor path, and production-parity
Rolling-Ball/Subtract Background, median, and 2D/3D Gaussian adapters. Phase 2B
also has a public-candidate ordinary CuPy Richardson-Lucy backend, ordered
multi-input exact benchmarking, and per-device accelerator coordination. Its
checkpoint-backed finite-float32, odd-PSF, default-safe envelope now covers authored
`filter_epsilon` values from `1e-12` through `1e-6` and 1 through 100
iterations under `rl-scientific-equivalence-v2`; parameters are never rewritten
to enter the region. Phase 2C adds a
public-candidate CuPy RL-TV provider without changing the existing formula,
sign, central-gradient stencil, initialization, padding, floor, or defaults.
Its lambda-zero profile inherits ordinary RL's strict gate; its initial
positive-TV profile covers only the exact shipped parameter tuple at the
measured 10- and 25-iteration points under a separately versioned nonlinear
parity study. Lambda-zero inherits ordinary RL's expanded region. Phase 3A adds
implemented, exact-mask CuPy/CuPyX Canny and Otsu providers with explicit luma,
stack/slice, progress, cancellation, memory, and CPU-fallback contracts. Their
validated regions are normal public `Auto`/`Custom` candidates;
`developer_hidden` is reserved for incomplete or unvalidated work.
Phase 4 adds the public clean-room Sigma Filter and its fused CuPy provider.
Phase 5 completes Connected Components with an exact CuPyX provider for boolean
2D/3D masks. It preserves SciPy-identical native `int32` IDs for face and full
connectivity, restarts IDs in every independent leading block, keeps compatible
segments resident, and exposes truthful block-boundary progress/cancellation.
Numeric nonzero-mask conversion, 1D labeling, oversized blocks, and unvalidated
environments visibly remain on CPU. A single plane or volume is currently one
atomic CuPyX call, so finer mid-volume progress/cancellation remains explicit
future work. Both phases are normal public candidates only inside their pinned
validated regions.
Phase 6 adds cuCIM candidates for the exact basic schemas of `Measure Objects`
and `Measure Objects + Intensity`, followed by an exact typed host-table
finalizer that preserves schema, order, units, missing-value behavior, and
deterministic provenance. Other measurement profiles remain visibly on CPU.
Normal public admission fails closed to native Windows, CUDA runtime API 13.2
(`13020`), a matching numeric driver API at least 13.3 (`13030`), an NVIDIA
CUDA device with compute capability at least 7.5, and the exact pinned Python,
scientific-stack, CuPy, provider, and cuCIM provenance gates. CUDA 12 remains
qualification-only and outside public admission. Auto, Prefer GPU, and Custom
use this same qualifying-device policy; the GPU model is recorded provenance,
not an exact-model admission gate. `Find fastest pipeline…` additionally
requires node parity and changed-boundary whole-pipeline parity before
proposing a forced preference. The applied preference is machine-local intent,
not reusable cross-device performance evidence, and a manually authored Custom
preference has no attached parity proof. Users should rerun Find Fastest after
any environment or workload change.

Floating-point results can vary slightly across GPU models and driver/JIT
combinations within an operation's declared tolerance; integer operations with
a bitwise parity contract remain exact. Reproducible publications should report
the VIPP version, GPU model, compute capability, driver API, CUDA runtime,
CuPy/cuCIM versions, authoritative CPU scientific-stack versions, workflow,
parameters, and actual per-node implementations recorded by execution
provenance.

RL-family v2 agreement uses NRMSE `<= 0.005` together with
`max_abs <= 1e-6 + 0.005 * max(abs(CPU reference))`, plus equal shape and
`float32` dtype, identical finite masks, completely finite results, and
nonnegative output for the default clipped contract. The previous ordinary-RL
NRMSE `2e-6` result and maximum ULP remain diagnostics, not admission gates.
This 0.5% margin is an engineering comparison against VIPP's CPU backend. It is
not a scientific-validity or image-quality threshold; PSF suitability,
iteration stopping, artifacts, recovered resolution, intensity fidelity, and
downstream measurements require separate validation.
Machine-local large-stack timing now records 45.03x, 85.06x, and 94.58x paired
median speedups on the RTX 5090 for one real 8.51-million-voxel ND2 volume and
16.78/67.11-million-voxel 3D shape stresses, respectively. Those short
descriptive results are not a reusable optimizer record or cross-platform claim;
see the
[raw and readable timing evidence](benchmarks/rl-cupy-performance-windows-rtx5090.md).
The matching positive-TV screen records 78.61x and 83.02x paired median
speedups for the private 8.51-million-voxel volume and a 16.78-million-voxel
shape stress; see the
[RL-TV timing evidence](benchmarks/rl-tv-cupy-performance-windows-rtx5090.md).
The source-current schema-v3 Canny/Otsu screen passed all 28 exact-mask admission cases.
On the 8x1024x1024 `uint16` stack, transfer-inclusive Canny and Otsu speedups
were 19.51x and 5.92x; on the privacy-redacted 8.51-million-voxel ND2 volume
they were 16.40x and 5.28x. Both providers passed bounded-memory,
cancellation, zero-residue cleanup, source-integrity, and strict private-metadata
checks. See the
[Canny/Otsu evidence](benchmarks/canny-otsu-cupy-windows-rtx5090.md); these are
single-host descriptive measurements, not portable Auto choices.
The toolbar now has the CPU/Auto/Prefer-GPU/Custom policy slice. Prefer GPU
considers every reviewed eligible accelerator region without applying Auto's
CPU-versus-GPU timing gate, while retaining transfer-economics placement so a
lightweight host view does not upload data that it will immediately discard.
Custom retains node choices and benchmarking. Actual backend badges and
the single message strip distinguish major and actionable paths by severity.
Optimizer UI lifecycle/snapshot hardening continues alongside the maintained
next order.
Completed but speed-inconclusive optimizer comparisons remain reviewable: the
current assignment is retained with no Apply action, while a node-grouped table
shows every tested implementation and offers optional compute, transfer,
first-run, memory, and evidence details. This prevents a whole-pipeline timing
tie from being mistaken for GPU ineligibility.
RL/RL-TV evidence ownership is isolated from broad shared-file hashes;
Measurements and durable batch/generated/export execution are implemented.
The maintained next order is native-Linux evidence, broader multi-device
performance characterization, M1 Max CPU qualification followed by an
Apple-provider study, general cuCIM/Clara packaging, explicit Convert
Dtype/residency bridges, and additional reasonable GPU node regions. See the
[GPU production plan](gpu-production-implementation-plan.md) and
[Phase 5 record](gpu-phase5-connected-components-implementation-report.md).

### Continuous Delivery Gates

Two delivery lanes apply continuously across every numbered priority; they are
not deferred until a later milestone:

- **Scientific validation:** every new operation, provider, reader contract, or
  execution mode ships with evidence proportionate to its claim. CPU-oracle GPU
  parity, external/golden-method comparison, metadata round-trips, and
  publication-grade biological validation are distinct claims and must not be
  substituted for one another.
- **Enabling architecture and quality:** split accelerator declarations into
  family-owned modules before parallel GPU expansion; generate one current
  capability manifest while retaining immutable promotion records; consolidate
  parity, lifecycle, memory, and performance evidence in the shared admission
  harness; retain schema/provenance compatibility goldens and CPU-only loading
  tests; and track transfer, memory, and performance regressions on stable
  hardware. Converge foreground and background calculation on one execution and
  result service while preserving the low-latency interactive path. Extract
  source-inspection, run-coordination, and generated-layer controllers only when
  that work enables a roadmap outcome. Define first-class points, transforms,
  and surfaces before adding dependent algorithm families.

### 1. Field Hardening And Supported Compute Reach

Priority 1 engineering is delivered in `0.13.0a6`:

- the a5 published unsigned installer, release manifest, checksum sidecar, and
  no-wheel cuCIM ZIP were independently rechecked; a6 carries the same
  checksum-first contract and exposes the shortest official route in the
  primary guides;
- CI defines genuinely clean wheel and source-archive installations across
  Windows, Linux, macOS, and both supported Python versions;
- Compute Doctor separately reports CUDA runtime health, optional CuPyX/cuCIM
  usability, and the current live VIPP admission catalogue. It gives one repair
  action and exports an atomic, recursively privacy-redacted support bundle;
- one strict real-GPU admission command accounts for all 13 public GPU
  implementations and requires owned parity, adversarial, metadata,
  input-integrity, memory, cancellation, cleanup, fallback, provenance, and
  transfer-inclusive timing evidence;
- the scheduled Windows cuCIM workflow reproducibly checks the public no-wheel
  bundle on hosted CI and defines a protected, explicitly enabled real-CUDA
  build/Doctor/admission route for a dedicated self-hosted runner;
- the Imaris `.ims`, Set Microscope Metadata, and multi-series collection-batch
  contributions from pull requests 8, 10, and 13 were absorbed through the
  shared source, metadata, execution, naming, and provenance contracts; and
- the optional-preview and cancellation reports in issues 11 and 14 pass the
  current Windows regression suite. They remain open until their macOS
  reporters or equivalent environments verify the released behavior.

A strict `quick` integration run for the a6 catalogue on the local RTX 5090
completed all 16 owners and all 130 implementation/facet mappings. Because it
ran from a dirty feature worktree, its temporary artifacts are integration
evidence only; they were not promoted into the canonical benchmark record or
used to broaden public support.

The integrated Convert Dtype Preserve region expanded the then-current
catalogue to 14 public GPU implementations, 18 executable owners, and all 140
required implementation/facet mappings. That strict RTX 5090 `quick` run
passed on 2026-08-13 in 989.4 seconds (aggregate SHA-256
`c31b230b1ccf67cfc2c5c65d66b55391bf6e421f0e5b71023b5d6463791cffe9`).
It remains dirty-worktree integration evidence until repeated from an immutable
candidate commit; it does not by itself broaden supported hardware or describe
later additions to the live catalogue.

The current mask-cleanup development slice expands the live strict catalogue
to 18 public GPU implementations and 23 executable owners. Its RTX 5090 quick
evidence passed exact parity, lifecycle, memory, provenance, and
transfer-inclusive timing checks from the feature worktree. The resulting
artifact remains temporary integration evidence until the suite is repeated
from an immutable candidate commit; it does not promote these Custom regions
to Auto or broaden the supported hardware matrix.

The remaining Priority 1 items are deliberately field qualification, not
unfinished implementation:

- run the exact published executable on a fresh qualified-CUDA Windows account,
  including live paths with spaces and non-ASCII characters;
- exercise exact-artifact cancellation and terminal network-failure rollback
  while proving that a previous working copy remains usable;
- complete the timed novice path through installation, Compute Doctor, a first
  example, owned data, save/reopen, visible fallback, and a small batch;
- register and protect the dedicated Windows CUDA canary runner before calling
  its scheduled real-GPU job operational; and
- collect reviewed evidence on at least one Windows RTX 40-series machine and
  one named native-Linux CUDA 12/13 environment. WSL2 remains separate evidence.

The [field-acceptance form](windows-installer-field-acceptance.md) records those
outcomes consistently and keeps every unrun check visibly pending. Earlier
development-installer runs, unit tests, or a different artifact remain useful
supporting evidence, never substitutes for the exact field check.

Release acceptance is not merely a successful CUDA import. A supported machine
must receive truthful admitted-node results, unsupported machines must receive a
clear CPU decision or repair path, and the base package must stay healthy on
Windows, macOS, and Linux without accelerator packages.

### 2. Complete Coherent GPU-Resident Workflows

GPU work should be chosen by end-to-end workflow value and transfer count, not
by the number of accelerated nodes or isolated kernel speed. The first target is
a standard-CUDA segmentation corridor that does not require cuCIM:

```text
Extract Channel
  -> Gaussian or Sigma Filter
  -> Binary Threshold
  -> Remove Small Objects / Fill Holes
  -> Connected Components
  -> Label Output
```

When a verified private cuCIM build is present, the enhanced corridor may add
device-resident Subtract Background before filtering and basic Measurements at
the end. Without cuCIM, VIPP must retain the CPU reference/fallback and must not
present the enhanced corridor as standard CUDA coverage.

Feature Sequence A's first release-blocking implementation wave contains only
what is needed to close and validate the standard corridor:

1. explicit `Convert Dtype`: the exact `uint8`/`uint16` to `float32` Preserve
   region and its visible one-click repair are integrated; bool, carefully
   bounded clip semantics, and any later rescale promotion remain separate
   evidence-gated regions;
2. the initial exact Extract Channel allocation-sharing view and scalar
   `float32` Binary Threshold regions enter as reviewed public Custom/Prefer-GPU
   candidates; Auto promotion still requires multi-device, transfer-inclusive
   evidence;
3. boolean Remove Small Objects and the exact fill-all-holes region enter as
   reviewed public Custom/Prefer-GPU candidates, while integer-label cleanup
   and positive bounded-hole sizes remain visible CPU regions; and
4. the existing exact int32 Connected Components output closes and verifies
   the label-output bridge without inventing another node.

Crop, Select Axis Slice, Reorder Axes, Clip, Mask Image, logical/arithmetic
operations, projections, broader binary morphology, label cleanup, additional
filters, and additional thresholds are ranked follow-ups. Promote them by
measured transfer/fallback hotspots in real workflows, not as an undifferentiated
first-wave catalogue.

The second wave may add distance transform, H-Maxima, Expand Labels, extended
measurements, and bounded colocalization regions after their exact scientific
contracts are validated. Watershed, bilateral/non-local-means filtering,
reduction-heavy Costes colocalization, skeleton/network algorithms, and mesh
work remain later or higher-risk because ties, label identities, floating
reductions, memory, and cancellation require stronger evidence.

Provider direction:

- CuPy/CuPyX is the ordinary implementation substrate for views, pointwise
  operations, filtering, thresholding, morphology, and residency bridges;
- cuCIM remains optional and operation-specific, preferred only where it offers
  unique coverage or a measured complete-adapter advantage, currently most
  notably rolling-ball background and basic measurement regions;
- VIPP will not host or redistribute the private native-Windows cuCIM wheel;
  each user builds the documented pinned source for their own environment. A
  future Clara/upstream contribution is a separate, time-boxed investigation,
  not a packaging promise;
- investigate a CuPy/CuPyX basic-measurement fallback and time-box an exact
  alternative rolling-ball study, but retain the CPU reference rather than
  weakening semantics to avoid a local cuCIM build; and
- never silently insert casts, reorder axes, alter parameters, or change
  dimensional meaning merely to make a GPU implementation eligible.

The first bounded one-click repair is **Add conversion** for exact
`uint8`/`uint16` to `float32` conversion with `Scaling = Preserve`. VIPP offers
it only when dtype is the sole blocker for an otherwise supported GPU path and
the current environment has a qualifying provider. Accepting the action inserts
a normal, visible **Convert Dtype** node on the affected input wire and records
one undoable graph edit; it does not rewrite other branches behind the user's
back. Before insertion it states that `float32` uses four times the input bytes
of `uint8` or twice those of `uint16`. The refreshed message must say
**GPU eligible**, not present eligibility
as a guarantee of GPU use: compute mode, workload policy, memory, runtime
health, and later graph changes may still select CPU or produce a visible
fallback. Lossy, clipping, rescaling, or otherwise unproved conversions require
review rather than this shortcut.

New providers should normally enter as reviewed Custom/Prefer-GPU choices, then
become Auto candidates only after multi-device, transfer-inclusive evidence.
The principal success measures are whole-workflow wall time, host/device transfer
count and bytes, observed peak VRAM, cancellation/cleanup behavior, visible
fallback rate, and scientific parity. A canonical linear GPU workflow should
aim for one host-to-device and one device-to-host boundary when only one terminal
output is retained. Previewing, branching to, or retaining intermediate results
creates additional legitimate boundaries. At a host entry, CPU Extract Channel
can reduce a multichannel source before upload; a resident GPU Extract Channel
instead keeps the full source allocation alive and exposes an allocation-sharing
view. Transfer count and transferred bytes must therefore both be reported.

### 3. First-Class Source Items And Acquisition Metadata

The LIF collection problem is one symptom of a broader model gap: one path is
not necessarily one scientific image. VIPP needs a durable `SourceItem` concept
that can represent:

- one standalone file;
- one series, scene, position, or acquisition item inside a file;
- one OME-Zarr image, label group, or selected multiscale level;
- one plate/well/field item; and
- one explicitly selected timepoint, channel, z-slice, or semantic-axis
  combination when iteration is requested.

The source-item identity must be stable enough to drive interactive selection,
batch rows, naming tokens, metadata-key pairing, source hashes, replay, and
provenance. It must retain container identity and item identity separately.

Required product behavior:

- a concise source inspector for series/scenes/positions, raw and effective
  axes, shape, scale/units, channels, objective/acquisition metadata, and an
  estimated materialization cost;
- selection and slicing before materialization when ND2, OME-Zarr, or another
  resource-backed reader can avoid decoding an entire acquisition;
- normalized mappings for axes, channels, wavelengths, objective NA and
  magnification, immersion/refractive index, detector/acquisition context,
  series/scene identity, and plate/well/field where present;
- explicit states such as supported, experimental, and metadata-incomplete;
- semantic-axis batch iteration and HCS traversal only through a saved,
  reviewable contract, never inferred silently from filenames or directory
  layout; and
- public/synthetic reader fixtures plus controlled private real-file evidence
  where licensing or size prevents redistribution.

The first compatible slice is now delivered: `.ims` uses the shared microscope
reader path, Set Microscope Metadata updates carried channel/acquisition facts
without changing pixels, and collection batch expands inspected multi-series
containers into named items used by preview, output naming, manifests, and
provenance. Those fields are not yet the complete durable `SourceItem` model;
the remaining work is to unify their identity across interactive selection,
OME-Zarr groups/levels, plate/well/field traversal, replay, and semantic-axis
iteration. That foundation also supplies the metadata needed for PSF generation,
deconvolution, calibrated measurement, output naming, and publication
provenance.

### 4. OME-Zarr Scale, Preview, And Lazy Execution

VIPP already reads and writes local OME-Zarr 0.4/0.5 data and can retain lazy
reader arrays internally, but ordinary interactive graph execution still
materializes complete selected sources. The next scale work is:

- generate useful multiscale pyramids and metadata for exported image datasets;
- select suitable pyramid levels for thumbnails and inspector previews while
  keeping analysis-resolution data explicit and unchanged;
- declare each operation as appropriate combinations of view-only, lazy-safe,
  chunkable, overlap/halo-dependent, global-reduction, eager-only,
  memory-heavy, and scale-aware;
- warn and request confirmation before an eager-only operation materializes an
  enormous lazy source;
- add chunked/tiled execution only where boundaries, overlap, global state, and
  exactness are explicitly defined—never through hidden analysis sampling;
- round-trip label colors and label-property tables where practical;
- calibrate memory estimates against observed host/GPU high-water use and avoid
  rejecting safe resident chains solely through accumulated conservative
  overestimation; and
- investigate anonymous HTTP access for public OME-Zarr datasets after local
  identity, cache, and failure semantics are stable.

Preview resolution, presentation statistics, and scientific analysis resolution
must remain visibly distinct. A faster thumbnail must never change an
operational result.

### 5. Workflow Health And Guided Corrections

The improved `QYX` batch experience establishes a general UI rule: present one
primary problem, recommend one safe action, and explain what VIPP changed.

The workflow-health direction is:

- replace walls of technical diagnostics and repeated per-node errors with one
  concise primary issue plus optional advanced details;
- choose sane defaults where evidence is sufficient, while making every
  persisted change visible and allowing an explicit opt-out;
- show automatic corrections as readable proposed or completed actions rather
  than requiring users to type internal grammar such as `QYX -> ZYX`;
- summarize source axes, calibration, selected item, output intent, estimated
  memory, and compute eligibility in plain language;
- provide a local exportable compute report answering what used CPU/GPU, what
  fell back, and why; and
- use the same issue/action model in interactive calculation, Batch workspace,
  Compute Doctor, generated runners, and saved provenance.

Automatic help must remain conservative: VIPP may explain or apply a reviewed
UI-level default, but it must not invent calibration, silently reorder pixels,
change scientific parameters, or hide incompatible source items.

### 6. Reproducibility Package And Resumable Batch

VIPP already creates the individual ingredients of a reproducible run. Add one
`Export reproducibility package` action that gathers a reviewed set containing:

- workflow JSON and canonical scientific hash;
- batch configuration and generated runner where applicable;
- environment and package records;
- manifests, item checkpoints, exact implementation provenance, fallbacks, and
  output digest links;
- stable source identities and metadata summaries;
- validation notes; and
- optional small reference outputs or previews explicitly labeled as such.

The package need not and should not silently include restricted raw image data.
Its inventory must say what is embedded, referenced, omitted, or privacy-
redacted, and the related files should be staged and published atomically where
practical.

Turn batch checkpoints from a recovery trail into an explicit automatic-resume
feature. Resume must:

- verify workflow/config hashes, schemas, source identities, destinations, and
  prior checkpoint integrity before work starts;
- preserve completed items and resume only incomplete, cancelled, or selected
  failed items;
- respect current collision and publication policy without silently
  overwriting completed outputs;
- record the relationship between the original and resumed run; and
- remain interruptible, checkpointed, and truthful when an operation itself is
  only cooperatively cancellable.

Maintain an immutable compatibility corpus for released workflow, batch-config,
manifest, execution-provenance, generated-runner, and unknown-GPU-preference
documents. Loading and migration must preserve the original, remain CPU-safe on
machines without accelerators, and demonstrate scientific-hash stability where
the semantics are unchanged.

### 7. Publication-Grade Scientific Validation Packs

GPU parity proves that an accelerator reproduces VIPP's CPU reference inside a
declared tolerance; it does not by itself prove that the method is biologically
or externally correct. Validation starts with every scientific delivery under
the continuous gate above; this priority turns that evidence into maintained,
publication-grade packs. Each pack should contain public or synthetic inputs,
expected values and tolerances, exact parameter mappings, regenerating scripts,
tables/figures, limitations, and publication-facing method text.

For RL-family floating-point admission, the declared 0.5% NRMSE and
peak-relative maximum-error bounds answer only whether two VIPP backends agree
closely enough. NRMSE normalization, ROI, scaling, and preprocessing must be
reported, and no single NRMSE cutoff should be reused as proof of restoration
quality. Publication-grade deconvolution packs therefore combine pixel/error
maps with forward-model residuals, frequency/resolution behavior, flux and
feature measurements, representative PSF/SNR variation, and downstream task
outcomes.

The maintained validation queue is:

- watershed and touching-object separation on geometric and microscopy-like
  phantoms, including split/merge and marker-QC scenarios;
- colocalization and ImageJ-threshold comparisons using deterministic overlap,
  threshold, ROI, native-range, and independently sourced golden cases;
- object association, nearest-distance, and event-localization assumptions;
- skeleton/network topology with known endpoints, junctions, cycles, branch
  lengths, and anisotropic spacing;
- real bead PSFs and representative 2D/3D microscopy deconvolution, including
  edge/crop guidance and measured-versus-generated PSF comparisons;
- microscope-reader metadata round-trips for axes, scale, channels, wavelength,
  objective, series/scene, and plate/well/field identity;
- interrupted, resumed, and large-collection batch replay; and
- OME-Zarr image/label/table/multiscale round-tripping.

The RACC numerical-core decision also remains open: keep the VIPP-owned
implementation, share a common core with the RACC plugin, or document the
intentional separation.

### 8. Graph Authoring Reuse And Insertion

The first implementation was manually accepted on Windows on 12 August 2026
and merged into the main development line. It treats graph reuse as a validated
workflow edit rather than as a visual duplication shortcut:

- Ctrl/Cmd toggle selection, Shift-additive selection, group movement, bulk
  deletion, and keyboard and context-menu copy/paste create new node IDs and
  preserve relative layout;
- only connections whose endpoints are both selected are copied, so a pasted
  fragment never gains a hidden dependency on the source workflow;
- internal named tunnels and attached notes are remapped, while cached arrays,
  accepted runtime decisions, benchmark evidence, pins, external connections,
  and other transient state are excluded;
- copied authored compute preferences and optimizer locks follow complete
  nodes, but `Paste values` retains the target node's execution intent and is
  available only for the exact same operation;
- clipboard content uses a size-limited, versioned VIPP MIME payload, is
  validated as a detached graph before mutation, pastes atomically as one undo
  action, and moves visibly on repeated paste;
- ordinary text copy/paste in editors is not intercepted by the graph; and
- a node can be inserted immediately before the source of a named tunnel from
  the tunnel menu, by dropping a palette operation on its source badge, or by
  dropping a genuinely loose existing node there. VIPP connects the old source
  to the inserted node and reroutes the same named tunnel and every subscriber
  in one rollback-safe edit. Ordinary wires from the old source remain intact.

The bundled `Graph Editing Acceptance Check` workflow records the intended
manual checks directly on the canvas. The complete Windows recipe passed during
the initial acceptance. Remaining cross-platform release evidence covers Linux
Ctrl behavior, macOS Cmd behavior, dynamic ports, invalid and older payloads,
external-dependency exclusion, and atomic failure paths; automated coverage
continues to protect single- and multi-node fragments, exact parameter
compatibility, tunnel subscriber preservation, repeated paste, and undo/redo.

## Versioned Roadmap

Version numbers after `0.13.0a6` are intentionally not assigned until field
evidence establishes the appropriate scope. Every release must ship with tests,
documentation, an example or validation artifact when appropriate, and
human-readable release notes.

### Released Milestones

| Version | Theme | Durable outcome |
| --- | --- | --- |
| `0.10.0a1` | Graph readability and interactive memory | Search, tunnels, notes, explicit axis/channel tools, branch-local reruns, cache modes, and memory controls. |
| `0.11.0a1` | PSF, deconvolution, and microscope-import foundation | Born-Wolf PSFs, PSF preparation, RL/RL-TV, optional-reader routing, and normalized first-pass acquisition metadata. |
| `0.11.0a2` | Workflow and release hardening | Atomic restore/export validation, complete example coverage, cross-platform CI, and reproducible release checks. |
| `0.11.0a3` | Exact and responsive large-image analysis | Exact bounded-memory statistics, native-range scientific behavior, background reruns, and responsive inspection. |
| `0.12.0a1` | Batch configuration, provenance, and explicit semantics | Saved configs, explicit outputs, deterministic planning, manifests, checkpoints, and failure isolation. |
| `0.12.0a2` | Interactive tuning and execution feedback | Isolated node tuning, execution frontiers, progressive previews, display-safe layer reuse, and PSF guidance. |
| `0.12.0a3` | Batch reliability and one-file setup | Direct plan-only batch launch, optional attached configs, safer cloud/Windows writes, and clearer Batch entry points. |
| `0.13.0a1` | Evidence-gated GPU execution and durable compute intent | Shared CPU/GPU execution across every surface, exact implementation provenance, independent workflow tabs, new analysis/UI tools, and safe generic-TIFF page interpretation. |
| `0.13.0a2` | Fresh-graph GPU planning correction | New and restored graphs receive truthful compute planning rather than retaining an incorrect initial CPU-only assumption. |
| `0.13.0a3` | Secondary NVIDIA qualification | Compatible secondary NVIDIA hardware can collect and apply local qualification evidence without turning one machine's result into a portable support claim. |
| `0.13.0a4` | Architecture-based GPU admission | The exact device-model allowlist was replaced by current compute-capability, runtime, driver, environment, provider, and scientific-region gates; the GPU model remains provenance. |
| `0.13.0a5` | Desktop launch and installer foundation | Branded Automatic, CPU-only, and Prefer-GPU launch profiles, an in-napari loading host, deterministic non-mutating install plans, a transactional apply engine, and a per-user Windows bootstrapper. |
| `0.13.0a6` | Graph reuse, clearer GPU diagnosis, and field hardening | Multi-node graph copy/paste and exact value transfer, insertion before tunnels, Compute Doctor 2.0, complete public-GPU admission orchestration, clean distribution checks, Imaris and multi-series source improvements, and microscope metadata editing. |

See the [changelog](../CHANGELOG.md) and versioned release notes for full
delivered detail.

### Installer Distribution Gate After `0.13.0a6`

Goal: make the safe path for an ordinary Windows microscopy user one download
and one double-click, without requiring napari, Python-environment, or terminal
expertise.

Release gate:

- a tagged `VIPP-Setup-<version>-Windows-x86_64-UNSIGNED.exe` is attached to
  the official GitHub release with a manifest, SHA-256, explicit `NotSigned`
  status, and checksum-first **More info > Run anyway** guidance;
- managed CPU is the default portable route, while eligible NVIDIA hardware
  can choose managed CUDA 13 and receive Automatic, CPU, and Prefer-GPU
  shortcuts;
- existing napari installation is clearly Advanced and cannot proceed until
  the selected environment and exact dependency changes pass review;
- dependency resolution is non-mutating, the user reviews location and package
  changes before Apply, and execution has progress, logs, cancellation,
  ownership, acceptance, and bounded rollback;
- the README, quick start, release notes, and versioned manual lead with the
  exact installer asset; terminal and planner commands remain secondary;
- clean CPU and real qualified-GPU installs pass from the tagged artifact on
  fresh Windows accounts, including spaces and Unicode paths; and
- cuCIM remains a separately downloaded optional local-build add-on after the
  standard CUDA environment passes acceptance.

Linux and macOS installers should reuse the same headless environment-plan
contract after their own platform qualification. A signed filename remains
reserved for a valid Authenticode artifact; an unsigned alpha must remain
explicitly labelled `-UNSIGNED`.

### Delivered Maintenance Alpha: `0.13.0a6`

Theme: field hardening and compute qualification.

Engineering delivered in this release:

- Compute Doctor distinguishes runtime health from actual VIPP admission and
  exports a strictly redacted support bundle;
- the `.ims`, microscope-metadata, and multi-series collection contributions
  are integrated through shared contracts;
- the base wheel and source archive have clean-install CI on every base OS,
  while the optional GPU extras and private cuCIM recipe have explicit,
  separately scoped checks;
- the accepted tunnel-insertion and graph-fragment editing implementation
  retains its atomicity, compatibility, and user-guide acceptance evidence; and
- Prefer-GPU planning now safely excludes deliberately loose connected graph
  fragments, matching ordinary CPU execution.

Remaining field release gate:

- exact-artifact Windows CPU and CUDA routes are independently reproduced on
  fresh accounts, including spaces and Unicode paths plus bounded rollback;
- the timed novice path and first small pilot are recorded;
- addressed batch preview/cancellation reports are verified by their reporters
  or equivalent macOS environments and then closed or retained with an exact
  remaining reproduction; and
- the protected real-CUDA canary is enabled only after its dedicated runner is
  registered and reviewed.

Qualification targets:

- produce reviewed parity/lifecycle evidence on at least one RTX 40-series
  Windows environment and one named native-Linux CUDA target; and
- if either target is not complete, leave the public support matrix unchanged
  and state the pending qualification precisely. Do not present an attempted
  target as delivered supported-compute reach.

### Feature Sequence A: Coherent Workflow Acceleration

Goal: keep a complete common segmentation/measurement corridor resident where
scientifically and operationally valid.

Release gate:

- explicit dtype conversion and selected view/pointwise bridges are available;
- the chosen morphology/label-cleanup corridor has CPU-oracle parity, memory,
  cancellation, cleanup, provenance, and cross-surface coverage;
- one canonical workflow demonstrates transfer-inclusive end-to-end benefit on
  more than one reviewed hardware class; and
- cuCIM remains optional, with CPU and independently eligible CuPy/CuPyX paths
  behaving honestly when it is absent.

### Feature Sequence B: Source-Aware Large Data And Graph Reuse

Goal: make multidimensional and multi-item acquisitions explicit while making
proven graph fragments easy to reuse.

Release gate:

- a saved source-item identity can distinguish a container from its selected
  series/scene/group/well/field or semantic-axis item;
- large local OME-Zarr data can be previewed without surprising full reads;
- exported OME-Zarr images contain useful multiscale metadata;
- eager-only materialization risks are explained before allocation;
- analysis-resolution and preview-resolution data remain clearly separate; and
- the accepted graph-reuse implementation remains protected by cross-platform
  checks for tunnel insertion, fragment paste, exact same-operation value paste,
  atomic undo, exclusion of runtime/cache/external dependency state, Linux Ctrl
  behavior, and macOS Cmd behavior.

### Feature Sequence C: Reproducible And Validated Analysis

Goal: turn implemented workflow families into defensible, shareable methods.

Release gate:

- automatic batch resume validates and extends prior checkpoints without
  overwriting completed work;
- one reproducibility package gathers the reviewed workflow/config/runner/
  environment/provenance inventory without silently embedding restricted data;
- an immutable compatibility corpus proves CPU-safe loading, non-destructive
  migration, and stable scientific hashes for unchanged semantics across the
  released workflow, batch, manifest, provenance, and generated-runner schemas;
- validation reports state expected values, tolerances, and limitations; and
- bundled examples can regenerate the reported tables and figures.

## Beta And 1.0 Readiness

Moving beyond alpha should depend on product evidence, not elapsed time or node
count. A beta candidate should have:

- stable documented schema migration with an immutable compatibility corpus;
- a named CPU/GPU/platform support matrix and clean installation evidence;
- several canonical workflows with publication-facing validation packs;
- predictable cancellation, checkpoint recovery, and automatic batch resume;
- explicit large-data materialization and preview behavior;
- a reproducibility-package export path; and
- novice-facing workflow health that remains transparent about every scientific
  assumption or automatic change.

## Later Milestones

These should wait until the source, scale, validation, and reproducibility base
is stronger:

- first-class points followed by puncta/spot detection and point measurements;
- first-class transforms followed by translation, drift correction, affine,
  and later non-rigid registration;
- first-class surfaces followed by mesh preview/export and specialist surface
  analysis;
- full plate/well/field browsing and broader HCS execution after the first
  source-item contract;
- model-backed segmentation such as Cellpose, StarDist, or ilastik through
  isolated optional dependencies and exact model provenance;
- Apple Metal acceleration through an MPS/MLX or other provider only after a
  time-boxed feasibility study; CPU remains the honest Apple fallback until a
  provider passes the same scientific and operational gates;
- stitching, mosaics, tracking, and specialist mitochondrial event metrics;
- AI-assisted graph authoring only after validated graph fragments, structured
  diffs, local approval, bounded context, and reproducibility provenance exist;
  and
- custom code nodes only with explicit trust, serialization, review, and
  sandboxing rules.

## Planning Rules

- Prefer a complete, documented, tested workflow over isolated nodes.
- Measure GPU value by end-to-end behavior, not raw kernel speed or badge count.
- Prefer metadata-preserving transformations over visually convenient
  shortcuts.
- Keep source identity, semantic axes, calibration, and acquisition metadata
  explicit.
- Prefer explicit output nodes and durable provenance for batch and publication
  workflows.
- Keep graph behavior serializable, migratable, and reproducible.
- Present one actionable problem at a time, apply only reviewed sane defaults,
  and explain every persisted automatic change.
- Treat validation, documentation, installation, recovery, and support evidence
  as part of a feature rather than cleanup after it.
