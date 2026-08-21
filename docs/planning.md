# napari-vipp Active Roadmap

Last reviewed: 2026-08-20

This document is the concise source of truth for active product priorities and
release order. Delivered chronology and old qualification detail are preserved
in the [0.13 planning archive](planning-history-0.13.md) and
[changelog](../CHANGELOG.md). Concepts that are worth retaining but are not
committed to a release belong in [product ideas](product-ideas.md).

The batch variability, volume-crop, and large-image work below incorporates
early tester feedback from Tom Naber on
[pull request 13](https://github.com/rensutheart/napari-vipp/pull/13).

## Product Direction

VIPP is a napari-native visual workflow builder for reproducible bioimage
analysis. The graph is the primary work surface: a user should be able to build,
inspect, tune, batch-run, export, and publish a workflow without losing the
relationship between pixels, source identity, semantic axes, calibration,
acquisition metadata, tables, and provenance.

The product is organized around complete scientific workflows rather than node
count or accelerator badges. A workflow should:

- retain the same scientific meaning across interactive, batch, generated
  Python/CLI, and export execution;
- run portably on CPU and use qualified accelerators only where their exact
  scientific and operational contracts pass;
- make source selection, analysis resolution, parameters, automatic decisions,
  fallbacks, and outputs reviewable; and
- produce enough structured evidence for another person to reproduce or audit
  the run.

Registration, model-backed segmentation, stitching, tracking, AI-assisted graph
authoring, and custom code remain possible future directions. They are recorded
in [product ideas](product-ideas.md) and do not displace the active source,
scale, interactivity, and reproducibility foundations below.

## Current Baseline

The current published alpha is `0.13.0a7`. It provides:

- workflow schema 4 and batch schema 3 with portable compute intent, explicit
  outputs, checkpoints, manifests, and exact implementation provenance;
- shared execution across interactive, batch, generated Python/CLI, and export;
- graph editing, manual/cached nodes, isolated tuning, cancellation, and memory
  controls;
- common image formats, local OME-Zarr, optional microscope readers, and an
  initial multi-series batch model;
- PSF/restoration, segmentation and cleanup, measurement, skeleton,
  colocalization, and spatial-association workflows; and
- a deliberately bounded set of reviewed CUDA implementations, with CPU as the
  portable scientific reference and fallback.

Important remaining limits are:

- interactive source execution still materializes complete selected images even
  when a reader exposes lazy arrays or resolution pyramids;
- one container can hold several scientific images, but selected-item identity
  is not yet durable across every interactive, batch, naming, and provenance
  surface;
- thumbnail presentation can publish provisional images, misstate cached
  backend provenance, and use an unreadably long status line;
- repeated Prefer-GPU parameter edits can pay avoidable setup, transfer,
  synchronization, and presentation costs; and
- validation and platform qualification remain uneven outside the exact
  published CPU and reviewed CUDA regions.

## Active Release Order

Small changes should land as independent, reviewable pull requests. A release
does not wait for every item in a cycle when a coherent, useful subset is ready.
Unchanged installer, GPU, schema, and UI evidence carries forward according to
the [qualification baseline](release-qualification-baseline.md).

### Remaining `0.13.0` Cycle: Interactive GPU And Correctness Stabilization

The remaining 0.13 alphas focus on thumbnail integrity, responsive parameter
tuning, truthful GPU/optimizer behavior, and contained correctness fixes. They
do not introduce SourceItem or broad lazy execution. Installer feature work is
deferred to `0.14.0a1`; a security or data-loss defect remains an exception.

Recommended dependency order:

1. concise thumbnail presentation and backend provenance;
2. atomic final-thumbnail publication;
3. interaction telemetry, device selection, and presentation scheduling;
4. bounded host-input float32 GPU thumbnail statistics, followed by measured
   resident-output reuse through a shared execution seam; and
5. evidence-justified cross-run scientific residency.

Optimizer fixes `#31` then `#32`, Rescale Axes `#33`, and the separate volume-
crop feature are non-blocking 0.13 workstreams that may proceed in parallel.
They do not gate the thumbnail/GPU sequence or every 0.13 alpha.

Items with disjoint ownership may proceed in parallel, but later work must use
the generation, provenance, and measurement contracts established earlier.

#### 0.13-A. Thumbnail Presentation Integrity ([#28](https://github.com/rensutheart/napari-vipp/issues/28), [#30](https://github.com/rensutheart/napari-vipp/issues/30))

Goal: a preview update must never show a provisional white/intermediate image or
fill the application status bar with technical provenance.

Implementation contract:

- keep the last complete thumbnail until a replacement image and its final
  contrast limits are both ready;
- use a per-node generation keyed by output identity and render settings, and
  publish exactly once only if that generation is still current;
- show an intentional loading state only when a node has no prior thumbnail;
- retain the old image on cancellation, failure, or stale completion;
- separate a short badge, a bounded one-sentence status summary, structured full
  details, and complete accessibility text;
- put only the summary in the one-line status bar and keep wrapped technical
  detail in an inspector/details surface; and
- identify this as **thumbnail contrast** computation, not the scientific node
  backend. Show Auto crossover information only when it actually selected the
  backend.

Acceptance:

- delayed, failed, cancelled, and out-of-order statistics jobs never replace a
  newer or previously valid thumbnail;
- first-load and cached-result states are visually and accessibly distinct;
- the longest ordinary summary remains readable in a narrow normal window; and
- CPU, Prefer GPU, scan-free masks, cached-policy mismatch, and actual fallback
  states have concise truthful labels.

Ordinary cross-platform Qt tests are sufficient; no real-GPU gate is required
unless accelerator behavior itself changes.

#### 0.13-B. Predictable GPU Tuning And Device Control ([#27](https://github.com/rensutheart/napari-vipp/issues/27))

Goal: repeated parameter edits should have explainable latency and use the
selected qualifying GPU consistently when that is beneficial.

Current evidence found the same underlying parameter-specialization pattern in
the originally reported Subtract Background Ball-size path and in Gaussian and
Median tuning. The historical Subtract Background provider compiled a new
CuPyX erosion specialization for each unseen footprint radius; Gaussian and
Median likewise specialized on radius or footprint size. The promoted
radius/size-independent kernels remove those repeated compile cliffs while
preserving their reviewed result contracts. The remaining evidence-led
residency decision is complete below; measured transfer time did not justify
cross-run scientific GPU residency.
The current development slice carries provider-neutral device observations
through detached pipeline results and adds explicit, bounded interaction
reports for standard scientific parameter controls. Those reports distinguish
invalidation, the actual debounce, worker queue and result-delivery delay,
scientific execution, thumbnail-statistics queue and execution, rendering,
final publication, and superseded generations. Pre-device observations now
separate graph restoration, cache preparation, workload preparation,
accelerator setup, runtime/library probing, compute planning, and device-plan
construction. They retain run-correlated runtime, device, implementation,
transfer, and synchronization spans without entering workflow or scientific
identity.

The final single-device RTX 5090 evidence matrix is now complete for the
standard scientific-control path. A per-workflow-session selector carries one
explicit qualified runtime/device through scientific and thumbnail requests
without entering workflow JSON. Device plans are fingerprint-bound to that
request, explicit devices must be reported by the provider, supplied
capability snapshots must describe the same device, and every completed GPU
observation now records terminal private-pool memory while the exact device
lease is still held. Specialized composite/source/presentation controls remain
outside the interaction reporter until their commit boundaries can be timed
exactly. Provider-internal compilation remains honestly included in operation
time when the provider exposes no separate boundary. A protected non-default
multi-GPU run remains conditional on access to a host with two qualified CUDA
devices; the single-GPU path fails closed rather than silently retargeting.

Ordered parameter-specialization follow-up:

1. **Completed:** Rolling-Ball Background and Subtract Background now share a
   radius-independent CuPy kernel whose compiled program does not change for
   each authored radius;
2. **Completed:** Median Filter now uses a size-independent CuPy radix kernel
   whose compiled program does not change for each supported odd kernel size;
3. **Completed:** a manifest-locked production sweep now varies and revisits
   every authored parameter or admitted branch across 14 directly executable
   GPU implementations, delegates RL and RL-TV to their multi-input sweep, and
   records the two fixed-profile measurement implementations explicitly. The
   2026-08-20 RTX 5090 run completed 122 production steps with the exact
   requested backends, full cleanup, no fallback, and no parameter-specific
   cliff signal;
4. **Completed:** the dedicated Richardson-Lucy and Richardson-Lucy TV sweep
   now varies and revisits PSF dimensions in 2D and 3D, changes iteration and
   TV-strength parameters, and pairs every GPU result with the authoritative
   CPU parity gate. The 2026-08-20 RTX 5090 run passed all four groups with no
   fallback, cleanup failure, parity failure, or avoidable PSF-shape stall; and
5. **Completed bounded host-input slice:** float32 Percentile and Min-max
   thumbnail statistics now use exact fixed-workspace CuPy radix reductions;
   the resident-output reuse slice remains in 0.13-C.

All five items have an implemented first slice in the current development tree.
Parameter-sweep timing remains machine-local diagnostic evidence rather than a
new per-release performance gate.

Implementation proceeds in measured phases:

1. **Implemented for standard scientific controls:** record invalidation,
   debounce, graph/cache/workload preparation, accelerator setup,
   runtime/library probing, compute planning, device-plan construction,
   scientific and presentation queueing, host-to-device transfer, compute,
   synchronization, device-to-host transfer, thumbnail work, rendering,
   publication, and discarded-generation time;
2. **Implemented:** a new scientific edit pre-empts queued presentation work,
   cancels obsolete work once, and publishes only the newest generation;
3. **Implemented, non-default multi-GPU hardware validation conditional:** enumerate
   qualifying devices, allow explicit per-workflow-session selection through
   the existing `device_id` contract, and display requested provider/device,
   actual per-node backend/device, and host/device boundaries; and
4. **Completed evidence decision:** do not add cross-run scientific residency;
   synchronized revisits measured only about 0.015-0.016 s of combined
   scientific transfer, while exact owned-revision source-context reuse removes
   the repeated preparation hash safely.

Fake clocks, two-device adapters, request fingerprints, and transfer counters
cover ordinary CI. The 2026-08-21 protected RTX 5090 matrix used an explicit
`cuda:0` selection and retained these machine-local artifacts:

- historical provider comparison:
  `D:\Temp\vipp-issue27-historical-radius-compile-baseline-20260821.json`
  (`SHA-256 49542130A12DC1C1567E6843A482BBC1AD404B6C6ED4332E23F30A9F1EEA9ED8`);
- exact bundled sample:
  `D:\Temp\vipp-issue27-ui-exact-owned-revision-final-20260821.json`
  (`SHA-256 6B5C3C3293CB4D592AA2386C9749A6C3284EFC8CD875C5DFC916184A43B5B21D`);
- 108 MiB bounded stack:
  `D:\Temp\vipp-issue27-ui-bounded-owned-revision-final-20260821.json`
  (`SHA-256 F20D81E824A08AD2C6FB45474C5AA2AEA8AA69B6C16CC0B23D877D2BDAAA1BE3`);
- 432 MiB resident-thumbnail stack:
  `D:\Temp\vipp-issue27-ui-resident-owned-revision-final-20260821.json`
  (`SHA-256 575C3E68433F5DFB8DC3BD35D53553965EC56A31A45E70EBAB13DA03E40BA206`);
  and
- separately synchronized 108 MiB diagnostic:
  `D:\Temp\vipp-issue27-ui-bounded-synchronized-owned-revision-final-20260821.json`
  (`SHA-256 F908BC86AE46B8CE7EAE8DB318601EB8DECB08EF39AC3C9F79ECD0DA58739B09`).

Every accepted GPU run selected `cuda-cupy/cuda:0`, retained exact CPU
dtype/shape/byte parity, used known-byte H2D and D2H with no fallback, and
reported zero terminal live/reserved private-pool bytes. Each fresh artifact
records all 137 production package Python files plus the harness, policy,
workflow, and project anchors (141 files total) under source-tree digest
`5cb9a53ffee8efa15f502d39df0e2fbbebf4eb3a3ccc2855b19d9fdd53cf6b9f`.
The evidence distinguishes explicit `cuda:0` affinity from real multi-device
validation, which was not performed on this one-GPU host. The large profiles
also superseded a signalled target-node execution, observed typed clean
cancellation, and immediately published a successful reuse run. Warm medians
were about 0.491 s for the small sample (CPU 0.387 s), 0.927 s for the 108 MiB
stack (CPU 5.773 s), and 1.728 s for the 432 MiB resident stack (CPU 11.390 s).
The resident path returned six exact thumbnail-statistics observations with
zero logical input upload. Synchronized 108 MiB evidence measured about
0.117-0.134 s of target device operation and 0.0151-0.0157 s of combined
scientific H2D and D2H on representative revisits, so cross-run scientific
residency is not justified by the measured transfer saving. Exact source
scientific-context reuse removes the repeated source-byte hash only for the
same accepted, read-only, VIPP-owned revision and cached output object. From
the second same-process warm edit onward, after the reusable source output was
accepted, later Prefer-GPU preparation measured roughly 0.02-0.04 s.
Any revision, source binding, metadata, state, mutability, or provenance
mismatch fails closed to a fresh exact hash. External Task Manager or
integrated-GPU activity remains compositor evidence, not a VIPP execution
claim.

The historical comparison uses the genuine pre-fix provider source from
commit `91a05a9`, the same bundled Subtract Background input, and separate
empty CuPy disk caches. Previously unseen Ball radii took a median 1.287 s in
the provider while revisits took 0.0155 s, reproducing an 83x intermittent
specialization cliff before any UI overhead. The current radius-independent
provider took 0.0133 s for both unseen values and revisits (96.8x faster for
unseen radii), with identical output hashes at every compared radius. This
provider-only baseline is paired with the complete current edit-to-publication
matrix above; it is not mislabeled as a historical UI trace that did not exist.

#### 0.13-C. Policy-Accurate Float32 Thumbnail Statistics ([#29](https://github.com/rensutheart/napari-vipp/issues/29))

Goal: common `float32` image outputs can use qualified GPU thumbnail statistics
under Prefer GPU without stale CPU decisions being presented as current.

Current status: the host-input and measured resident-output slices are
implemented and qualified. The ordinary worker continues to record its one
logical host-to-device upload. Under Prefer GPU, one warm, selected, retained
`float32` image output of at least 128 MiB can instead be scanned through a
synchronous non-retaining hook before its existing scientific device scope is
released. Small, cold, CPU, fallback, unretained, and non-selected outputs stay
on the established asynchronous and pre-emptible worker path.

Implementation contract:

- separate the current requested policy from the policy, runtime, provider,
  device, and algorithm that produced reusable cached limits;
- reuse exact numeric limits where safe, but label a policy-mismatched cached
  result as cached rather than as a fresh fallback;
- add bounded CuPy percentile and min/max paths for `float32`, matching the CPU
  contract for finite/non-finite values, channel axes, interpolation, empty and
  constant inputs, signed zero, and extreme values;
- add a synchronous, non-retaining execution hook that can consume a compatible
  resident GPU output without a device-to-host then host-to-device round trip;
- measure the host-only path explicitly when no resident allocation is
  available, rather than assuming a new upload is interactively beneficial;
- define workspace estimates, memory caps, cancellation boundaries, fallback,
  and zero-residue pool cleanup; and
- benchmark cold and warm behavior before changing Auto crossover policy.

Acceptance uses CPU golden cases and fake adapters in ordinary CI. Qualified
hardware must record parity, cold/warm timing, transfer bytes, cancellation, and
terminal cleanup for the startup workflow and a representative
Convert Dtype -> Subtract Background -> Gaussian -> Otsu workflow. The mode
matrix proves CPU labels fresh results as CPU rather than fallback, Prefer GPU
uses the qualified GPU path for the three compatible float outputs, and Otsu is
reported as scan-free. Thumbnail statistics must never change scientific arrays
or results.

The 2026-08-20 RTX 5090 host-input qualification passed all 26 protected
provider cases after cancellation cleanup, with exact CPU parity and zero
terminal private allocations. Fresh-process calibration at 2, 32, and 128 MiB
kept exact parity; cold calls were about 1.0-1.1 seconds while warm calls were
about 0.008, 0.024, and 0.079 seconds. The measured warm crossover was 32 MiB,
so the existing 512 MiB cold and 32 MiB warm Auto thresholds remain unchanged.
The exact 576 KiB Convert Dtype -> Subtract Background -> Gaussian -> Otsu
workflow then reported GPU float32 statistics for all three image outputs with
no fallback, while the boolean Otsu mask remained correctly scan-free.

The resident comparison then measured the same exact Percentile contract at 2,
32, 128, and 512 MiB. Avoiding the redundant upload saved about 0.6, 5.1, 24.8,
and 92.0 ms respectively (roughly 18-27% of the host-input provider body).
This supports a conservative 128 MiB resident threshold; it does not change
the 32 MiB warm Auto-selection threshold. The hook runs after the required
scientific device-to-host result transfer, returns only immutable host limits
and bounded transfer facts, and is excluded from scientific timing history.
Recoverable presentation failures remain soft misses, cancellation propagates,
and any scratch-cleanup failure is fatal and quarantines accelerator work. The
resident provider matrix proves zero logical input upload, bitwise CPU parity,
no borrowed alias or release, and zero terminal residue on every healthy exit.
These results complete the technical acceptance for #29 without claiming that
VIPP keeps scientific arrays resident between separate calculations.

#### 0.13-D. Optimizer Integrity ([#31](https://github.com/rensutheart/napari-vipp/issues/31), [#32](https://github.com/rensutheart/napari-vipp/issues/32))

Goal: Find fastest can optimize the scientific graph without executing output
writers and never treats CPU execution as proof for a requested GPU assignment.

Retained writer handling is complete for #31:

- requested retained IDs remain in optimizer identity and staleness checks;
- private execution and validation derive an effective scientific retention
  set containing only safe nodes;
- connected and disconnected Batch Output or Save Image nodes remain in the
  live workflow but stay outside the detached benchmark graph;
- during benchmark, cancellation, failure, or a no-change result, writers never
  execute or create artifacts, and the optimizer does not mutate the live
  graph, cached outputs, parameters, compute preferences, or provenance;
- an explicit accepted **Apply** action continues to change only the intended
  compute preferences and leaves scientific parameters and topology unchanged;
  and
- unknown-node validation occurs before writer filtering rather than silently
  filtering malformed requests.

Assignment integrity and review handling are implemented for #32:

- exact same-shape/same-dtype planning projections for Rescale Intensity and
  Unsharp Mask close the two host-operation gaps in the reported
  Gaussian/Otsu/Remove Small Objects workflow without executing either CPU
  kernel during planning;
- every private validation run attests requested, planned, device-segment, and
  actual implementation identity before any numerical comparison;
- a non-current, unlocked implementation that cannot execute as requested is
  excluded once and the remaining graph is solved again under the original
  deadline, allowing unrelated improvements to continue;
- current or locked assignment mismatches remain fatal, and strict provenance
  means a CPU result can never validate a requested GPU implementation;
- after exact backend attestation, a genuine small numerical difference can be
  offered for expert review: discrete outputs require at most 0.1% differing
  values, while floating outputs require both normalized RMSE and normalized
  maximum error at or below 0.1%; comparison metrics use bounded chunks for
  large or strided outputs; and
- review is never automatic. The dialog reports per-output counts and errors,
  requires an explicit acceptance checkbox, binds that acceptance to the exact
  proposal digest, and rechecks the normal workflow/input identity before
  Apply. Structural, dtype, shape, non-finite-class, and larger differences
  remain hard failures. Review does not promote a provider, relax its declared
  parity policy, or authorize CPU/GPU cache sharing.

User-facing failure text distinguishes assignment/planning disagreement from a
numerical parity failure and identifies the stage that diverged. Native
`uint16` Otsu must not receive an unnecessary dtype-conversion suggestion.

Tests cover connected and disconnected writers, every cache-retention mode,
absence of output files, cancellation/failure atomicity, fresh/cached/opaque
upstream states, staged assignment divergence, bounded re-solving, conservative
near-parity review, and the multi-node Gaussian/Otsu/morphology chain. Native
`uint16` Otsu receives no unnecessary dtype-conversion suggestion. `#32` also
has qualified real-GPU coverage for the reported Subtract Background → Rescale
Intensity → Convert float32 → Gaussian → Unsharp → Otsu → Remove Small Objects
corridor: the exact Gaussian/Otsu/Remove GPU identities were planned and
executed without fallback, Gaussian passed its production tolerance, both masks
were bitwise exact, and private GPU memory cleaned successfully.

#### 0.13-E. Named-Axis Correction ([#33](https://github.com/rensutheart/napari-vipp/issues/33))

Goal: common named Y/X inputs work safely in Rescale Axes without weakening the
axis contract for ambiguous data.

- accept carried, unambiguous named Y/X axes with inferred confidence under one
  visible warning;
- leave an unknown leading axis such as Q untouched;
- require an explicit mapping before nontrivial unresolved Z scaling;
- allow a true identity/no-op without an axis error; and
- do not promote unrelated inferred metadata merely because shape changed.

Tests cover QYX, ZYX, TCZYX, reordered axes, images/masks/labels, metadata
confidence/history, workflow restore, export, and batch. This contained bug fix
may ship independently and does not wait for the crop feature below.

#### 0.13-F. Responsive Volume Crop (new issue required)

Goal: make Crop Stack a discoverable volume ROI tool without recalculating on
every drag event. Track this as a separate feature issue before implementation;
it is not part of `#33`.

- preserve the existing operation and old workflows while adding persisted
  nonnegative `z_start` and `z_end` crop margins, measured as samples removed
  from the leading and trailing ends of an explicitly named Z axis;
- default both new parameters to zero when loading an old workflow;
- show the controls only when metadata contains explicit Z or the user has
  authored an exact axis mapping; inferred QYX must never treat Q as Z;
- preserve T/C axes, leave at least one sample on every cropped axis, and update
  physical origins, history, and output-size summaries;
- while dragging, draw an immediate translucent ROI outline/mask over the
  cached display rather than constructing a full zero-filled image;
- keep draft control values during interaction, then commit one undoable
  scientific edit on release or after roughly 300-500 ms idle; and
- flush a pending draft before Calculate, Calculate all, any Run or execution
  snapshot, save, export, batch start, tab change, or close.

Tests cover ZYX, TCZYX, noncanonical explicit layouts, rejection of inferred QYX,
images/masks/labels, physical origins, save/reopen/export/batch, rapid drag
events, one committed calculation, and stale/cancelled completion. This work can
land in 0.13 if it remains contained; it is not a blocker for every 0.13 alpha.

#### 0.13 Release Boundary

- Run focused ordinary CI for presentation, optimizer-writer, and axis/crop
  changes.
- Run protected real-GPU evidence only when GPU selection, implementations,
  transfers, residency, or strict assignment validation changes.
- Do not repeat installer lifecycle qualification when installer code and its
  dependencies are unchanged.
- Run the complete cross-platform suite once on the integrated release
  candidate rather than after every small documentation or UI pull request.
- Describe measured performance as machine-local unless more than one reviewed
  hardware class supports a broader claim.

### `0.14.0a1`: Source-Aware Loading And Per-Sample Batch Alpha

The first 0.14 alpha targets three connected source-aware outcomes and one
parallel installer-usability outcome:

1. durable SourceItem identity;
2. pyramid-aware source loading and presentation preview;
3. typed per-sample batch parameters; and
4. clearer installer/cuCIM environment discovery.

The source-aware outcomes form one coherent user story: select a scientific item
once, see a useful preview before the exact full-resolution snapshot is ready,
and apply reviewed item-specific batch parameters. The independent installer
lane makes the managed environment discoverable when an optional component is
installed. The a1 preview improves time to first display; exact background
verification and level-0 materialization still complete before scientific
execution.

Implementation order and pull-request boundaries:

1. define SourceItem v1, canonical identity, reader adapters, legacy migration,
   and compatibility goldens;
2. propagate the record through interactive persistence, batch planning/naming,
   manifests, generated execution, replay, export, and provenance;
3. add the OME-Zarr early presentation-preview worker on the stable identity and
   generation contracts established by SourceItem and 0.13-A;
4. add the typed batch-override schema, resolver, execution/provenance path, and
   workspace table on the propagated identity;
5. implement installer/cuCIM discovery independently in parallel; and
6. finish with one integration example containing multiple source items, two
   effective thresholds, and a pyramid preview whose level differs visibly from
   the unchanged full-resolution analysis level.

#### 0.14-A. Immutable SourceItem Identity

Goal: distinguish the source container from the scientific item selected inside
it, and carry that same identity through every execution surface.

The first supported item kinds are ordinary images, series/scenes, NPZ members,
and OME-Zarr image or label groups. Plate/well/field and semantic-axis iteration
remain later extensions. Level 0 is the explicit scientific analysis selection
for a1; a presentation pyramid level is recorded separately and never changes
the SourceItem's scientific selection.

Implementation contract:

- define a frozen, serializable logical selector and resolved identity with
  separate container URI/format, item key/index/name/kind, and observed content-
  revision proof; changing bytes invalidates the proof without silently
  retargeting the authored logical selector;
- record deterministic reader key, shape, dtype, axes, scale, and available
  level/capability metadata without materializing pixels during inspection;
- resolve saved selection by stable key with a documented legacy-index fallback;
  missing, duplicate, or mismatched items fail visibly rather than silently
  selecting a replacement;
- use the same canonical item document in interactive Image Source,
  `SourcePayload`, batch rows, names, collision handling, manifests,
  checkpoints, generated runners, replay, export, and provenance;
- keep public identities privacy-safe and keep container revision separate from
  item selection; and
- migrate existing workflow and batch documents without inventing scene, group,
  axis, or calibration meaning. Preserve compatibility properties during the
  transition.

SourceItem owns schema v1. The expected integration bumps are workflow schema 5,
batch config 4, and manifest/item record 4. Continue reading immutable workflow
v4, batch v1-v3, and manifest v3 fixtures; migrate legacy `series_index`
deterministically, reject unknown future versions, and write only the new
canonical representation.

Acceptance:

- two items in one container always receive different stable identities;
- save/reopen selects the same item when inspection order changes;
- source revision invalidates stale identity and cache proof;
- old schemas remain CPU-safe, save/reopen/re-save deterministically, and retain
  scientific hash behavior where meaning is unchanged;
- TIFF/NPZ/OME-Zarr multi-item fixtures agree across inspector, interactive,
  batch, generated execution, filenames, manifests, and provenance; and
- inspection remains pixel-lazy, optional readers remain optional imports, and
  public provenance contains no private absolute path.

#### 0.14-B. Pyramid-Aware Loading And Preview

Goal: show a useful large-image preview quickly while keeping scientific
analysis resolution explicit and unchanged.

Implementation contract:

- attach reader capabilities to SourceItem inspection, including declared
  levels, scale transforms, chunk shape, preview-level reads, exact region
  reads, and lazy support;
- model presentation preview separately from the canonical analysis payload;
- start with local OME-Zarr and select the coarsest declared level that still
  satisfies the requested display detail, slicing required T/Z/C and Y/X chunks
  before compute;
- label the result `Preview level N - analysis remains full resolution` and
  never place a lossy preview into the graph, scientific cache, full-stack
  statistics, or output provenance as analysis data;
- generation-key and cancel preview/refinement work, retain the previous valid
  preview until replacement, and close tasks/handles on source change, tab
  close, or cancellation;
- give exact verification and level-0 materialization the same generation and
  cancellation ownership, so a superseded source stops I/O, releases handles
  and buffers, and cannot continue consuming full-source RAM in the background;
- permit a display-only provisional preview while exact container verification
  completes, but publish the scientific source only after identity/revision
  checks pass; and
- add IMS pyramid support only after its adapter proves level enumeration,
  transforms, and cheap reads on controlled real files.

Acceptance measures time to first preview, bytes read, peak RAM, cancellation,
stale-source rejection, cache bounds, and analysis-result parity. Preview level
and analysis level must remain visibly and structurally distinct. Record initial
preview I/O separately from the still-required complete verification and level-0
load. Fixtures prove that a lower OME-Zarr level is read without computing level
0 for the preview, its coordinate transform is applied correctly, a label-group
preview keeps label semantics without intensity-statistics leakage, and a
single-level or unsupported reader falls back without making a false pyramid or
region-read claim.

Direct full-resolution Crop ROI pushdown is the committed follow-up specified
below, not hidden a1 scope.

#### 0.14-C. Per-Sample Batch Parameters

Goal: let a saved batch use reviewed parameter values for individual SourceItems
without mutating the base workflow or encoding scientific decisions in filename
expressions.

First coherent slice:

- rows use a stable composite batch-item key made from each varying source node
  ID and its logical SourceItem selector, never the revision-bound proof,
  ordinal batch ID, filename alone, or series display name; every observed
  source revision is bound separately to the plan and item evidence;
- columns are explicitly selected authored scientific scalar parameters and a
  blank cell inherits the saved workflow value. Exclude Image Source paths and
  item selectors, Save/Batch Output destinations, `_vipp_` or derived fields,
  compute/manual/cache controls, and topology;
- store exact typed values and validate them through the target node's normal
  workflow/scientific parameter contract before any item runs; UI range hints
  do not silently become a stricter second execution contract;
- begin with numeric scalar parameters needed for cases such as per-sample
  manual thresholds; exclude Python expressions, substring rules, topology
  changes, and arbitrary code;
- reject missing nodes/parameters, duplicate override rows, and an authored row
  that resolves to zero or multiple planned items. A newly discovered item with
  no override row visibly inherits workflow defaults; removed, renamed, or
  re-paired items leave a stale authored row and stop preflight;
- resolve each item against a detached effective workflow without mutating the
  live or saved base workflow;
- use the same resolver in representative preview, full batch, saved batch
  runner, and its CLI execution; and
- store the normalized authored override table and config hash in the batch
  config. During planning derive each effective workflow digest and record it
  with requested/effective overrides and observed source revisions in the plan,
  manifest, checkpoint, and item provenance. Those facts become the refusal
  boundary when verified automatic resume is implemented later.

Acceptance covers standalone files, multiple series in one container, and rows
with multiple varying source bindings; reorder, added-item default inheritance,
removed/renamed stale rows, and changed multi-source pairing; distinct
thresholds producing distinct expected outputs; preview/batch/saved-runner/CLI
parity; config and effective hash changes; checkpoint evidence; and unchanged
base-workflow state. CSV import/export may follow after the typed core and UI are
stable.

#### 0.14-D. Installer And Optional cuCIM Discovery ([#26](https://github.com/rensutheart/napari-vipp/issues/26))

Goal: a user installing the optional cuCIM component should not have to discover
where VIPP placed its managed CUDA Python.

Implementation contract:

- retain explicit target, environment override, and active-environment
  precedence;
- otherwise perform read-only standalone discovery from the canonical Windows
  LocalAppData managed CUDA ownership record before Python is known;
- before executing any discovered interpreter, resolve and re-check strict
  containment and require every component from the canonical managed root
  through the ownership record, active environment, `Scripts`, and
  `python.exe` to be non-reparse/non-link, with the interpreter a regular file;
- validate ownership schema/version, track exactly `cuda13`, canonical managed
  root, and active environment as a strict managed-store descendant;
- visibly show the discovered path with Continue/Cancel before invoking the
  existing Python backend, which remains the authoritative environment and
  package validation boundary;
- on failure, explain the reason and open the picker at the best safe managed
  location rather than the user profile; and
- on CUDA13 success, show the active environment location on the main installer
  success screen and state that the optional cuCIM add-on will find it
  automatically. CPU success must not make that claim.

Ordinary Windows CI covers valid current/repaired ownership, retired roots,
CPU/foreign/malformed/oversized/outside-store/missing-Python records, junction or
symlink redirection at the environment, `Scripts`, or interpreter boundary,
override precedence, bundle contents, CUDA-versus-CPU success text, and zero
backend invocation on discovery failure. This feature does not require real-GPU
or full installer lifecycle qualification unless the authoritative installer
transaction changes.

#### 0.14.0a1 Release Boundary And Non-Goals

The source-aware core of the alpha is complete when SourceItem identity,
pyramid-aware presentation, and per-sample scalar parameters agree across their
documented surfaces and compatibility gates. Installer and cuCIM discovery is a
planned parallel `0.14.0a1` work package, but it does not gate that source-aware
core. Every work package must pass its own acceptance before it ships; an
unfinished independent package moves visibly to the next alpha rather than
holding a coherent release. The milestone does not imply completion of:

- arbitrary conditional branches or filename expressions;
- full operation-level lazy/chunked execution;
- remote stores, plate/well/field traversal, or semantic-axis batch iteration;
- scientific progressive outputs;
- arbitrary per-node or per-item topology changes; or
- general branch-aware ROI-union planning.

Those items require later evidence or remain in [product ideas](product-ideas.md).

### Committed Follow-Ups After `0.14.0a1`

#### Safe Node Bypass And Batch Execution Profiles

Goal: compare workflows and omit preview-only preparation without deleting or
rewiring nodes or hiding a scientific change.

- Add authored **Run / Bypass** only to explicitly reviewed pure unary,
  single-output, contract-preserving operations. Start with Crop Stack; review
  Gaussian Blur separately with effective image-kind and downstream-port
  compatibility. Do not infer safety from a generic type-preservation flag.
- Exclude sources, Save/Batch Output, multi-input/output operations,
  image-to-mask type changes, tables, conditionals, and side-effect boundaries
  unless reviewed separately.
- Persist and scientifically hash bypass state; make it undoable and invalidate
  the node and descendants when changed.
- Alias input data, metadata, and device residency to output without calling the
  operation or forcing a host transfer. Record `bypassed` provenance and exclude
  the node from optimizer timing.
- Require identical interactive, batch, generated Python/CLI, export, CPU, and
  supported-GPU behavior.

After that contract is stable, add an explicit batch execution-profile override
with **Use workflow / Run / Bypass**. A whole-batch override comes first;
per-item bypass may later reuse the typed per-sample resolver. Batch-only behavior
must never be an invisible intrinsic node setting.

#### Exact Source-Window Pushdown

Goal: avoid decoding pixels that a direct source Crop will discard while keeping
the authored graph and result exactly equivalent to eager execution.

- Replace `full read -> direct Crop Stack` with an exact full-resolution region
  read only when the reader advertises the capability, Crop is the source's sole
  direct consumer, semantic spatial axes and parameters are fixed, and no branch
  or tunnel needs the complete source.
- Preserve the Crop node, history, physical origin, cache identity, scientific
  hash, and provenance. Verify eager-versus-region bytes and ImageState parity.
- Include source revision, SourceItem, ROI, reader implementation/version, and
  axis declaration in the read identity. Fall back to the complete read when any
  proof is absent or stale.
- Start with local OME-Zarr. Require real-file evidence before claiming exact
  region reads for IMS, BioIO, TIFF, or another adapter.

This depends on SourceItem, reader capability metadata, and the named-axis
contract from 0.13-E. Branch-aware ROI unions remain an uncommitted idea.

#### Selective Lazy And Chunked Scientific Execution

- Declare each operation as combinations of view-only, lazy-safe, blockwise,
  halo/boundary-dependent, global-reduction, eager-only, memory-heavy, and
  scale-aware.
- Start with exact slice/channel/crop/reorder views and simple pointwise
  operations. Filters require halo parity; thresholds may require a global pass;
  connected components, watershed, and similar operations remain global until
  an exact chunk-boundary contract exists.
- Explain estimated materialization before an eager-only node allocates a very
  large source.
- Reuse batch cancellation/checkpoint semantics for source and chunk workers,
  and require deterministic fingerprints, bounded caches, atomic writes, and
  truthful CPU/GPU transfer accounting.

#### Reproducibility, Resume, And Validation Packs

- Export a reviewed reproducibility package containing workflow, batch config,
  generated runner, environment, source-item summaries, manifests, checkpoints,
  implementation provenance, output digests, and explicit omissions without
  silently embedding restricted raw data.
- Turn checkpoints into verified automatic resume: validate schemas, workflow,
  effective per-item intent, source identities, destinations, and prior
  checkpoint integrity before preserving completed work.
- Maintain compatibility goldens for released workflow, batch, manifest,
  provenance, and generated-runner schemas.
- Continue publication-grade validation packs for watershed, colocalization,
  object association, skeleton/network topology, real PSFs/deconvolution,
  microscope metadata, interrupted batch, and OME-Zarr round-tripping.

## Continuous Product And Release Gates

These gates apply proportionately to every change:

- **Scientific meaning:** exact inputs, parameters, axes, calibration, output
  type, metadata, and tolerances are declared. A convenience or performance
  feature cannot silently change analysis resolution or population.
- **Compatibility:** persisted behavior has migration and immutable old-schema
  coverage. A migration must not invent source items, axes, calibration, or
  parameters.
- **Execution truth:** planned and actual CPU/GPU decisions, fallback, transfer,
  cancellation, cleanup, and provenance agree across interactive, batch,
  generated, and export routes.
- **Responsive publication:** asynchronous work is generation-owned; stale,
  cancelled, provisional, or failed work cannot replace a newer valid result.
- **Privacy and recovery:** identities and support artifacts avoid private
  absolute paths; writes are bounded, atomic where practical, and recoverable.
- **Impact-based qualification:** focused tests run for every change; protected
  GPU, installer lifecycle, schema migration, or extensive UI evidence reruns
  only when that domain changes. A complete cross-platform suite runs once on a
  release candidate.

Accelerator convenience remains conservative. VIPP may offer **Add conversion**
only for the reviewed `uint8`/`uint16` to `float32` conversion with
`Scaling = Preserve`. The visible result says **GPU eligible**, not that GPU use
is a guarantee. VIPP must never silently insert casts, reorder pixels, rewrite
scientific parameters, or manufacture axis/calibration meaning to enter an
accelerator region.

## Beta And 1.0 Readiness

A beta candidate should have:

- stable SourceItem, workflow, batch, manifest, and provenance migration;
- a named CPU/GPU/platform support matrix and clean installation evidence;
- canonical workflows with publication-facing validation packs;
- predictable cancellation, checkpoint recovery, and automatic batch resume;
- explicit large-data preview, analysis-resolution, and materialization
  behavior;
- reproducibility-package export; and
- novice-facing workflow health that remains transparent about every scientific
  assumption or persisted automatic change.

## Governing References

- [architecture.md](architecture.md): runtime, data model, persistence, and
  known seams.
- [user-guide.md](user-guide.md): current end-user behavior.
- [io-user-guide.md](io-user-guide.md), [ome-io-plan.md](ome-io-plan.md), and
  [cache-and-memory.md](cache-and-memory.md): current source, I/O, and
  eager/lazy boundaries.
- [gpu-production-implementation-plan.md](gpu-production-implementation-plan.md)
  and [durable-gpu-execution.md](durable-gpu-execution.md): detailed accelerator
  admission and cross-surface execution contracts.
- [desktop-startup-and-installer-plan.md](desktop-startup-and-installer-plan.md):
  installer and launcher architecture.
- [node-roadmap.md](node-roadmap.md): implemented operations and candidate node
  families.
- [research-and-publication.md](research-and-publication.md) and validation
  reports: evidence boundaries and publication-facing results.

## Planning Rules

- Prefer a complete, documented, tested workflow over isolated nodes.
- Measure GPU value by end-to-end behavior, not raw kernel speed or badge count.
- Keep source identity, semantic axes, calibration, analysis resolution, and
  acquisition metadata explicit.
- Keep graph and batch behavior serializable, migratable, and reproducible.
- Present one actionable problem at a time and explain every persisted automatic
  change.
- Put completed engineering detail in release notes or implementation records,
  not back into this active roadmap.
- Put uncommitted possibilities in [product ideas](product-ideas.md) until they
  have a user outcome, owner, dependency boundary, and acceptance contract.
