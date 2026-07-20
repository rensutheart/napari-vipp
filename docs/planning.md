# napari-vipp Planning And Roadmap

Last reviewed: 2026-07-20

This is the concise planning source of truth. It records the current public
baseline, the work that is still genuinely open, and the intended order for the
next alpha releases. Detailed implementation notes stay in the specialist docs
listed below.

## Product Direction

VIPP is a napari-native visual workflow builder for bioimage analysis. The graph
canvas is the primary work surface: users should be able to build, inspect,
reuse, export, batch-run, and publish workflows without losing the connection
between image data, metadata, tables, and provenance.

The core workflow families are:

- segmentation and label cleanup for 2D and true 3D data;
- object, intensity, mesh, and skeleton measurements;
- two-channel pixel and object colocalization;
- multi-channel, z-stack, and time-lapse fluorescence data;
- native PSF generation and PSF-aware restoration/deconvolution;
- microscope acquisition import across common vendor formats, with normalized
  axes, channel, objective, and scale metadata;
- reproducible batch execution with explicit outputs.

PSF generation, deconvolution foundations, and optional microscope-reader
routing are part of the 0.11 baseline; explicit batch configuration and
provenance are part of 0.12. Near-term work is validation on real data,
scalable OME-Zarr previews, and safe graph/parameter copy and paste.
Registration, model-backed segmentation, stitching, and AI-assisted graph
authoring remain later milestones.

## Reference Documents

- [architecture.md](architecture.md): implementation architecture and data model.
- [user-guide.md](user-guide.md): current end-user workflow behavior.
- [node-roadmap.md](node-roadmap.md): detailed node inventory and candidate
  nodes.
- [io-user-guide.md](io-user-guide.md) and [ome-io-plan.md](ome-io-plan.md):
  supported I/O and OME architecture.
- [cache-and-memory.md](cache-and-memory.md): cache modes, memory guard, and
  operation memory policy.
- [psf-and-deconvolution-plan.md](psf-and-deconvolution-plan.md): PSF
  generation, deconvolution, and microscope metadata requirements.
- [measurement-workflows.md](measurement-workflows.md),
  [skeleton-nodes.md](skeleton-nodes.md), and
  [object-mesh-morphology-plan.md](object-mesh-morphology-plan.md):
  measurement workflow guidance.
- [analytical-phantom-validation.md](analytical-phantom-validation.md),
  [colocalization-method-notes.md](colocalization-method-notes.md), and
  [research-and-publication.md](research-and-publication.md): validation and
  publication-facing evidence.
- [mitomorph-feature-parity.md](mitomorph-feature-parity.md): MitoMorph-inspired
  feature parity tracking.
- Future GPU backend research is kept off main on the
  [`codex/gpu-cross-platform-support`](https://github.com/rensutheart/napari-vipp/tree/codex/gpu-cross-platform-support)
  branch. Main remains the CPU production baseline until that work passes its
  scientific-parity, packaging, memory, and cross-platform promotion gates.

## Current Public Baseline

Current alpha release: `0.12.0a3`.

The 0.12 alpha adds deterministic batch configuration/provenance, explicit
scientific source/grid/axis contracts, shared-executor Python export, workflow
schema version 3, and modular core/UI boundaries on top of the 0.11 exact
large-image, PSF/restoration, and microscope-import foundation.
Implemented and documented work includes:

- searchable categorized palette and searchable graph canvas;
- pan/zoom graph with typed ports, cycle rejection, undo/redo, duplicate/delete,
  contextual graph menus, auto-structure, and insert-on-wire make-room behavior;
- named port tunnels, tunnel reveal/highlight, tunnel manager, and saved graph
  notes;
- ambiguous insert-on-wire port mapping, dynamic multi-output port handling,
  `Split Channels`, and explicit `Split Axis`;
- workflow JSON with canvas positions, named tunnels, graph notes, selected
  inspector state, optional per-node thumbnail visibility, strict loading, and
  compatibility when optional VIPP UI metadata is absent;
- Python export, retained collection batch workspace, explicit `Batch Output`
  nodes, deterministic plan review, representative navigation, multi-source
  bindings, and saved workflow/config/manifest artifacts;
- background execution, stale-result rejection, cooperative cancellation where
  supported, manual/cached measurement nodes, branch-local dirty reruns, cache
  modes, auto memory guard, and per-node `Keep output cached`;
- OME-NGFF-inspired image/table metadata, semantic axes, scale/units/origin,
  channel metadata, source history, OME-TIFF/ImageJ TIFF/common raster/NumPy
  I/O, local OME-Zarr 0.4/0.5 read/write, and OME-Zarr image plus label
  analysis packages;
- object, intensity, derived morphology, 3D mesh, skeleton/network,
  colocalization, object association, nearest-object distance, event
  localization, table merge, table annotation, and grouped summary workflows;
- example workflows for label cleanup, measurement, derived morphology, 3D mesh
  morphology, skeleton/network QC, pixel/object colocalization, named tunnels,
  graph notes, selected-inspector metadata, and 2D/3D PSF-aware deconvolution;
- Born-Wolf PSF generation, PSF preparation, baseline Richardson-Lucy
  deconvolution, Richardson-Lucy TV deconvolution, channel-specific PSF
  outputs, and measured-PSF workflow guidance;
- optional microscope-reader routing plus normalized acquisition metadata fields
  used by PSF generation and provenance checks;
- slice/stack thumbnail contrast range controls, cached stack contrast limits,
  and a linked/unlinked napari/VIPP slider setting for large data review;
- automated tests plus calibrated analytical morphology phantom validation.

Known constraints:

- progress/cancellation coverage depends on third-party operations and remains
  uneven;
- most processing remains eager, so very large OME-Zarr workflows still need a
  more deliberate lazy and pyramid-aware preview strategy;
- broad proprietary microscope import is active development rather than a
  public baseline guarantee; reader support should be documented per format as
  supported, experimental, or metadata-incomplete;
- validation is strong for calibrated morphology but still uneven for
  colocalization, watershed, skeleton/network, batch/provenance, and OME-Zarr
  round-tripping.

## Active TODOs After 0.12

These are the items that should guide near-term work. Items not listed here are
either already implemented enough to build on or intentionally deferred.

### 1. PSF Generation And Deconvolution

Already implemented for the first 0.11 alpha: normalized objective/channel metadata from
OME-TIFF where available, a native `Born-Wolf PSF` node that can use connected
image metadata, `Prepare / Validate PSF`, baseline Richardson-Lucy
deconvolution, Richardson-Lucy TV deconvolution, deterministic synthetic
2D/3D deconvolution samples, example workflows, and a dedicated
PSF/deconvolution plan.

Still needed before positioning restoration as publication-ready:

- real bead-PSF and microscopy-image validation datasets for at least one 2D
  and one 3D workflow;
- boundary-policy follow-up for reflect-padded edge handling and explicit
  crop-margin guidance;
- release-facing tutorial screenshots or walkthroughs for measured PSF,
  generated PSF, baseline RL, and RL-TV comparison workflows;
- performance profiling on larger 3D volumes before considering chunking,
  vector acceleration, or optional GPU work;
- continued documentation of wavelength, numerical aperture, refractive index,
  pixel size, z step, channel selection, and when metadata is being used versus
  manually overridden.

### 2. Microscope File Import Expansion

Already implemented: shared headless I/O registry, OME-TIFF/ImageJ TIFF/
conventional TIFF/common raster/NumPy/OME-Zarr import and export, normalized
image/source metadata, adaptive TIFF series selection, OME-TIFF objective
metadata preservation where OME-XML exposes it, and the first optional
microscope reader boundary for ND2, CZI, LSM, Leica, Olympus, and BioIO/
Bio-Formats-style fallback routes.

Still needed:

- real sample-file validation for Nikon ND2, Zeiss CZI/LSM, Leica LIF/LOF/XLIF,
  and Olympus OIR/OIB/OIF/VSI files across facilities and acquisition modes;
- richer native metadata extraction for objective, detector, plate/well/field,
  scene/position, and acquisition-loop details beyond the first normalized
  fields;
- fallback-reader documentation that explains which optional extras are native,
  BioIO-backed, or Bio-Formats-backed;
- normalized metadata mapping for every reader: axes, scale/units, channel
  names/colours, excitation/emission wavelengths, objective NA/magnification,
  immersion/refractive index, series/scene identity, plate/well/field where
  present, and raw metadata provenance;
- source-inspection UI that can select series/scenes/positions without turning
  ordinary graph execution into a hidden list-of-images operation;
- small public or synthetic fixtures for reader dispatch and metadata
  normalization, with larger proprietary sample files kept out of the repository
  when licensing or size requires it.

### 3. Batch Configuration And Provenance

Already implemented: local collection batch execution, explicit `Batch Output`
nodes, sorted multi-source binding, shared-planner plan review with one
calculated graph representative, low-memory batch retention,
workflow/config/manifest reproducibility artifacts, versioned
`vipp_batch_config.json`, existing-file policies, latest/archive manifests with
atomic per-item checkpoints, and default-on configurable continuation after
item failures.

A bundled deterministic batch/provenance demo now supplies the release-gate
fixture: three sorted-position pairs, mixed image/label/table outputs, exact
ground truth, a portable saved config and runner, and checks for hashes,
manifests, archives, sidecars, replay policies, and isolated corrupt inputs.

The 0.12 implementation records workflow/config hashes, software versions,
input identity and source metadata, resolved output paths, item/output status,
and completed/partial/skipped/failed summary counts. `Batch Output` nodes remain
the authoritative selected outputs; single-output terminal nodes remain only
as a warned compatibility fallback.

Deferred beyond this milestone:

- semantic-axis iteration for timepoints, channels, z-slices, or selected
  combinations;
- first-pass plate/well/field collection traversal for HCS-style layouts.

### 4. OME-Zarr Scale And Preview Strategy

Already implemented: local OME-Zarr 0.4/0.5 image read/write, lazy reads,
OME-Zarr image plus label analysis packages, cache modes, and operation memory
documentation.

Still needed:

- generated OME-Zarr pyramids for exported image datasets;
- label colors and label-property table round-tripping where practical;
- preview-resolution controls for thumbnails and inspector views;
- lazy/chunked all-pixel histograms and pyramid-aware thumbnails for large
  arrays, without changing operational results through hidden sampling;
- operation capability declarations such as eager, lazy-safe, memory-heavy, and
  scale-aware;
- warnings before eager-only nodes materialize very large lazy arrays;
- anonymous HTTP read investigation for public OME-Zarr datasets.

### 5. Scientific Validation Pack

Already implemented: automated tests, calibrated analytical morphology phantom
validation, and publication-facing colocalization method notes.

Still needed:

- colocalization validation report using deterministic threshold and overlap
  scenarios;
- object association validation report for overlap, nearest distance, and event
  localization assumptions;
- watershed/touching-object validation on geometric and microscopy-like
  phantoms;
- skeleton/network validation report with known endpoints, junctions, cycles,
  branch lengths, and anisotropic spacing;
- reproducible example-output artifacts for methods figures and supplementary
  material;
- RACC numerical-core decision: keep VIPP-owned implementation, share a common
  core with the RACC plugin, or document the intentional separation.

### 6. Graph Copy, Paste, And Parameter Transfer

Add copy/paste as a first-class graph-authoring operation in the next release.
This should cover both copying graph structure and transferring settings between
two instances of the same operation.

Interaction contract:

- Ctrl-click on Windows/Linux and Cmd-click on macOS toggles nodes into or out
  of a multi-selection. Clicking a node without the platform modifier returns
  to single selection; clicking empty canvas clears the selection. The
  inspector continues to show the most recently focused node within the
  selection rather than attempting to merge several parameter forms.
- `Ctrl+C`/`Cmd+C` and a node context-menu `Copy` action copy every selected
  node. Right-clicking an unselected node first makes that node the copy target;
  right-clicking a node already in a multi-selection preserves the selection.
- `Ctrl+V`/`Cmd+V` pastes onto the canvas near the mouse position, or near the
  viewport centre when the pointer is outside the canvas. An empty-canvas
  context menu exposes `Paste here` at the clicked graph position.
- Pasting creates new node ids, preserves operation ids, current serialized
  parameters, relative positions, and connections whose two endpoints are both
  in the copied selection. Connections to nodes outside the selection, named
  tunnel subscriptions/definitions, cached results, runtime/error state, pin
  state, and transient inspector state are not copied in the first iteration.
  This keeps the pasted group self-contained and prevents hidden dependencies
  or large data copies.
- Repeated paste offsets the group slightly so every result remains visible.
  The newly pasted nodes become the active selection, retain their relative
  layout, and the whole paste is one undo/redo action. If any node or internal
  connection cannot be validated, the paste fails atomically with a clear
  status message.
- `Paste parameters` appears when the clipboard contains exactly one copied
  node and the user right-clicks another node with the same operation id. It
  replaces all serialized inspector parameters on the target, using the normal
  parameter validation, dirty/stale propagation, dynamic-port refresh, and
  recalculation rules. It does not change the target's id, position,
  connections, tunnels, note, pin/cache state, or output data. The parameter
  replacement is one undo/redo action.
- `Paste parameters` is hidden or disabled with an explanatory reason for a
  different operation type, malformed/outdated data, or a multi-node clipboard.
  Compatibility is exact by operation id for this release; superficially
  similar nodes must not receive best-effort parameter mappings.
- Clipboard data should use a versioned VIPP MIME payload backed by the same
  validated node/connection serialization concepts as workflow JSON. A private
  in-process fallback may support environments where the system clipboard is
  unavailable, but plain text or arbitrary clipboard content must never be
  interpreted as a graph fragment.

Release acceptance:

- keyboard and context-menu paths behave consistently on macOS, Windows, and
  Linux, including focus in parameter editors so ordinary text copy/paste is
  not intercepted by the graph;
- tests cover selection toggling, single- and multi-node copying, preservation
  of internal wiring and relative layout, exclusion of external wiring and
  runtime state, repeated paste placement, atomic validation failure, and
  undo/redo;
- parameter-paste tests cover exact-operation compatibility, dynamic parameters
  and ports, invalid or older payloads, stale-result propagation, and rejection
  across operation types;
- the user guide and release notes document the shortcuts, both context menus,
  what is and is not copied, and the distinction between `Paste here` and
  `Paste parameters`.

### 7. Tune A Node In Isolation (Implemented For Next Release)

The unreleased implementation adds a temporary interactive tuning mode for
cases where one node is quick to
calculate but its downstream branch is expensive. The user should be able to
adjust and recalculate the selected node repeatedly, inspect that node's latest
output, and defer all downstream recalculation until the chosen parameters are
ready.

The action is exposed as `Tune node in isolation`, with a
visible `Downstream paused` state. Avoid calling this only "freeze" or "focus",
because those terms can be confused with cached outputs, pinned nodes, or
ordinary inspector selection.

Implemented interaction contract:

- activating isolation on a node continues to use its current upstream inputs
  and allows that node itself to recalculate normally, but parameter changes do
  not schedule any downstream node;
- every isolated recalculation replaces the node's local preview/output with
  the newest result, while descendants are visibly marked as darker-amber
  blocked/waiting and held so an old downstream result cannot be mistaken for
  a result of the new parameters;
- the graph canvas, node, and inspector show a persistent and accessible
  `Downstream paused` indicator, and provide a direct `Apply and continue`
  action rather than relying on the user to remember that propagation is
  paused;
- `Apply and continue` leaves isolation mode, invalidates the affected
  descendants once, and resumes normal branch-local execution using only the
  latest accepted node output. Intermediate tuning attempts must not enter the
  downstream execution queue;
- `Cancel tuning` restores the parameters and output that were current when
  isolation began, then leaves downstream results valid when restoration is
  possible. If safe restoration is unavailable, it must clearly invalidate the
  branch instead of silently presenting mismatched results;
- unrelated branches remain runnable, and manual/cached downstream nodes keep
  their existing execution policy when propagation resumes;
- isolation is transient execution/UI state, not part of the saved scientific
  workflow, generated Python, or batch contract. Those durable forms contain
  the current parameters but never carry a paused-propagation flag; toolbar
  `Calculate all` explicitly applies the session before ordinary execution;
- any graph, layout, note, or other history-backed workflow edit applies the
  active session before mutation so Cancel never restores state from a
  different graph revision;
- initially allow only one isolated tuning node at a time. Activating another
  node must first resolve the existing session so nested pauses do not create
  ambiguous stale-state boundaries.

Acceptance coverage should include rapid parameter edits, stale-result
rejection for calculations already in flight, expensive multi-node downstream
branches, branch-local execution, undo/redo during tuning, apply/cancel
semantics, failures in the isolated node, and attempts to save, export, or
batch-run while downstream propagation is paused.

### 8. Graph Polish To Revisit Later

The 0.10 graph-readability work is implemented enough for the current alpha.
Do not treat search, tunnels, notes, insert-on-wire mapping, inspector state, or
thumbnail-visibility persistence as open 0.11 work.

Revisit only when very large workflows show the need:

- minimap/navigation aids;
- alignment guides and optional snap-to-grid;
- additional layout polish beyond current auto-structure and connector
  rerouting.

### 9. AI-Assisted Graph Authoring

This remains later-platform work.

Architecture requirements:

- keep a provider-neutral assistance contract; MCP may be an external adapter,
  but not the internal graph representation;
- accept a structured workflow patch that uses normal operation ids, typed
  ports, and parameter schemas; never let a model mutate the graph directly;
- validate the patch locally, show its assumptions and graph diff, and require
  user approval before applying it;
- expose only bounded, user-visible context such as metadata summaries,
  thumbnails, explicitly labelled sampled context summaries, overlays, table
  previews, and known caveats; never present a sampled context summary as an
  operational VIPP histogram;
- support hosted and local/private providers without arbitrary code execution;
- record provider, model, context summary, validation result, and approval in
  workflow provenance for every applied patch.

Candidate follow-up features can use the same contract for metadata audits,
stale/error and memory-risk review, suspicious-output checks, and proposed
parameter changes. The first implementation should prove safe graph generation
and review before attempting automated tuning.

## Versioned Roadmap

Version numbers are planned alpha milestones, not promises. Each release should
ship with tests, documentation, an example workflow when appropriate, and a
clear release note.

### Released: 0.10.0a1

Theme: graph readability and interactive memory.

Delivered:

- graph search/focus;
- named tunnel management and tunnel reveal/highlight;
- saved graph notes;
- ambiguous insert-on-wire port mapping;
- `Split Axis` and stricter semantic `Split Channels`;
- workflow UI metadata for selected inspector state and optional thumbnail
  visibility;
- branch-local dirty reruns, cache modes, memory guard, and low-memory batch
  retention;
- example workflow metadata and release-facing docs.

### Released: 0.11.0a1 - PSF, Deconvolution, And Microscope Import Foundation

Theme: make PSFs normal graph data, ship the first deconvolution path, and start
the broad microscope-import layer needed for real acquisition files.

Delivered:

- `Born-Wolf PSF` can generate inspectable, saveable 2D/3D PSFs from metadata
  or explicit overrides;
- PSF preparation/validation makes measured or generated PSFs safe to reuse;
- baseline Richardson-Lucy and Richardson-Lucy total-variation deconvolution
  accept named `Image` and `PSF` inputs and use manual/cached execution;
- optional proprietary microscope reader paths are routed through
  `ImageDataset`/`ImageState` boundaries with normalized first-pass metadata;
- docs explain which vendor formats are supported, experimental, or still under
  evaluation.

### Released: 0.11.0a2 - Workflow And Release Hardening

Theme: make saved graphs, generated scripts, and packaged examples fail safely
and behave consistently at the 0.11 baseline.

Delivered:

- atomic validation of restored connections, dynamic outputs, cycles, and
  tunnel definitions;
- generated-Python validation for incomplete, colliding, source-only, and
  custom-entry-point graphs;
- stricter output-path, overwrite, clipboard-retry, and input-count behavior;
- complete example-launcher coverage with explicit unknown-id errors;
- preservation of saved table-column choices before upstream manual
  calculation;
- cross-platform CI, package verification, community guidance, and
  reproducible documentation screenshot capture.

### Released: 0.11.0a3 - Exact And Responsive Large-Image Analysis

Theme: keep large-image scientific calculations exact while moving expensive
inspection and pipeline work away from the Qt thread.

Delivered:

- exact bounded-memory threshold, percentile, metadata, contrast, and
  colocalization calculations without hidden large-array sampling;
- explicit dtype-faithful threshold, Rescale, and Clip behavior plus workflow
  schema version 2 scientific parameters;
- serialized background reruns, stale-result rejection, and responsive
  histogram/label-volume cache reuse;
- consistent split-channel inspection and clearer exact colocalization counts
  with ROI percentages;
- refreshed VIPP name, tagline, and reusable branding assets.

### Released: 0.12.0a1 - Batch Configuration, Provenance, And Explicit Semantics

Goal: make batch execution explicit enough for real analysis runs.

Release gate:

- saved workflow plus saved batch config can reproduce output file names and
  selected outputs;
- every item can emit provenance/status metadata;
- failed items do not hide successful item outputs;
- docs explain how `Batch Output` nodes define what gets saved.

Implementation for this gate includes the saved `vipp_batch_config.json`,
shared deterministic preview/execution planning, `Error`/`Skip`/`Overwrite`
collision policy, latest/archive `vipp_batch_manifest.json` files with atomic
per-item checkpoints, per-item failure isolation with default continuation,
and a final completed/partial/skipped/failed summary. The retained batch
workspace keeps reviewed plans and run evidence inspectable, while a persistent
Previous/Next/slider navigator calculates any paired representative through the
graph without saving the complete batch. Semantic-axis iteration and
plate/well/field HCS traversal are intentionally outside the 0.12 release gate.

### Released: 0.12.0a2 - Interactive Tuning And Execution Feedback

Goal: make expensive interactive graphs easier to tune, interpret, and inspect
without weakening the atomic scientific-cache contract.

Delivered:

- isolated node tuning with apply/cancel behavior and a transient downstream
  execution boundary;
- generic bright actionable and dark waiting execution frontiers, plus an
  attention-colored `Calculate all` control;
- progressive run-scoped thumbnails and inspection payloads while later nodes
  continue, without publishing partial runs into the scientific cache;
- exact-pixel Image-layer reuse, bounded presentation conversion, and
  display-resolution thumbnail rendering;
- configurable graph port-label modes and size-aware auto layout; and
- structured PSF preflight, support, Nyquist, centering, and boundary-tail
  guidance.

Workflow schema remains version 3. Existing schema-3 workflow documents stay
structurally loadable; generated Python exports remain runtime-version pinned
and should be regenerated under the release that will execute them.

### Released: 0.12.0a3 - Batch Reliability And One-File Setup

Goal: make real collection runs quicker to start, safer to recover, and easier
to reopen without weakening batch planning or provenance.

Delivered:

- direct `Run batch` execution through a fresh plan-only preflight, while live
  representative preview remains optional;
- a user-confirmed default output-folder suggestion and a fast path for items
  whose resolved `Skip` destinations all already exist;
- retry handling for transient Windows/cloud-sync artifact locks and
  continuation after exhausted item-sidecar writes when configured;
- optional validated batch-config attachment inside workflow JSON, restored
  without scanning or calculating a preview; and
- one separated Batch workspace toolbar entry with consistent Load-before-Save
  ordering.

Workflow schema remains version 3 and batch-config schema remains version 1.
The optional attached config is excluded from the scientific workflow hash.

### 0.13.0a1 - OME-Zarr Scale And Preview Strategy

Goal: make large, multidimensional OME datasets feel deliberate rather than
accidental, while making graph fragments and proven parameter settings easy to
reuse.

Release gate:

- large local OME-Zarr data can be loaded and previewed without surprising full
  reads for ordinary inspection;
- exported OME-Zarr datasets include useful multiscale metadata;
- docs distinguish analysis-resolution data from preview-resolution rendering;
- users can multi-select nodes and copy/paste a self-contained graph fragment
  with its internal wiring and relative layout through shortcuts or graph
  context menus;
- users can copy one node and paste its complete validated parameter set onto
  another node with the exact same operation id;
- graph paste and parameter paste are atomic, undoable, and do not copy cached
  arrays, runtime state, or external graph dependencies;
- previews remain explicitly separate from analysis-resolution scientific
  arrays and do not change saved numerical results.

### 0.14.0a1 - Scientific Validation Pack

Goal: turn implemented analysis families into defensible scientific methods.

Release gate:

- validation reports state expected values, tolerances, and known limitations;
- methods documentation is consistent with implementation and tests;
- examples can regenerate the reported tables/images.

### 0.15.0a1 - AI-Assisted Pipeline Authoring

Goal: let users describe a workflow and receive a normal, inspectable VIPP
graph without weakening reproducibility.

Release gate:

- generated graphs are ordinary saved workflows;
- invalid generated graphs are rejected before touching the canvas;
- model output is a validated workflow patch, not direct graph mutation;
- user-visible diff/assumption review is required before applying generated
  changes;
- AI context is bounded and uses metadata summaries plus optional low-resolution
  previews, not full-resolution pixels by default;
- applied AI changes record provider/model/context/provenance summaries;
- docs clearly state what leaves the machine for hosted providers.

## Later Milestones

These should wait until the platform, validation base, and user demand are
stronger:

- registration;
- model-backed segmentation;
- stitching and alignment workflows;
- mesh export or surface-output graph contracts;
- broader proprietary reader coverage after the first optional-reader layer is
  stable;
- specialist mitochondrial indices beyond the generic skeleton/network summary
  nodes;
- custom code nodes with explicit review, trust, serialization, and sandboxing
  rules.

## Planning Rules

- Prefer a complete, documented, tested workflow over isolated nodes.
- Prefer metadata-preserving transformations over visually convenient
  shortcuts.
- Prefer explicit output nodes for batch and publication workflows.
- Keep graph behavior serializable and reproducible.
- Treat validation and documentation as part of the feature, not as cleanup.
