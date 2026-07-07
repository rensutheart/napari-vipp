# napari-vipp Planning And Roadmap

Last reviewed: 2026-07-07

This file is the concise planning source of truth. It should answer:

- what VIPP is trying to become;
- what is already good enough to build on;
- what remains risky or incomplete;
- which features belong in the next minor alpha releases.

Detailed implementation notes live in the specialist documents linked below.
Avoid duplicating long node lists or method explanations here.

## Product Direction

VIPP is a napari-native visual workflow builder for bioimage analysis. The graph
canvas is the primary work surface: users should be able to build, inspect,
reuse, export, batch-run, and publish workflows without losing the connection
between image data, metadata, tables, and provenance.

The core scientific workflow families are:

- segmentation and label cleanup for 2D and true 3D data;
- object, intensity, mesh, and skeleton measurements;
- two-channel pixel and object colocalization;
- multi-channel, z-stack, and time-lapse fluorescence data;
- reproducible batch execution with explicit outputs.

Registration, deconvolution, model-backed segmentation, and stitching remain
later milestones. They should not displace graph usability, metadata fidelity,
batch provenance, and validation work.

## Reference Documents

- [architecture.md](architecture.md): implementation architecture and data model.
- [user-guide.md](user-guide.md): current end-user workflow behavior.
- [node-roadmap.md](node-roadmap.md): detailed node inventory and candidate nodes.
- [ome-io-plan.md](ome-io-plan.md): OME and OME-Zarr architecture.
- [io-user-guide.md](io-user-guide.md): supported import/export formats.
- [cache-and-memory.md](cache-and-memory.md): cache modes, memory guard, and
  operation memory policy.
- [measurement-workflows.md](measurement-workflows.md): object, table, mesh, and
  skeleton workflow guidance.
- [analytical-phantom-validation.md](analytical-phantom-validation.md):
  calibrated morphology phantom validation.
- [colocalization-racc-plan.md](colocalization-racc-plan.md): colocalization
  implementation tracking.
- [colocalization-method-notes.md](colocalization-method-notes.md):
  publication-facing colocalization methods.
- [skeleton-nodes.md](skeleton-nodes.md): skeleton node usage.
- [mitomorph-feature-parity.md](mitomorph-feature-parity.md):
  MitoMorph-inspired feature parity.
- [object-mesh-morphology-plan.md](object-mesh-morphology-plan.md): mesh and
  object morphology details.
- [research-and-publication.md](research-and-publication.md): publication and
  evidence tracking.

## Current Baseline

Current public release: `0.10.0a1`.

The `0.10.0a1` alpha adds two implemented clusters:

- graph readability: named-tunnel example workflows, schematic net-port tunnel
  badges, tunnel reveal/highlight, a tunnel manager, graph search/focus,
  ambiguous insert-on-wire mapping, saved graph notes, and workflow UI-state
  metadata;
- interactive execution and memory: branch-local dirty reruns, explicit cache
  modes, cache/RAM status, auto memory guard, per-node `Keep output cached`, and
  low-memory batch retention.

### Workflow Platform

Implemented enough to build on:

- searchable categorized node palette and a pan/zoom graph canvas;
- typed ports, multi-input/multi-output nodes, cycle rejection, compatibility
  checks, and click/drag wiring;
- delete, duplicate, undo/redo, code inspection, contextual graph menus, and
  insert-on-wire behavior with local make-room movement;
- connector rerouting, named port tunnels, tunnel reveal/highlight, first-pass
  tunnel management, and saved graph notes with canvas positions;
- auto-structure command;
- portable workflow JSON, canvas positions, named tunnels, graph notes, and
  strict loading;
- Python export and a first-pass batch runner;
- explicit `Batch Output` nodes for image/table outputs that matter.

### Execution And Responsiveness

Implemented enough to build on:

- background execution for slow graphs;
- global and per-node processing indicators;
- coalesced reruns and stale-result rejection;
- cooperative cancellation for selected expensive operations;
- dirty-node caching and branch-local reruns for graph edits;
- manual/cached execution for expensive measurement and table nodes;
- node execution states: not calculated, current, stale, running, and error;
- cache modes: keep all node outputs, smart interactive cache, and low-memory
  mode;
- cache/RAM status, auto memory guard, and a per-node `Keep output cached`
  affordance;
- batch execution retention that keeps explicit outputs and prunes item-level
  intermediates.

Known gap: cancellation/progress coverage is uneven. Some expensive operations
still behave like black boxes because the underlying libraries do not expose
fine-grained progress hooks. Cache size is a practical estimate, not a complete
Python heap profile, and most processing nodes are still eager.

### Data, Metadata, And I/O

Implemented enough to build on:

- OME-NGFF-inspired `ImageState` on image-like outputs;
- `TableState` on table outputs;
- semantic axes, scale, units, origin, dtype, image kind, channels, timepoints,
  and history;
- explicit axis/dtype/spatial calibration nodes;
- OME-TIFF, ImageJ TIFF, conventional TIFF, common raster formats, NumPy, and
  local OME-Zarr 0.4/0.5;
- OME-Zarr label-group import/export through image-plus-label analysis packages.

Known gap: large-data behavior is still mostly pragmatic rather than fully
lazy. Thumbnail and histogram generation need clearer sampling and pyramid
strategies before very large OME-Zarr datasets become comfortable.
Multiple series should be investigated as source/import or batch-binding
structure rather than a normal image axis; only expose series to axis-splitting
tools when a reader materializes it as a real metadata-backed image axis.

### Analysis Coverage

Implemented enough to build on:

- filtering, projection, thresholding, morphology, label cleanup, watershed
  separation, image math, channel composition, explicit axis splitting, and
  dtype conversion;
- object morphology, object intensity, table merge, metadata annotation, table
  column selection, grouped summaries, and property-based label filtering;
- calibrated extended morphology and first-pass true-3D mesh morphology;
- skeletonization, skeleton keypoints, branch labels, graph overlays, pruning,
  branch tables, graph tables, branch summaries, and overall network summaries;
- pixel colocalization metrics, ROI-masked colocalization, Costes thresholding,
  colocalized-voxel overlays, RACC-like index outputs, object colocalization,
  label-overlap association, nearest-object distances, and event localization.

Known gap: several analysis families are implemented but not yet validated to
publication depth across realistic datasets. The next research risk is not only
adding nodes; it is proving that the nodes behave correctly under common
microscopy assumptions.

### Validation And Examples

Implemented examples cover:

- label cleanup;
- object intensity measurement;
- table merge and measurement summaries;
- derived object morphology;
- true-3D mesh morphology;
- skeleton/network QC and advanced skeleton networks;
- pixel and object colocalization;
- named channel tunnels in colocalization workflows.

Implemented validation includes automated tests plus the calibrated analytical
phantom report for rectangles, cuboids, spheres, ellipsoids, and anisotropic
voxel sizes.

Known gap: validation reports are still uneven across segmentation, skeleton,
colocalization, batch/provenance, and OME-Zarr round-tripping.

## Active Gaps

These are the active areas that should guide work after `0.10.0a1`.

### Large Graph Usability

Implemented progress: tunnel reveal/highlight, tunnel management, graph notes,
named-tunnel examples, graph search/focus, insert-on-wire make-room behavior,
ambiguous insert-on-wire port mapping, optional thumbnail-visibility
persistence, selected inspector-state persistence, and connector rerouting.

Minimap/navigation remains useful, but it should wait until search, tunnel
management, and layout polish are in place.

### Reproducible Batch And Provenance

Implemented progress: first-pass collection batch execution, explicit
`Batch Output` nodes, dry-run preview, workflow/script reproducibility
artifacts, and low-memory item retention.

Next steps:

- define a saved `batch_config.yaml` or equivalent schema;
- emit per-item provenance manifests with workflow hash, package versions,
  input identity, source metadata, output paths, and status;
- improve the dry-run table so input bindings, skipped items, and planned
  outputs are visible before execution;
- add semantic-axis iteration for timepoints, channels, z-slices, or selected
  combinations;
- add first-pass plate/well/field collection traversal for HCS-style layouts;
- summarize failed, skipped, and completed items after a batch run.

### Large OME-Zarr And Scalable Previews

Implemented progress: local OME-Zarr 0.4/0.5 read/write, OME-Zarr image plus
label analysis packages, cache modes, and a documented operation memory policy.

Next steps:

- generate OME-Zarr pyramids for exported images;
- round-trip label colors and label-property tables where practical;
- add preview-resolution controls for thumbnails and inspector views;
- make histograms and thumbnails sampled or chunk-aware for large arrays;
- declare operation capabilities such as eager, lazy-safe, memory-heavy, and
  scale-aware;
- warn before eager-only nodes materialize very large lazy arrays;
- investigate anonymous HTTP reads for public OME-Zarr datasets.

### Scientific Validation

Implemented progress: automated tests plus calibrated analytical morphology
phantoms for rectangles, cuboids, spheres, ellipsoids, and anisotropic voxel
sizes.

Next steps:

- add colocalization validation with deterministic threshold and overlap
  scenarios;
- validate object overlap, nearest-distance, and event-localization assumptions;
- validate watershed/touching-object separation on geometric and
  microscopy-like phantoms;
- validate skeleton/network outputs with known endpoints, junctions, cycles,
  branch lengths, and anisotropic spacing;
- produce reproducible example outputs that can become methods figures or
  supplementary artifacts.

### AI-Assisted Graph Authoring

Implemented progress: none; this remains later-platform work.

Next steps:

- generate ordinary workflow JSON through the existing workflow validator;
- validate operation ids, port contracts, and parameter schemas before applying
  a generated graph;
- show a preview/diff before changing the canvas;
- keep full-resolution pixels local by default and make hosted-provider data
  sharing explicit.

## Versioned Roadmap

Version numbers are planned alpha milestones, not promises. Each release should
ship with tests, documentation, an example workflow when appropriate, and a
clear release note. If a feature is not validated enough for scientific use, the
UI and docs should say so explicitly.

### 0.10.0a1: Graph Readability And Interactive Memory

Goal: make dense workflows easier to understand, edit, and inspect without
surprising recomputation or hidden memory growth.

Implemented in current development:

- tunnel subscriber reveal/highlight: select a tunnel and show every input that
  consumes it;
- tunnel management panel for filtering, renaming, deleting, focusing, and
  auditing named sources;
- graph search by node title, operation id, tunnel name, and output tag;
- graph notes/annotations saved in workflow JSON with canvas position;
- branch-local dirty reruns for adding nodes, connecting new branches, inserting
  on wires, disconnecting branches, deleting nodes, and tunnel edits;
- ambiguous insert-on-wire port chooser with dynamic Split Channels output
  inference;
- explicit `Split Axis` node for splitting time, Z, or other non-channel stack
  axes separately from semantic channel splitting;
- selected inspector state plus optional per-node thumbnail visibility in
  workflow JSON;
- cache modes, cache/RAM status, auto memory guard, per-node `Keep output
  cached`, and low-memory batch retention;
- user documentation in [cache-and-memory.md](cache-and-memory.md).

Release status: ready for the `0.10.0a1` alpha after automated release checks
and the final manual smoke pass.

Release gate:

- large colocalization workflow remains readable without drawing repeated
  channel wires;
- routine graph edits reuse unaffected cached outputs;
- workflow JSON round-trips tunnel management, graph-note changes, inspector
  state, and optional thumbnail visibility;
- smart/low-memory cache modes prune and restore expected outputs without losing
  explicit output intent;
- graph tests cover tunnel management, notes, search, reveal/highlight, cache
  modes, memory guard behavior, and insert-on-wire mapping behavior.

### 0.11.0a1: Batch Configuration And Provenance

Goal: make batch execution explicit enough for real analysis runs.

Already implemented foundations:

- local collection batch runner;
- explicit `Batch Output` nodes;
- low-memory retention during item execution;
- optional workflow snapshot and Python script artifacts.

Next steps:

- `batch_config.yaml` or equivalent saved batch configuration;
- per-item provenance manifest with workflow hash, package versions, input
  identity, source metadata, and output paths;
- richer batch dry-run table before execution;
- semantic-axis iteration over timepoints, channels, z-slices, or selected
  combinations;
- first-pass plate/well/field collection traversal for HCS-style layouts;
- clearer output manifest for all `Batch Output` nodes;
- better handling of multiple Image Source nodes bound to related collections;
- batch failure summary with skipped, failed, and completed items.

Release gate:

- a saved workflow plus saved batch config can reproduce output file names and
  selected outputs;
- a failed item does not hide successful item outputs;
- docs explain how `Batch Output` nodes define what gets saved.

### 0.12.0a1: OME-Zarr Scale And Preview Strategy

Goal: make large, multidimensional OME datasets feel deliberate rather than
accidental.

Already implemented foundations:

- local OME-Zarr 0.4/0.5 image read/write;
- OME-Zarr image plus label analysis package import/export;
- cache modes and operation memory policy documentation.

Next steps:

- generated OME-Zarr pyramids for exported image datasets;
- OME-Zarr label colors and label-property table round-tripping where practical;
- preview-resolution selection for thumbnails and inspector views;
- lazy/sampled histograms for large arrays;
- thumbnail strategy that avoids full-array materialization unless explicitly
  requested;
- operation capability declarations for eager, lazy-safe, memory-heavy, and
  scale-aware operations;
- anonymous HTTP read investigation for public OME-Zarr datasets.

Release gate:

- large local OME-Zarr data can be loaded and previewed without surprising full
  reads for ordinary inspection;
- exported OME-Zarr datasets include useful multiscale metadata;
- docs distinguish analysis-resolution data from preview-resolution rendering.

### 0.13.0a1: Scientific Validation Pack

Goal: turn implemented analysis families into defensible scientific methods.

Already implemented foundations:

- automated tests for implemented operations and workflows;
- calibrated analytical morphology phantom validation;
- publication-facing colocalization method notes.

Next steps:

- colocalization validation report using deterministic synthetic data and known
  threshold/overlap scenarios;
- object association validation report for overlap, nearest distance, and event
  localization assumptions;
- watershed/touching-object validation on rectangles, disks/spheres, and
  representative microscopy-like phantoms;
- skeleton/network validation report with known endpoints, junctions, cycles,
  branch lengths, and anisotropic spacing;
- reproducible example-output artifacts for publication figures and methods;
- RACC numerical-core decision: keep VIPP-owned implementation, share a common
  core with the RACC plugin, or document the intentional separation.

Release gate:

- validation reports state expected values, tolerances, and known limitations;
- methods documentation is consistent with implementation and tests;
- examples can regenerate the reported tables/images.

### 0.14.0a1: AI-Assisted Pipeline Authoring

Goal: let users describe a workflow and receive a normal, inspectable VIPP
graph, without weakening reproducibility.

Next steps:

- provider-agnostic AI settings, with user-managed keys and local-provider room;
- generated workflow JSON from natural language plus the live node registry;
- validation against real operation ids, port contracts, and parameter schemas;
- preview/diff before applying generated graph changes;
- optional sharing of metadata summaries and downsampled thumbnails;
- no full-resolution pixel sharing by default;
- safe failure modes when the model proposes unavailable or incompatible nodes.

Release gate:

- generated graphs are ordinary saved workflows;
- invalid generated graphs are rejected before touching the canvas;
- docs clearly state what leaves the machine for hosted providers.

## Later Milestones

These should wait until the platform and validation base are stronger:

- minimap/navigation aids for very large graphs;
- alignment guides and optional snap-to-grid for manual graph layout polish;
- registration;
- deconvolution with PSF handling;
- model-backed segmentation;
- stitching and alignment workflows;
- mesh export or surface-output graph contracts;
- specialist mitochondrial indices beyond the generic skeleton/network summary
  nodes;
- custom code nodes with explicit review, trust, serialization, and sandboxing
  rules.

## Planning Rules

- Prefer a complete, documented, tested workflow over isolated nodes.
- Prefer metadata-preserving transformations over visually convenient shortcuts.
- Prefer explicit output nodes for batch and publication workflows.
- Keep graph behavior serializable and reproducible.
- Treat validation and documentation as part of the feature, not as cleanup.
