# napari-vipp Planning And Roadmap

Last reviewed: 2026-07-06

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

Current public release: `0.9.0a1`.

Current `main` adds post-release graph polish, including named-tunnel example
workflows, schematic net-port tunnel badges, tunnel reveal/highlight, and a
first-pass tunnel manager.

### Workflow Platform

Implemented enough to build on:

- searchable categorized node palette;
- pan/zoom graph canvas with movable node cards;
- typed ports, multi-input nodes, multi-output nodes, cycle rejection, and
  compatibility checks;
- click-to-connect and drag-to-connect wiring;
- delete, duplicate, undo/redo, code inspection, and contextual graph menus;
- insert-on-wire behavior with local make-room movement;
- connector rerouting around nodes;
- named port tunnels for reused channel, mask, ROI, and reference-image sources;
- auto-structure command;
- portable workflow JSON, canvas positions, named tunnels, and strict loading;
- Python export and a first-pass batch runner;
- explicit `Batch Output` nodes for image/table outputs that matter.

### Execution And Responsiveness

Implemented enough to build on:

- background execution for slow graphs;
- global and per-node processing indicators;
- coalesced reruns and stale-result rejection;
- cooperative cancellation for selected expensive operations;
- dirty-node caching;
- manual/cached execution for expensive measurement and table nodes;
- node execution states: not calculated, current, stale, running, and error.

Known gap: cancellation/progress coverage is uneven. Some expensive operations
still behave like black boxes because the underlying libraries do not expose
fine-grained progress hooks.

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

### Analysis Coverage

Implemented enough to build on:

- filtering, projection, thresholding, morphology, label cleanup, watershed
  separation, image math, channel composition, and dtype conversion;
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

These are the active areas that should guide work after `0.9.0a1`.

1. Large graph usability

   Very large workflows need better tunnel management, graph notes, search, and
   layout tools. The graph is the product surface, so visual readability is not
   cosmetic. Minimap/navigation remains useful, but it is a later scale feature
   rather than part of the first `0.10.0a1` pass.

2. Reproducible batch and provenance

   Batch execution exists, but publication-ready batch work needs configuration
   files, per-item manifests, environment capture, richer collection traversal,
   and clearer output intent.

3. Large OME-Zarr and scalable previews

   VIPP can read/write local OME-Zarr, but very large datasets need pyramids,
   preview-resolution controls, lazy histogram/thumbnail sampling, and richer
   label metadata handling.

4. Scientific validation

   Core measurement and colocalization nodes need validation artifacts that can
   be cited or converted into paper figures. The morphology phantom report is a
   good template.

5. AI-assisted graph authoring

   Natural-language workflow generation is still a later feature. It should
   generate ordinary workflow JSON through the existing validator, not bypass
   the graph model.

## Versioned Roadmap

Version numbers are planned alpha milestones, not promises. Each release should
ship with tests, documentation, an example workflow when appropriate, and a
clear release note. If a feature is not validated enough for scientific use, the
UI and docs should say so explicitly.

### 0.10.0a1: Tunnel Management And Graph Readability

Goal: make dense workflows easier to understand and edit.

Implemented in current `main`:

- tunnel subscriber reveal/highlight: select a tunnel and show every input that
  consumes it;
- tunnel management panel for filtering, renaming, deleting, focusing, and
  auditing named sources.

Planned features:

- graph search by node title, operation id, tunnel name, and output tag;
- phase-2 insert-on-wire chooser for ambiguous ports;
- alignment guides and optional snap-to-grid;
- persistence for per-node thumbnail visibility and selected inspector display
  state;
- example workflow update showing dense colocalization or measurement graph
  navigation.

Next after tunnels:

- graph notes/annotations saved in workflow JSON, with canvas position and
  optional node association.

Release gate:

- large colocalization workflow remains readable without drawing repeated
  channel wires;
- workflow JSON round-trips tunnel management changes;
- graph tests cover tunnel management, search, and reveal/highlight behavior.

### 0.11.0a1: Batch Configuration And Provenance

Goal: make batch execution explicit enough for real analysis runs.

Planned features:

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

Planned features:

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

Planned features:

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

Planned features:

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
