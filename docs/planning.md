# napari-vipp Planning And Roadmap

Last reviewed: 2026-07-10

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
routing are part of the 0.11 baseline. Near-term work is validation on real
data, batch provenance, and scalable OME-Zarr previews. Registration,
model-backed segmentation, stitching, and AI-assisted graph authoring remain
later milestones.

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

## Current Public Baseline

Current public release: `0.11.0a2`.

The 0.11 alpha is now the hardened PSF/restoration and
microscope-import-foundation baseline on top of the 0.10 graph-readability and
interactive-memory work.
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
- Python export, first-pass collection batch execution, explicit `Batch Output`
  nodes, dry-run preview, multi-source bindings, and saved workflow/script
  artifacts;
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
  more deliberate lazy/sampled preview strategy;
- broad proprietary microscope import is active development rather than a
  public baseline guarantee; reader support should be documented per format as
  supported, experimental, or metadata-incomplete;
- validation is strong for calibrated morphology but still uneven for
  colocalization, watershed, skeleton/network, batch/provenance, and OME-Zarr
  round-tripping.

## Active TODOs After 0.11

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
nodes, dry-run preview, multi-source binding, low-memory batch retention, and
workflow/script reproducibility artifacts.

Still needed:

- saved `batch_config.yaml` or equivalent batch configuration;
- per-item provenance manifest with workflow hash, package versions, input
  identity, source metadata, output paths, and status;
- clearer failure summary with skipped, failed, and completed items;
- richer output manifest for all `Batch Output` nodes;
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

### 6. Graph Polish To Revisit Later

The 0.10 graph-readability work is implemented enough for the current alpha.
Do not treat search, tunnels, notes, insert-on-wire mapping, inspector state, or
thumbnail-visibility persistence as open 0.11 work.

Revisit only when very large workflows show the need:

- minimap/navigation aids;
- alignment guides and optional snap-to-grid;
- additional layout polish beyond current auto-structure and connector
  rerouting.

### 7. AI-Assisted Graph Authoring

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

### Next: 0.12.0a1 - Batch Configuration And Provenance

Goal: make batch execution explicit enough for real analysis runs.

Release gate:

- saved workflow plus saved batch config can reproduce output file names and
  selected outputs;
- every item can emit provenance/status metadata;
- failed items do not hide successful item outputs;
- docs explain how `Batch Output` nodes define what gets saved.

### 0.13.0a1 - OME-Zarr Scale And Preview Strategy

Goal: make large, multidimensional OME datasets feel deliberate rather than
accidental.

Release gate:

- large local OME-Zarr data can be loaded and previewed without surprising full
  reads for ordinary inspection;
- exported OME-Zarr datasets include useful multiscale metadata;
- docs distinguish analysis-resolution data from preview-resolution rendering.

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
