# napari-vipp Planning Notes

Last reviewed: 2026-07-03

This document is the consolidated planning source of truth. It separates
implemented behavior from active TODOs so future development can start from the
right place without re-reading old discussion history.

Status labels used below:

- **Implemented**: present in the codebase and covered by normal operation.
- **Partial**: useful behavior exists, but known planned work remains.
- **TODO**: not implemented yet.
- **Later**: deliberately deferred until the core workflow platform is more
  stable.

Related reference documents:

- [architecture.md](architecture.md) for implementation architecture.
- [node-roadmap.md](node-roadmap.md) for prioritized node ideas.
- [ome-io-plan.md](ome-io-plan.md) for OME import/export decisions.
- [io-user-guide.md](io-user-guide.md) for current file-format behavior.
- [measurement-workflows.md](measurement-workflows.md) for object, mesh,
  skeleton, and table-composition workflow guidance.
- [analytical-phantom-validation.md](analytical-phantom-validation.md) for
  deterministic calibrated morphology phantom validation.
- [colocalization-racc-plan.md](colocalization-racc-plan.md) for the current
  colocalization/RACC implementation plan.
- [colocalization-method-notes.md](colocalization-method-notes.md) for
  publication-facing colocalization method documentation.
- [skeleton-nodes.md](skeleton-nodes.md) for skeleton node usage.
- [mitomorph-feature-parity.md](mitomorph-feature-parity.md) for
  MitoMorph-inspired feature parity.
- [object-mesh-morphology-plan.md](object-mesh-morphology-plan.md) for current
  object and 3D mesh morphology status plus remaining follow-up work.
- [research-and-publication.md](research-and-publication.md) for publication
  and evidence tracking.

## Product Direction

The graph is the primary work surface. A useful VIPP feature should form part
of a reproducible bioimage workflow, not merely add an isolated image filter.

First-class workflow families:

- nuclei and cell segmentation;
- puncta and spot analysis;
- mitochondrial object and network analysis;
- pixel-based and object-based colocalization/localization;
- 2D images and true 3D fluorescence z-stacks.

Registration and deconvolution are important, but remain later milestones.

## Implemented Capabilities

### Graph Editing And Layout

Implemented:

- searchable grouped node palette;
- movable node cards on a pan/zoom canvas with auto-expanding scene bounds;
- node and connection deletion without deleting attached nodes accidentally;
- widened connector hit targets for easier selection, deletion, and context
  menus;
- click-to-connect and drag-to-connect wiring with compatible/incompatible drop
  feedback;
- cycle rejection and port-type validation;
- slot-aware multi-input connections;
- per-port multi-output connections;
- dynamic Split Channels output counts;
- connector rerouting while nodes move;
- right-click node menus for Delete, Inspect Code, Duplicate Node, and
  contextual Pin/Unpin;
- code inspection dialogs with Python syntax highlighting;
- undo/redo for graph and parameter edits;
- save/load of canvas positions;
- one-shot `Auto structure graph` command with undo;
- responsive toolbars that progressively collapse controls into `Settings`;
- insert-on-wire by dragging a palette node, dragging an existing loose node,
  or using `Insert node here...` from the connector context menu;
- insert-on-wire full splice for one-input/one-output nodes;
- partial upstream-only insert for single-input/multi-output nodes such as
  Split Channels;
- place-in-gap behavior for ambiguous multi-input/multi-output nodes;
- deterministic local make-room movement for insert-on-wire, shifting only the
  target/downstream side as needed.

### Workflow Persistence, Export, And Batch Foundation

Implemented:

- portable version-1 JSON workflow files;
- stable node ids, operation ids, parameter values, source/target ids, input
  slots, output slots, and canvas positions;
- strict loading that rejects unknown operations, malformed nodes, duplicate
  ids, invalid positions, dangling connections, and multiply occupied slots;
- Image Source parameters persisted as ordinary workflow parameters;
- Python export generated from the same graph model used by the UI;
- exported `run_pipeline()` and folder-oriented `batch_process()`;
- command-line entry point for exported scripts;
- terminal graph outputs saved by generated scripts;
- image-like and table outputs handled by the exporter.
- first-pass `Run batch...` UI that runs a workflow once per matched file in a
  folder, binds collection Image Source nodes per item, saves terminal image
  outputs and table outputs, and optionally writes a workflow JSON snapshot plus
  exported Python script next to the results.

The exporter is headless but still requires the `napari-vipp` Python package.

### Execution Model

Implemented:

- background-thread execution for known slow image-processing graphs;
- global indeterminate processing indicator;
- per-node busy rings for the slow node that triggered background execution;
- dirty-node caching of prior outputs and states;
- coalesced reruns while a long calculation is active, discarding stale worker
  results;
- `Run all in BG` toggle to force graph updates onto the worker;
- manual/cached execution for expensive table-producing nodes:
  `Measure Objects`, `Measure Objects + Intensity`,
  `Measure 3D Mesh Morphology`, `Analyze Skeleton`,
  `Measure Skeleton Branches`, `Skeleton Graph Tables`, and `Measure Overall
  Skeleton Network`;
- automatic branch-summary tables from row-per-branch skeleton measurements;
- graph-card and inspector `Calculate` / `Recalculate` controls;
- inspector `Auto Recalculate`, off by default, for manual nodes;
- explicit node execution states: not calculated, current, stale, running, and
  error;
- node-card state colours: gray for not calculated, green for current, orange
  for stale, and red for error;
- workflow persistence of the auto-recalculate preference.

Generated Python export and headless pipeline execution calculate manual nodes
by default so batch output is deterministic.

### Data State And Metadata

Implemented:

- OME-NGFF-inspired `ImageState` carried with every image-like output;
- `TableState` carried with every table output;
- shape, dtype, semantic axes, axis types, units, scale, origin, image kind,
  value summaries, bit-depth summaries, source metadata, and operation history;
- OME-NGFF-like `multiscales` metadata when available;
- inferred axes for plain arrays with UI indication that inference was used;
- table row count, column count, stable column names, units where known, source,
  measurement set, and history;
- table outputs shown in the inspector and excluded from image thumbnails and
  histograms;
- explicit axis and dtype nodes: `Convert Dtype`, `Select Axis Slice`,
  `Reorder Axes`, `Set Pixel Size / Units`, and `Rescale Axes`;
- physical scale/units following the moved or rescaled data axis;
- channel and time metadata remaining attached to their data.

### Preview, Inspection, And Display

Implemented:

- per-node thumbnails with Slice/MIP modes;
- global thumbnail show/hide;
- thumbnail contrast modes: min-max, percentile, and raw;
- monochrome thumbnail colormap selection;
- RGB/multichannel thumbnail rendering with channel colors;
- input histograms for thresholding and intensity nodes where relevant;
- output image and label histograms;
- clearer 2D versus stack histogram labeling;
- label volume distribution histogram with log-volume-axis option;
- image/mask/label/RGB pinning as persistent napari preview layers;
- pinned-node outline on the graph;
- optional `Follow napari dims`;
- VIPP-local `View dims` controls for non-XY semantic axes such as T, Z, and C;
- bidirectional synchronization between VIPP-local dims and napari dims;
- canonical axis mapping through dropped axes and rescaled axes, so selecting a
  downstream node such as Split Channels -> Gaussian Blur still moves the same
  source Z/T/C position;
- compact/menu view-dims layout for narrow dock widths;
- preservation of napari layer scale for `Set Pixel Size / Units` and
  `Rescale Axes`.
- native colocalization scatter inspection with square axes, endpoint axis
  labels, selectable density colormap, log-count display, channel-coloured
  draggable threshold lines, and stable layout while thresholds are dragged.

### OME And Raster I/O

Implemented:

- shared headless reader/writer registry;
- normalized dataset metadata;
- OME-TIFF, ImageJ TIFF, and conventional TIFF import/export;
- common raster import including PNG/JPEG-style formats;
- 2D raster export for standard formats when output is 2D;
- local OME-Zarr 0.4/0.5 image import/export with lazy Dask reads;
- OME-Zarr label-group import/export via image-plus-label analysis packages;
- adaptive image/series selection;
- stored collection-binding intent.

### Tables, Measurements, And Skeleton Analysis

Implemented:

- object morphology and optional intensity measurements;
- derived object morphology ratios and 2D shape moments on the object
  measurement nodes;
- first-pass true-3D mesh morphology with marching-cubes surface area, mesh
  volume, sphericity, convex-hull metrics, 3D solidity, anisotropic scale
  handling, and per-object mesh status/error reporting;
- table merging, metadata columns, column selection, and grouped summaries;
- CSV/TSV saving for selected table outputs and exported scripts;
- skeletonization and skeleton graph analysis for 2D/3D;
- endpoint, junction, isolate, edge, cycle, and connected-component skeleton
  metrics;
- skeleton keypoint masks;
- skeleton component and branch labels;
- terminal spur pruning;
- RGB skeleton graph overlays;
- branch-level length, endpoint distance, tortuosity, start/end coordinates,
  and calibrated physical length where available.
- branch-summary distributions with length/tortuosity statistics and
  branch-type counts/fractions;
- domain-normalized skeleton connectedness summaries, including per-component
  and per-length endpoint/junction/branch/cycle metrics.
- first-pass two-channel colocalization metrics, thresholded colocalized-voxel
  visualization, Costes threshold write-back, inspector scatter-density
  threshold controls, RACC index images, and masked/ROI-restricted variants of
  the same node family.

### Implemented Node Catalogue

The live registry currently contains 98 nodes.

- **Image Data**: Image Source, Crop Stack, Select Axis Slice, Reorder Axes,
  Set Pixel Size / Units, Rescale Axes, Extract Channel, Combine Channels,
  Split Channels, Composite -> RGB, Assign Channel Colors, Calculate New Image,
  Add, Subtract, Ratio, Mask Image, Logical AND, Logical OR, Logical XOR,
  Convert Dtype, Invert, Save Image.
- **Intensity & Contrast**: Linear Scale + Offset, Gamma Correction, Rescale
  Intensity, Normalize, Clip.
- **Filtering**: Average Blur, Gaussian Blur, Gaussian Blur 3D, Median Filter,
  Bilateral Filtering, Non-Local Means, Rolling-Ball Background, Subtract
  Background, Difference of Gaussians, Unsharp Mask, Sobel Edges, Canny Edges,
  Laplace Filter.
- **Projection**: Maximum Projection, Project Image, Orthogonal Projection.
- **Segmentation (thresholds)**: Otsu, Triangle, Li, Yen, Isodata, Minimum,
  Binary, Hysteresis, Adaptive Mean, Adaptive Gaussian, Sauvola, Niblack.
- **Segmentation (object separation)**: Auto Watershed From Mask, Euclidean
  Distance Transform, H-Maxima Markers, Marker-Controlled Watershed, Expand
  Labels.
- **Morphology**: Dilation, Erosion, Opening, Closing, Top Hat, Black Hat,
  Morphological Gradient, Fill Holes, Remove Small Objects, Skeletonize,
  Skeleton Keypoints, Skeleton Graph Overlay, Prune Skeleton Branches.
- **Label Operations**: Label Connected Components, Clear Border Objects, Filter
  Labels By Volume, Filter Labels By Property, Relabel Sequential, Label
  Skeleton Components, Label Skeleton Branches.
- **Measurements**: Measure Objects, Measure Objects + Intensity,
  Measure 3D Mesh Morphology, Analyze Skeleton, Measure Skeleton Branches,
  Summarize Skeleton Branches, Skeleton Graph Tables, Measure Overall Skeleton
  Network, Merge Tables, Add Metadata Columns, Select Table Columns,
  Summarize Measurements.
- **Colocalization & Spatial Analysis**: Colocalization Metrics, Masked
  Colocalization Metrics, Colocalized Voxels, Masked Colocalized Voxels, RACC
  Index, Masked RACC Index, Object Colocalization Metrics, Label Overlap
  Association, Nearest Object Distance, Event Localization.

## Validated Reference Workflows

Implemented reference label-cleanup workflow:

```text
image
  -> Gaussian Blur
  -> Otsu Threshold
  -> Split Channels
  -> Fill Holes
  -> Label Connected Components
  -> Clear Border Objects
  -> Filter Labels By Volume
  -> cleaned labels
```

Implemented reference measurement-summary workflow:

- `examples/synthetic-measurement-summary.json`;
- deterministic time-series sample;
- grouped object counts and area summaries.

Implemented reference object/mesh morphology workflows:

- `examples/synthetic-derived-object-morphology.json`;
- `examples/synthetic-3d-mesh-morphology.json`;
- deterministic 2D object-shape and anisotropic true-3D object samples covering
  derived object morphology, 2D moments, mesh surface/volume, convex-hull
  metrics, 3D solidity, sphericity, and tiny-object mesh failure reporting.

Implemented reference skeleton/network workflows:

- `examples/synthetic-skeleton-qc.json`;
- `examples/synthetic-advanced-skeleton-network.json`;
- deterministic sparse 3D skeleton samples covering keypoints, components,
  branch labels, graph overlays, branch tables, graph node/edge tables,
  branch-summary tables, overall-network summary tables, pruning,
  disconnected fragments, loops, time-indexed blocks, and anisotropic physical
  calibration.

Implemented reference colocalization/RACC workflow:

- `examples/synthetic-colocalization-racc.json`;
- `examples/synthetic-object-colocalization-association.json`;
- deterministic two-channel `CZYX` sample covering partial overlap,
  single-channel-only structures, offset puncta, background gradients,
  inspector scatter threshold guides, colocalized-voxel overlays, whole-image
  and ROI-masked metrics, RACC index output, object-level colocalization rows,
  label-overlap association, nearest-object distances, event localization, and
  merged morphology/colocalization tables.

## Recently Completed Milestone

### Pixel Colocalization And RACC First Pass

Implemented:

- `Colocalization Metrics`, `Colocalized Voxels`, and `RACC Index` for
  unrestricted same-shaped two-channel inputs.
- `Masked Colocalization Metrics`, `Masked Colocalized Voxels`, and `Masked
  RACC Index` for ROI-restricted pixel/voxel analysis.
- Manual thresholding and `Costes auto` thresholding in normalized `0..255`
  intensity units.
- Costes-derived thresholds written back into the visible node parameters, so
  the automatic threshold choice is inspectable and reproducible.
- A native inspector scatter panel on colocalization threshold nodes, replacing
  the earlier separate scatter graph node idea.
- Interactive scatter threshold dragging that switches the node to manual
  thresholds and updates the corresponding threshold parameter.
- Scatter-panel UX polish: square X/Y plot area, axis start/end labels,
  rotated channel-2 axis label, channel-coloured threshold lines, selectable
  density colormap, log-count toggle, and stable layout while dragging.
- Deterministic synthetic colocalization sample and example workflow:
  `examples/synthetic-colocalization-racc.json`.
- Deterministic object-aware colocalization example workflow:
  `examples/synthetic-object-colocalization-association.json`.
- Tests covering colocalization metrics, overlays, RACC output, ROI masking,
  Costes threshold write-back, inspector scatter interaction, and the example
  workflow.

This completed the first pixel-based colocalization/RACC batch. The
object-level and association-table batch is now tracked below as implemented.

### Object Colocalization And Association Tables

Implemented:

- `Object Colocalization Metrics`: label-aware two-channel metrics with one row
  per object, `label_id`, leading axis indices, Pearson, Manders, overlap
  coefficients, threshold voxel counts, intensity sums, and Costes diagnostics.
- `Label Overlap Association`: label-label overlap rows with reference/target
  label ids, overlap voxel counts, overlap fractions, and intersection over
  union.
- `Nearest Object Distance`: nearest target-object centroid distance per
  reference label, including calibrated physical distance when axis metadata is
  available.
- `Event Localization`: event/puncta-to-region table rows for labels, masks, or
  ROIs, reporting dominant overlapping region and event overlap fraction.
- Tests covering the object colocalization, overlap association,
  nearest-distance, and event-localization table contracts.
- Publication-facing method documentation for Pearson, Manders, overlap
  coefficient, Costes thresholding, RACC-like index, ROI restriction,
  object-restricted analysis, and object association assumptions:
  [colocalization-method-notes.md](colocalization-method-notes.md).

### Calibrated Extended Object Morphology

Implemented:

- calibrated physical centroid and bounding-box coordinate columns when spatial
  scale metadata is available;
- calibrated equivalent diameter, bounding/fill size, 2D convex area, maximum
  Feret diameter, major/minor axis length, bounding-box side lengths, and
  inertia tensor eigenvalues for `Measure Objects` and `Measure Objects +
  Intensity`;
- isotropic 2D physical perimeter, Crofton perimeter, and perimeter-to-area
  ratio, with anisotropic 2D physical perimeter columns left as `NaN` rather
  than estimated from an ambiguous scalar scale;
- regression tests covering calibrated 2D and 3D extended morphology columns;
- analytical phantom validation report covering exact rectangle/cuboid phantoms
  and tolerance-based sphere/ellipsoid phantoms:
  [analytical-phantom-validation.md](analytical-phantom-validation.md).

## Active TODOs

### Current Near-Term Order

1. Add cancellation/progress for expensive manual/background feature families
   where libraries expose useful progress hooks.
2. Extend the first-pass batch UI with stable item identities, output
   templates, and multiple independently bound sources.

### Execution Platform TODOs

- cancellation for running background/manual work;
- percentage progress for operations that can report it;
- richer failure recovery and retry behavior;
- adoption by additional colocalization/localization, mesh export/preview,
  deconvolution, and very expensive background-estimation nodes;
- visible progress support for slow nodes without making the UI feel frozen.

### Graph Editor TODOs

- **Port tunnels / named wires**: allow a user to label an output port or input
  port and reuse that named connection elsewhere on the graph without drawing a
  long visible edge. This is especially important for colocalization workflows,
  where the same two split-channel outputs repeatedly feed metrics, overlays,
  RACC, ROI-restricted variants, and object-level table workflows.
  - Output tunnel: mark a node output as a named source such as `Ch1` or `Ch2`;
    downstream tunnel receivers can subscribe to that source as if a normal
    wire were connected.
  - Input tunnel: mark a node input as consuming a named source, keeping the
    port contract and type validation identical to explicit wiring.
  - The graph should render compact tunnel badges/labels at the participating
    ports and avoid drawing full-length edges unless the user asks to reveal
    them.
  - Workflow JSON and Python export must serialize tunnels as ordinary graph
    semantics, not UI-only state, so batch execution remains reproducible.
  - Validation must still reject ambiguous names, incompatible port types,
    cycles, and missing tunnel sources.
- phase-2 insert-on-wire chooser for ambiguous input/output ports;
- Obsidian-like live structure mode with optional animation;
- optional pinned/anchored nodes for live layout;
- graph annotations/notes;
- alignment guides and snap-to-grid;
- manual local "make room downstream" command;
- minimap/navigation aids;
- optional edge labels for multi-port workflows.

Architecture preference for future graph work:

- keep layout computation in a pure helper module such as
  `core/graph_layout.py`;
- expose graph-view position application through one method so workflow restore,
  auto-structure, and future live layout share the same path;
- keep insert-on-wire validation and rollback centralized so palette drops,
  connector menus, and AI-assisted graph edits use the same behavior.

### Workflow Persistence And Provenance TODOs

- persist per-node thumbnail visibility;
- persist inspector and histogram UI state;
- persist graph notes/annotations;
- record environment/package provenance;
- optional YAML workflow format;
- portable sharing that rebases paths or packages required input assets;
- `batch_config.yaml` for reproducible collection processing.

### Preview, Display, And Scalability TODOs

- tune responsive breakpoints after testing more dock sizes;
- decide whether compact menus remember their last open position;
- generated OME-Zarr pyramids;
- preview-resolution selection;
- large-data thumbnail/histogram strategies that avoid expensive full-array
  materialization;
- operation capability declarations for memory-aware lazy materialization.

### OME And Collection I/O TODOs

- OME-Zarr label colors;
- OME-Zarr label-property tables;
- richer collection batch execution with stable item identities and output
  templates;
- multiple independently bound sources;
- plate/well/field browsing;
- anonymous HTTP reads.

### Analysis Node TODOs

Skeleton and network analysis:

- richer skeleton feature families beyond the implemented branch summaries and
  normalized overall-network metrics, especially specialist mitochondrial
  connectedness indices that should remain optional rather than forced into
  the generic nodes;
- optional graph-library export/analysis hooks for users who want to take the
  explicit node/edge tables into external network tooling.

Object and mesh morphology:

- richer region/object properties beyond the current measurement set, planned
  as additional checkbox groups on `Measure Objects` and `Measure Objects +
  Intensity`;
- optional mesh export/preview and later specialist mesh metrics after a
  surface-output contract is designed for the graph;
- final table-combination path suitable for PCA and treatment-group analysis.

Colocalization and localization:

- optional validation notebook or scripted figure using the synthetic
  colocalization sample;
- deeper RACC interop decisions if VIPP and the standalone RACC plugin should
  share a numerical core.

Segmentation polish:

- marker QC improvements;
- optional Multi-Otsu class images;
- watershed default validation on representative nuclei, cell, puncta, and
  mitochondrial datasets.

Mitochondria-specific analysis:

- network fragmentation metrics;
- specialist branch/network indices tuned for mitochondrial biology beyond the
  implemented generic branch-summary distributions and domain-normalized
  connectedness metrics;
- specialist morphology inspired by the old MitoMorph system without forcing
  mitochondrial assumptions into generic object-measurement nodes.

### Batch Execution UI TODOs

- stable item identities;
- output templates;
- per-item provenance;
- multiple independently bound sources;
- explicit iteration over semantic axes.

### AI-Assisted Pipeline Authoring TODOs

Goal: let the user describe a pipeline in natural language and have an AI model
assemble a normal, inspectable VIPP graph on the canvas.

First milestone:

- generate a new workflow JSON from a text prompt and optional source context;
- apply it through the existing validating loader;
- constrain generated graphs to real operation ids, real port contracts, and
  existing connection rules.

Required model context:

- live node catalogue with operation ids, titles, categories, and descriptions;
- parameter types, ranges, defaults, and human labels;
- port contracts and compatible output/input kinds;
- connection and slot rules;
- workflow JSON schema;
- worked examples;
- `ImageState` and `TableState` contracts where relevant.

Data sharing controls:

- default to text-only prompts and generated registry/schema context;
- opt-in sharing of `ImageState`/`TableState` metadata;
- opt-in sharing of downsampled thumbnails for visual grounding;
- do not send full-resolution pixels.

Provider configuration:

- user-supplied hosted provider keys, such as OpenAI or Anthropic;
- local/self-hosted providers, such as Ollama;
- provider abstraction so the graph feature is provider-agnostic;
- keys user-managed and never serialized into workflow files.

Custom or black-box node generation:

- allow proposals when the requested step has no existing VIPP node;
- expose generated code to the user in full;
- require human review and explicit approval by default;
- leave room for sandboxed execution or trust-and-run modes as explicit user
  choices;
- decide how custom nodes serialize and round-trip.

Open questions:

- deterministic/reproducible generation suitable for publication;
- validating that generated graphs actually run before committing them;
- cost and rate-limit handling;
- conversational graph edits while keeping the canvas as source of truth.

## Later Milestones

- registration;
- deconvolution;
- model-backed segmentation;
- advanced stitching/alignment workflows.

## Planned Artifacts

| Artifact | Status |
| --- | --- |
| `workflow.json` | Implemented |
| exported `pipeline.py` | Implemented |
| example label workflow JSON | Implemented |
| measurement CSV/TSV | Implemented |
| `batch_config.yaml` | TODO |
| environment/provenance manifest | TODO |
