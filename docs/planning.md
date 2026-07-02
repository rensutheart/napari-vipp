# napari-vipp Planning Notes

Last reviewed: 2026-07-02

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
- [skeleton-nodes.md](skeleton-nodes.md) for skeleton node usage.
- [mitomorph-feature-parity.md](mitomorph-feature-parity.md) for
  MitoMorph-inspired feature parity.
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
  `Measure Objects`, `Measure Objects + Intensity`, `Analyze Skeleton`, and
  `Measure Skeleton Branches`;
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

### Implemented Node Catalogue

The live registry currently contains 88 nodes.

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
- **Measurements**: Measure Objects, Measure Objects + Intensity, Analyze
  Skeleton, Measure Skeleton Branches, Merge Tables, Add Metadata Columns,
  Select Table Columns, Summarize Measurements.

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

## Active TODOs

### Current Near-Term Order

1. Adopt manual/cached execution for the next expensive feature families as
   they are added, including cancellation/progress where libraries expose it.
2. Add skeleton graph export, branch-summary distributions, physical-length
   pruning units, and domain-normalized connectivity summaries.
3. Add richer object morphology and 3D mesh/surface morphology, with calibrated
   physical variants for length/shape measurements.
4. Add colocalization/localization table nodes once the measurement and graph
   platform is stable.
5. Build a real collection-batch UI on top of the existing Python export and
   source-collection foundations.

### Execution Platform TODOs

- cancellation for running background/manual work;
- percentage progress for operations that can report it;
- richer failure recovery and retry behavior;
- adoption by future colocalization/localization, mesh morphology,
  deconvolution, and very expensive background-estimation nodes;
- visible progress support for slow nodes without making the UI feel frozen.

### Graph Editor TODOs

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
- collection batch execution with stable item identities and output templates;
- multiple independently bound sources;
- plate/well/field browsing;
- anonymous HTTP reads.

### Analysis Node TODOs

Skeleton and network analysis:

- explicit graph export;
- branch-summary distributions;
- physical-length pruning units;
- domain-normalized connectivity summaries for mitochondrial and other network
  structures.

Object and mesh morphology:

- richer region/object properties beyond the current measurement set;
- selectable expensive feature groups so users choose which classes of
  measurements to calculate;
- 3D mesh/surface estimates;
- convexity and surface-derived shape metrics;
- calibrated physical variants for extended length/shape measurements;
- final table-combination path suitable for PCA and treatment-group analysis.

Colocalization and localization:

- Pearson and Manders pixel-based channel metrics;
- object overlap and association tables;
- nearest-object distance tables;
- event localization against objects, masks, or ROIs;
- mask/ROI-restricted measurements;
- scalar/table result contracts;
- channel and mask input UI designed to avoid ambiguous wiring.

Segmentation polish:

- marker QC improvements;
- optional Multi-Otsu class images;
- watershed default validation on representative nuclei, cell, puncta, and
  mitochondrial datasets.

Mitochondria-specific analysis:

- network fragmentation metrics;
- branch-summary distributions tuned for mitochondrial networks;
- domain-normalized connectivity metrics;
- specialist morphology inspired by the old MitoMorph system without forcing
  mitochondrial assumptions into generic object-measurement nodes.

### Batch Execution UI TODOs

- stable item identities;
- source collections;
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
