# napari-vipp Planning Notes

Last reviewed: 2026-06-23

For the prioritized algorithm and node catalogue, see
[node-roadmap.md](node-roadmap.md). For implementation details, see
[architecture.md](architecture.md). The proposed first-class OME import/export
architecture is in [ome-io-plan.md](ome-io-plan.md). Current user-facing format
behavior is in [io-user-guide.md](io-user-guide.md), and research evidence is
tracked in [research-and-publication.md](research-and-publication.md).
MitoMorph-derived high-dimensional feature extraction and table-combination
requirements are tracked in
[mitomorph-feature-parity.md](mitomorph-feature-parity.md).

## Current Product Direction

The graph is the primary work surface. A useful VIPP feature should form part
of a reproducible bioimage workflow, not merely add another isolated image
filter.

The target first-class workflow families are:

- nuclei and cell segmentation;
- puncta and spot analysis;
- mitochondrial object and network analysis;
- pixel-based and object-based colocalization;
- 2D images and true 3D fluorescence z-stacks.

Registration and deconvolution remain later milestones.

## Implemented Platform Capabilities

### Manual Graph Editing

The graph currently supports:

- a large pan/zoom canvas with movable node cards;
- node creation from the searchable palette;
- node and connection deletion;
- node right-click menus for deletion, code inspection, duplication, and
  contextual Pin/Unpin;
- click-to-connect and drag-to-connect wiring;
- visual compatible/incompatible drop feedback;
- cycle and port-type rejection;
- slot-aware multi-input connections;
- per-port multi-output connections;
- dynamic Split Channels output counts;
- connector updates while nodes move;
- save/load of canvas positions.

Remaining graph-editor refinements include automatic insertion when dropping a
node onto an existing wire, alignment/layout tools, graph annotations, and
larger-workflow navigation aids. The next two prioritized refinements are
described below.

#### Planned: Insert Node Between Connected Nodes

Make it easy to splice a new node into an existing connection without manually
deleting and rewiring. When a user adds or drops a node onto an existing wire,
or chooses an "Insert node here" action on a connection, the editor should:

- detect the targeted source -> target connection;
- remove that single connection;
- wire `source -> new node` and `new node -> target`, honoring port types and
  the existing source output slot and target input slot;
- reject the insertion with clear feedback when the new node's input or output
  port contracts are incompatible with the spliced connection;
- position the new node sensibly on or near the original wire;
- treat the whole splice as one undoable action.

This should work for the common single-input/single-output case first, with a
defined fallback for multi-input or multi-output nodes (for example, prompt for
or default to the first compatible port pair).

#### Planned: User-Initiated Automatic Layout Cleanup

Add an explicit, user-initiated command that repositions nodes to tidy up a
messy graph, similar to how Obsidian relaxes its knowledge graph. Requirements:

- expose a toolbar button next to "Export OME dataset..." that improves graph
  workflow formatting/positioning on demand;
- never reposition nodes automatically; layout cleanup must only run when the
  user invokes it;
- compute a readable layout from the existing connections, for example a
  layered/topological left-to-right (or top-to-bottom) arrangement that reduces
  edge crossings and overlap;
- keep the result deterministic and stable so repeated invocations on an
  unchanged graph do not reshuffle the canvas;
- treat the relayout as a single undoable action and persist the new positions
  through the existing save/load of canvas positions.

### Workflow Persistence

Portable JSON workflow persistence is implemented. Version 1 stores:

- stable node and operation ids;
- parameter values;
- source and target node ids;
- target input slots and source output slots;
- canvas positions;
- workflow type and version.

Loading derives titles, categories, and port contracts from the installed node
library. The loader accepts only the current workflow type and version and
rejects unknown operations, malformed nodes, duplicate ids, invalid positions,
and dangling or multiply occupied connections with a clear error.

Not yet stored:

- per-node thumbnail visibility;
- inspector and histogram UI state;
- graph notes or annotations;
- environment/package provenance;
- YAML format.

Image Source selections, including file paths and napari layer names, are saved
as ordinary node parameters. File paths are currently literal local paths;
workflow files do not embed input data, rebase paths, or package assets for
portable sharing.

The checked-in example
[`examples/otsu-red-channel-labels.json`](../examples/otsu-red-channel-labels.json)
demonstrates the current format.

### Python Export And Batch Execution

Python export is implemented from the same graph model used by the UI. The
generated script:

- imports pure functions from `napari_vipp.core.operations`;
- reconstructs slot-aware multi-input and multi-output routing;
- exposes `run_pipeline()`;
- exposes a folder-oriented `batch_process()` helper;
- provides a command-line entry point;
- saves terminal graph outputs.

The current exporter is headless but still requires the `napari-vipp` Python
package. It handles image-like and table outputs, but its folder batch helper
assumes one primary image source. A richer batch UI, environment lock file,
embedded provenance, multiple independently bound sources, and explicit
iteration over semantic axes remain future work.

### Data State Visibility

Every graph output carries an OME-NGFF-inspired `ImageState` alongside its
array or a `TableState` alongside table outputs. Image state includes:

- shape and dtype;
- semantic axes and axis types;
- units, scale, and origin where available;
- image/mask/labels/RGB/multichannel kind;
- value and bit-depth summaries;
- source and operation history.

Table state includes row count, column count, stable column names, column units
where known, source, measurement set, and operation history. Table outputs are
shown in the inspector, hidden from image thumbnails/histograms, and can be
saved as CSV or TSV.

OME-NGFF-like `multiscales` metadata is used when available. Plain arrays fall
back to inferred axes, and the UI identifies that inference.

Type conversion and axis subsetting are explicit graph operations. The
`Convert Dtype` node exposes rescale, clip, and preserve-cast behavior.
`Select Axis Slice` supports retaining ranges and removing one or more axes
while updating metadata. `Reorder Axes` uses a draggable axis list in the
inspector and serializes to a compact axis-order string. It transposes the
data, then reinterprets spatial axis names by output position so downstream
nodes treat the result as a rotated/reoriented volume. Physical scale and units
follow the moved data axis, while channel and time metadata stay attached to
their data.

### Label Workflow

The first object-cleanup workflow is implemented:

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

Current label support includes:

- a distinct `labels` graph type;
- napari Labels inspection/pinning for label outputs, plus image-layer pinning
  for image outputs;
- 2D-per-plane and true 3D connected-component labeling;
- face or full connectivity;
- metadata-aware, size-limited 2D/3D hole filling for masks;
- mask/label-preserving minimum-size filtering with configurable connectivity
  for mask inputs;
- minimum and optional maximum pixel/voxel-volume filtering;
- mask/label-preserving removal of objects touching a spatial boundary;
- table-driven label filtering by measured object properties;
- logarithmic, data-aware volume sliders;
- an incoming object-volume histogram with threshold markers;
- sequential relabeling;
- integer-preserving TIFF and NumPy saving;
- workflow persistence and Python export.

## Current Node Coverage

The main single-input Sharratt/VIPP operations have been ported, including
contrast, filtering, thresholding, morphology, cropping, channel extraction,
and saving.

Broader graph capabilities have also enabled:

- Split Channels and Combine Channels;
- configurable multichannel-to-RGB display;
- typed Mask Image, image arithmetic, and logical operations;
- workflow save/load and Python export;
- image and label histograms, including clearer 2D versus stack threshold
  histogram labeling;
- common raster import and 2D raster export alongside TIFF, OME-TIFF, and
  OME-Zarr;
- image/mask/label pinning as persistent napari preview layers;
- first-class label cleanup;
- touching-object separation using distance transforms, H-maxima markers,
  marker-controlled watershed, and label expansion;
- background execution for known slow image-processing graphs, with a global
  indeterminate processing indicator, per-node busy rings for the slow node
  that triggered background execution, and coalesced reruns while a long
  calculation is active;
- first-class table outputs and label-object measurements;
- generic skeletonization and skeleton-network measurement tables.

Still requiring new platform types or UI:

- spot/peak detection benefits from points outputs;
- Channel Overlap and colocalization need scalar/table result contracts and
  careful channel/mask input UI;
- slow measurements, deconvolution, and other expensive operations need a
  manual/cached execution mode with progress, stale-state display, and
  deterministic export behavior;
- batch processing needs a dedicated UI beyond exported scripts.

## Immediate Implementation Sequence

Recently completed:

- Fiji/ImageJ-style rolling-ball background correction is implemented as
  `Rolling-Ball Background` and `Subtract Background` under
  `Filtering -> Background Correction`. The estimated background can be
  inspected separately, while the subtract node provides the common direct
  correction workflow.
- Touching-object separation is implemented under
  `Segmentation -> Object Separation`: `Euclidean Distance Transform`,
  `H-Maxima Markers`, `Marker-Controlled Watershed`, and `Expand Labels`.
  These nodes use metadata-aware `Auto from axes` processing, so z-stacks
  default to true 3D while explicit `2D YX` slice-wise processing remains
  available.

The current recommended near-term order is:

1. **Grouped table summaries for analysis-ready exports**
   Add `Summarize Measurements` so merged morphology/intensity/skeleton tables
   can be grouped by metadata and summarized cleanly for PCA or treatment
   comparisons. `Select Table Columns` now handles trimming and reordering.

2. **Skeleton QC and pruning**
   Add endpoint/junction masks, branch labels, connected skeleton-component
   labels, and short-branch pruning so network table measurements can be
   visually audited.

3. **Manual/cached execution for expensive nodes**
   Add an explicit `Calculate`/`Recalculate` execution mode for nodes that are
   too expensive to recompute continuously, starting with large 3D
   rolling-ball/sliding-paraboloid background estimation and later extending to
   deconvolution, 3D mesh morphology, and heavy colocalization workflows.
   Basic background-thread execution with a global busy indicator, node-level
   busy feedback, and queued reruns is implemented for known slow live graphs;
   this future item is about explicit cached results, process-based or
   cooperative cancellation, persisted stale-state semantics, and optional true
   progress reporting where algorithms expose meaningful progress.

## Deferred TODOs From Recent Decisions

These items were deliberately deferred while implementing table outputs,
`Measure Objects`, OME/raster I/O, table merge/annotation, label cleanup, typed
masking, graph pinning, and histogram polish. Keep them visible so future
implementation work does not have to infer old discussion context:

1. **Named heterogeneous input ports**
   Implemented for graph execution, visual ports, workflow JSON, and Python
   export. Nodes can declare explicit named inputs such as `labels`, `image`,
   `mask`, and `table`, and the first user-facing example is
   `Measure Objects + Intensity`.

2. **Intensity-aware object measurements**
   Implemented as `Measure Objects + Intensity`, with separate `Labels` and
   `Intensity image` ports and per-label mean, minimum, maximum, sum, and
   standard deviation intensity measurements.

3. **Table merge and metadata annotation**
   Implemented as `Merge Tables` and `Add Metadata Columns`. The merge node
   joins table branches by shared stable identity columns such as `t_index` and
   `label_id`, or by row position when no identity key exists and row counts
   match. `Select Table Columns` now handles keep/drop/reorder workflows.
   Remaining table work is grouped summaries and richer sample metadata import
   for batch processing.

4. **Property-based label filtering**
   Implemented as `Filter Labels By Property`. It accepts named `Labels` and
   `Measurements table` inputs, filters by numeric table columns such as
   physical volume, intensity, branch count, or other table-derived properties,
   and preserves label IDs unless an explicit relabeling step is used.

5. **Skeleton/network analysis**
   Base `Skeletonize` and `Analyze Skeleton` nodes are implemented. Remaining
   work is richer branch tracing, tortuosity, explicit graph export, and
   validation on benchmark curvilinear structures. The generic analyzer already
   reports endpoint, junction, isolate, graph edge, voxel-graph edge, cycle,
   and connected-component context metrics for 2D and 3D skeletons. These nodes
   should apply beyond mitochondria to neurites, vessels, fibers, hyphae, and
   other curvilinear structures.

6. **Skeleton QC outputs**
   Add endpoint masks, junction masks, branch labels, connected
   skeleton-component labels, and short-branch pruning so users can verify how
   table metrics were produced.

7. **Touching-object separation**
   Implemented as `Euclidean Distance Transform`, `H-Maxima Markers`,
   `Marker-Controlled Watershed`, and `Expand Labels`. The watershed node uses
   named `Image / distance`, `Markers`, and `Mask` ports. All four nodes expose
   `Auto from axes`, `2D YX`, and `3D ZYX`, with z-stacks defaulting to 3D via
   metadata-aware spatial resolution.

8. **Mitochondrial-specific measurements**
   Treat the old MitoMorph code as inspiration for future specialist nodes:
   mesh/surface estimates, convexity, branch length distributions,
   domain-normalized connectivity, and network fragmentation. Do not force
   these assumptions into the generic `Measure Objects` node. The larger goal
   is broad selectable feature extraction for downstream PCA/treatment-group
   analysis; see [mitomorph-feature-parity.md](mitomorph-feature-parity.md).

9. **Colocalization and localization analysis**
   Add pixel-based and object-based colocalization/localization nodes that
   produce scalar or table outputs rather than synthetic images by default.
   Planned examples include Pearson/Manders channel metrics, object overlap or
   association tables, nearest-object distances, event localization, and
   optional mask/ROI-restricted measurements.

10. **Manual expensive-node execution and stale cached outputs**
    Some nodes should not recalculate continuously while the user drags
    sliders or edits upstream nodes. Expensive measurement families,
    colocalization/localization, skeleton graph refinements, 3D mesh
    morphology, deconvolution, and possibly large-stack background estimation
    should support an explicit `Calculate`/`Recalculate` action on the node
    card and in the inspector. While running, the node thumbnail/card should
    show progress or busy state so the UI does not feel frozen. After
    completion, the last result becomes the node output. If an upstream input
    or relevant parameter changes, the node should become visually stale while
    preserving the last valid output until recalculated. A node may instead be
    an explicit pass-through calculation checkpoint only if that behavior is
    clear in the node name, status, and exported script. The implementation
    needs an invalidation model, cancellation, batch/export semantics, and a
    clear policy for whether cached results are persisted in workflow files.

11. **Fiji/ImageJ-style background subtraction**
    Implemented as native `Rolling-Ball Background` and `Subtract Background`
    nodes under Filtering -> Background Correction. Remaining background work
    is no longer basic rolling-ball subtraction; it is validation against Fiji
    behavior on representative microscopy images, optional sliding-paraboloid
    parity if needed, and deciding whether very large 3D workflows should use
    the manual/cached slow-node execution model rather than live recomputation.

12. **Batch execution UI**
   Python export exists, but a real collection-batch UI still needs stable item
   identities, output templates, source collections, and per-item provenance.

The first OME I/O foundation is implemented:

1. shared headless reader/writer registry and normalized dataset metadata;
2. OME-TIFF, ImageJ TIFF, and conventional TIFF import/export;
3. local OME-Zarr 0.4/0.5 image import/export with lazy Dask reads;
4. OME-Zarr label-group import/export via image-plus-label analysis packages;
5. adaptive image/series selection and stored collection-binding intent.

The next platform work is:

1. generated pyramids and preview-resolution selection;
2. label colors and label-property tables in OME-Zarr;
3. collection batch execution with stable item identities and output templates;
4. plate/well/field browsing and anonymous HTTP reads;
5. operation capability declarations and memory-aware lazy materialization.

See [ome-io-plan.md](ome-io-plan.md) for the accepted decisions and status.

The next measurement-focused milestone is:

1. grouped table summaries;
2. calibrated physical variants for extended length/shape measurements;
3. 3D mesh morphology as an opt-in expensive measurement family.

The next network-analysis milestone remains skeleton endpoint/junction QC masks,
branch labels, and short-branch pruning.

Touching-object separation is implemented. Remaining segmentation refinements
are now marker QC polish, optional Multi-Otsu class images, and validation of
watershed defaults on representative nuclei/cell/object datasets.

Table outputs and basic `Measure Objects` are implemented. A dedicated `Save
Table` graph node is not yet required because selected table outputs and
exported scripts already write CSV/TSV.

## Planned Artifacts

| Artifact | Status |
| --- | --- |
| `workflow.json` | Implemented |
| exported `pipeline.py` | Implemented |
| example label workflow JSON | Implemented |
| `batch_config.yaml` | Not implemented |
| measurement CSV/TSV | Implemented |
| environment/provenance manifest | Not implemented |
