# napari-vipp Planning Notes

Last reviewed: 2026-06-25

This document is the consolidated source of truth for what is **implemented**
versus **planned**. It was reconciled against the live node registry
(`napari_vipp.core.pipeline.NODE_LIBRARY`, currently 81 nodes) and the widget
code, so the status labels below reflect the actual codebase rather than older
intentions.

For the prioritized algorithm and node catalogue, see
[node-roadmap.md](node-roadmap.md). For implementation details, see
[architecture.md](architecture.md). The proposed first-class OME import/export
architecture is in [ome-io-plan.md](ome-io-plan.md). Current user-facing format
behavior is in [io-user-guide.md](io-user-guide.md), and research evidence is
tracked in [research-and-publication.md](research-and-publication.md).
MitoMorph-derived high-dimensional feature extraction and table-combination
requirements are tracked in
[mitomorph-feature-parity.md](mitomorph-feature-parity.md).

## Product Direction

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

---

## Implemented

### Graph Editing

- a large pan/zoom canvas with movable node cards and auto-expanding scene;
- node creation from the searchable, grouped palette;
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
- undo/redo for graph and parameter edits;
- save/load of canvas positions.

### Workflow Persistence

Portable JSON workflow persistence (version 1) stores:

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

Image Source selections, including file paths and napari layer names, are saved
as ordinary node parameters. The checked-in example
[`examples/otsu-red-channel-labels.json`](../examples/otsu-red-channel-labels.json)
demonstrates the current format.

### Python Export And Headless Batch

Python export is generated from the same graph model used by the UI. The
generated script:

- imports pure functions from `napari_vipp.core.operations`;
- reconstructs slot-aware multi-input and multi-output routing;
- exposes `run_pipeline()`;
- exposes a folder-oriented `batch_process()` helper;
- provides a command-line entry point;
- saves terminal graph outputs.

The exporter is headless but still requires the `napari-vipp` Python package.
It handles image-like and table outputs.

### Background Execution

- background-thread execution for known slow image-processing graphs;
- a global indeterminate processing indicator;
- per-node busy rings for the slow node that triggered background execution;
- incremental dirty-node caching of prior outputs/states;
- coalesced reruns while a long calculation is active, discarding stale
  background results;
- a user-facing "Run all in BG" toggle to force all updates onto the worker.

### Data State Visibility

Every graph output carries an OME-NGFF-inspired `ImageState` alongside its
array, or a `TableState` alongside table outputs.

Image state includes shape, dtype, semantic axes and axis types, units/scale/
origin where available, image/mask/labels/RGB/multichannel kind, value and
bit-depth summaries, and source/operation history. OME-NGFF-like `multiscales`
metadata is used when available; plain arrays fall back to inferred axes, and
the UI identifies that inference.

Table state includes row count, column count, stable column names, column units
where known, source, measurement set, and operation history. Table outputs are
shown in the inspector, hidden from image thumbnails/histograms, and can be
saved as CSV or TSV.

Type conversion and axis handling are explicit graph operations: `Convert
Dtype` (rescale/clip/preserve-cast), `Select Axis Slice` (retain ranges, remove
axes), `Reorder Axes` (draggable axis list, compact axis-order string), and
`Rescale Axes`. Physical scale and units follow the moved data axis while
channel and time metadata stay attached to their data.

### Preview, Thumbnails, And Dims

- per-node thumbnails with Slice/MIP/projection modes and contrast controls;
- image and label histograms, including clearer 2D versus stack threshold
  labeling;
- image/mask/label pinning as persistent napari preview layers;
- optional "Follow napari dims" so thumbnails and histograms track the slider;
- cross-node slice mapping: nodes with the same Z length use the exact napari
  index, while nodes with a different Z length (for example through `Rescale
  Axes`) use the equivalent relative position.

### OME / Raster I/O Foundation

- shared headless reader/writer registry and normalized dataset metadata;
- OME-TIFF, ImageJ TIFF, and conventional TIFF import/export;
- common raster import and 2D raster export;
- local OME-Zarr 0.4/0.5 image import/export with lazy Dask reads;
- OME-Zarr label-group import/export via image-plus-label analysis packages;
- adaptive image/series selection and stored collection-binding intent.

See [ome-io-plan.md](ome-io-plan.md) for accepted decisions and status.

### Implemented Node Catalogue (81 nodes)

Counts and names below match the live registry.

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
  Morphological Gradient, Fill Holes, Remove Small Objects, Skeletonize.
- **Label Operations**: Label Connected Components, Clear Border Objects, Filter
  Labels By Volume, Filter Labels By Property, Relabel Sequential.
- **Measurements**: Measure Objects, Measure Objects + Intensity, Analyze
  Skeleton, Merge Tables, Add Metadata Columns, Select Table Columns.

A reference label-cleanup workflow is implemented end to end:

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

---

## Planned / TODO

Items are grouped by area. None of the following are implemented yet unless
explicitly noted as partial.

### Graph Editor Usability

**Insert Node Between Connected Nodes.** Splice a new node into an existing
connection without manual delete-and-rewire. When a node is dropped onto a wire,
or an "Insert node here" action is chosen on a connection, the editor should:
detect the targeted source -> target connection; remove that single connection;
wire `source -> new node` and `new node -> target` honoring port types and the
existing output/input slots; reject incompatible splices with clear feedback;
position the new node on or near the original wire; and treat the splice as one
undoable action. Target the single-input/single-output case first, with a
defined fallback for multi-port nodes.

**User-Initiated Automatic Layout Cleanup.** An explicit command that tidies a
messy graph (similar to Obsidian relaxing its knowledge graph). Requirements:
expose a toolbar button next to "Export OME dataset..." that improves graph
formatting/positioning on demand; never reposition automatically; compute a
readable layered/topological layout that reduces edge crossings and overlap;
keep results deterministic and stable across repeated invocations; treat the
relayout as one undoable action persisted through the existing canvas-position
save/load.

**Other refinements.** Graph annotations/notes, alignment guides, and
larger-workflow navigation aids.

### AI-Assisted Pipeline Authoring

Let the user describe a pipeline in natural language and have an AI model
assemble it on the canvas from the existing nodes, so the result is a normal,
fully interactive VIPP graph whose steps can be inspected, reparametrized, and
swapped like any hand-built workflow. The user can optionally attach context
(for example the already-added Image Source nodes or sample images) and then
ask for either a brand-new pipeline or a modification of an existing one. This
is a large, forward-looking direction; nothing here is implemented yet.

**First milestone: generate a new pipeline from a description.** Given a text
prompt plus optional source context, the model returns a workflow that lands on
the canvas. Modification of an existing graph (described changes, added
branches) is a follow-on milestone built on the same machinery.

**Output via the validating loader.** The model emits a workflow JSON in the
existing version-1 format and it is applied through the current validating
loader rather than mutating the graph model directly. This constrains the model
to real operation ids, real port contracts, and the existing connection rules
(cycle/port-type rejection, slot occupancy), and any invalid graph is rejected
with the same clear errors as a hand-edited file. The model's job is reduced to
"produce a valid `workflow.json` using the declared node catalogue."

**Context priming.** Useful generation requires giving the model substantial
grounding before it can produce anything valid: the full node catalogue with
operation ids, titles, categories, and human descriptions; each node's
parameters with types/ranges/defaults; the port contracts and which output
kinds (image/mask/labels/table) connect to which inputs; the connection and
slot rules; the workflow JSON schema and a few worked examples; and, where
helpful, the `ImageState`/`TableState` contract so the model can reason about
axes, kinds, and table shapes. Most of this can be generated directly from the
live registry so the prompt context stays in sync with the codebase.

**Data sharing controls (privacy).** By default only text is sent: the prompt,
the generated node-catalogue context, and the workflow schema. Sending anything
derived from the user's images is strictly opt-in and per-request under the
user's control, with two escalating levels: `ImageState`/`TableState` metadata
(shape, dtype, axes, value summaries) for the active sources, and downsampled
thumbnails of sample images for stronger visual grounding. Full-resolution
pixels are not sent.

**Provider configuration.** Support both user-supplied API keys for hosted
providers (e.g. OpenAI, Anthropic) and local/self-hosted models (e.g. Ollama),
behind a small provider abstraction so the rest of the feature is
provider-agnostic. Keys are user-managed and never persisted into workflow
files.

**Black-box / custom nodes for unsupported steps.** When a requested step has
no matching existing node, the model may propose a custom node that wraps the
missing functionality and still behaves like a normal node on the graph
(typed ports, parameters, inspectable, swappable), acting as a small black box
in an otherwise standard pipeline. Because such a node implies AI-generated code
running locally against the user's data, the execution policy is **user-
configurable**, with options spanning: require human review and explicit
approval of the generated code before it runs (default); run it in a
restricted/sandboxed environment; or trust-and-run automatically. Custom-node
generation is a separate, gated mechanism from ordinary catalogue-only
generation, and any code surfaced to the user is shown in full.

**Open questions.** How custom nodes serialize and round-trip in `workflow.json`
(and whether their code is embedded, referenced, or regenerated); how to make
generation deterministic/reproducible enough for publication; how to validate
that a generated graph actually runs before committing it to the canvas; cost
and rate-limit handling; and how to let the user iterate conversationally
("now add a colocalization branch") while keeping the canvas as the source of
truth.

### Workflow Persistence Gaps

Not yet stored: per-node thumbnail visibility; inspector and histogram UI state;
graph notes/annotations; environment/package provenance; YAML format. Workflow
files do not embed input data, rebase paths, or package assets for portable
sharing.

### Preview / Dims Usability

Add VIPP-local Z/T/C sliders to the workflow UI and keep them synchronized with
napari's dims state. The main reason is 3D viewing: when napari is in 3D mode,
the usual Z slider is no longer directly available, which makes it awkward to
choose the thumbnail slice or inspect a specific plane from within VIPP.
Requirements: expose whichever semantic axes are present (at least Z, T, and C
where applicable); keep them bidirectionally synced with napari's own dims so a
change in either place updates the other; preserve the existing "Follow napari
dims" behavior for thumbnails and histograms; and make sure thumbnail slice
selection remains easy even when napari's native slider UI is hidden by 3D
view.

### Calibration / Napari Display Regression

Fix the current regression where `Set Pixel Size / Units` no longer changes the
apparent Z versus X/Y scaling in napari's 3D viewer (for example setting Z step
size to 10 while leaving X/Y at 1 has no visible effect). This should be the
next debugging task. Best current guess: the metadata pipeline still updates
`ImageState.axes` correctly for both `Set Pixel Size / Units` and `Rescale
Axes`, but the final preview-layer handoff into napari is only attaching that
information as layer metadata instead of applying it to the actual napari layer
`scale` property. If so, the calibration exists in VIPP's carried metadata and
inspector/history, but never reaches the viewer transform that controls 3D
display spacing. A secondary possibility is that `Rescale Axes` and
`Set Pixel Size / Units` interact correctly in `transform_image_state(...)` for
some paths but a later layer-update/replacement path drops the transformed axis
scales when refreshing generated layers.

### Table Analysis

- **Grouped table summaries** (`Summarize Measurements`): group merged
  morphology/intensity/skeleton tables by metadata and summarize for PCA or
  treatment comparison. (`Merge Tables`, `Add Metadata Columns`, and `Select
  Table Columns` already exist; the summary/group-by step does not.)
- **Calibrated physical variants** for extended length/shape measurements.
- A dedicated `Save Table` node is intentionally deferred: selected table
  outputs and exported scripts already write CSV/TSV.

### Skeleton / Network QC

`Skeletonize` and `Analyze Skeleton` are implemented (endpoint, junction,
isolate, graph-edge, cycle, and connected-component metrics for 2D/3D). Still
TODO: endpoint masks, junction masks, branch labels, connected
skeleton-component label images, and short-branch pruning so users can visually
audit how table metrics were produced; plus richer branch tracing, tortuosity,
and explicit graph export.

### Colocalization And Localization

Pixel-based and object-based colocalization/localization nodes producing scalar
or table outputs (not synthetic images by default): Pearson/Manders channel
metrics, object overlap/association tables, nearest-object distances, event
localization, and optional mask/ROI-restricted measurements. Needs scalar/table
result contracts and careful channel/mask input UI.

### Segmentation Polish

Marker QC polish, optional Multi-Otsu class images, and validation of watershed
defaults on representative nuclei/cell/object datasets.

### Manual / Cached Execution For Expensive Nodes

The current background execution is automatic and live (incremental cache plus
coalesced reruns). It is **not** the planned manual mode. Expensive families
(heavy measurements, colocalization, skeleton graph refinements, 3D mesh
morphology, deconvolution, large-stack background estimation) should support an
explicit `Calculate`/`Recalculate` action on the node card and inspector, with:
busy/progress feedback while running; the last result becoming the node output;
visual stale state when an upstream input or relevant parameter changes while
preserving the last valid output; and a defined invalidation model,
cancellation, batch/export semantics, and persistence policy for cached results.

### Batch Execution UI

Python export exists, but a real collection-batch UI still needs stable item
identities, output templates, source collections, per-item provenance, multiple
independently bound sources, and explicit iteration over semantic axes.

### OME I/O Next

1. generated pyramids and preview-resolution selection;
2. label colors and label-property tables in OME-Zarr;
3. collection batch execution with stable item identities and output templates;
4. plate/well/field browsing and anonymous HTTP reads;
5. operation capability declarations and memory-aware lazy materialization.

### Mitochondria-Specific Measurements

Treat the old MitoMorph code as inspiration for future specialist nodes:
mesh/surface estimates, convexity, branch-length distributions,
domain-normalized connectivity, and network fragmentation, without forcing
these assumptions into the generic `Measure Objects` node. The larger goal is
broad selectable feature extraction for downstream PCA/treatment-group analysis;
see [mitomorph-feature-parity.md](mitomorph-feature-parity.md).

### Later Milestones

Registration and deconvolution.

---

## Near-Term Order

1. Fix the `Set Pixel Size / Units` / napari 3D calibration-display regression.
2. Grouped table summaries (`Summarize Measurements`).
3. Skeleton QC masks, branch labels, and short-branch pruning.
4. Manual/cached `Calculate`/`Recalculate` execution for expensive nodes.
5. Graph editor usability: insert-node-on-wire and user-initiated layout cleanup.

---

## Planned Artifacts

| Artifact | Status |
| --- | --- |
| `workflow.json` | Implemented |
| exported `pipeline.py` | Implemented |
| example label workflow JSON | Implemented |
| measurement CSV/TSV | Implemented |
| `batch_config.yaml` | Not implemented |
| environment/provenance manifest | Not implemented |
