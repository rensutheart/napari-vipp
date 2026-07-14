# napari-vipp Architecture Reference

This document is a developer handoff map for the current `napari-vipp`
prototype.

Last reviewed: 2026-07-13

It reflects the live codebase through current 0.12 development, including
restoration, optional microscope-reader routing, reproducible collection batch
execution, and graph restore hardening.

For product framing and longer-range ideas, see [README.md](../README.md) and
[planning.md](planning.md). The accepted OME I/O architecture is documented in
[ome-io-plan.md](ome-io-plan.md), current user behavior in
[io-user-guide.md](io-user-guide.md), and research evidence in
[research-and-publication.md](research-and-publication.md). MitoMorph-derived
measurement parity and table-combination requirements are tracked in
[mitomorph-feature-parity.md](mitomorph-feature-parity.md).

## Current Shape

`napari-vipp` is a napari `npe2` plugin that embeds a node-graph image
processing workflow inside a dock widget. The graph is the main work surface:
users add processing nodes, wire outputs to inputs, tune parameters in the
inspector, view live thumbnails, inspect full-resolution outputs in napari, and
pin image, mask, or label outputs as persistent napari preview layers.

The implementation separates a Qt-free scientific core from reusable UI
components and one napari composition root:

```text
napari / npe2 host
  _widget.py                   VippWidget composition root and compatibility facade
    _graph.py                  node-canvas presentation and signals
    ui/                        controls, dialogs, plots, adapters, controllers
      source_adapter.py        revisioned live-layer snapshots
      file_sources.py          Qt scheduling for verified file snapshots
      workers.py               Qt adapter for headless execution
      diagnostic_workers.py    typed Qt adapters for UI diagnostics
      batch_controller.py      batch application/controller boundary

Qt-free scientific core
  core/pipeline.py             graph model, node library, executor
  core/operations.py           NumPy/scikit-image/scipy operation functions
  core/metadata.py             ImageState/TableState propagation
  core/grid.py                 physical-grid compatibility contracts
  core/source_identity.py      exact local file/store identities
  core/file_sources.py         verified, owned file-source snapshots
  core/diagnostics.py          exact all-value diagnostic reductions
  core/snapshots.py            typed graph/workflow runtime snapshots
  core/execution.py            typed headless execution service
  core/workflow.py             validated, atomic workflow persistence
  core/batch*.py               setup, deterministic planning, execution, provenance
  core/io/                     normalized image readers and writers
```

Boundary rule: files under `core/` should stay free of `qtpy` and `napari`
imports. Modules under `ui/` may depend on Qt, but must not import `_widget.py`.
`_widget.py` may import both layers because it is the `npe2` composition root.
These rules are enforced by `test_architecture.py`.

## Repository Map

```text
src/napari_vipp/
  __init__.py          package version
  napari.yaml          npe2 widget and sample-data manifest
  _widget.py           npe2 composition facade and remaining orchestration
  _graph.py            QGraphicsView node canvas, ports, wires, node cards
  _theme.py            category colour tokens
  _sample_data.py      synthetic ZYX, CZYX, TCZYX, TYX, and skeleton samples
  core/
    pipeline.py        graph model, NODE_LIBRARY, executor
    operations.py      pure processing kernels
    metadata.py        ImageState and AxisMetadata propagation
    grid.py            aligned-image and image/PSF physical-grid validation
    source_identity.py exact identities for local files and directory stores
    file_sources.py    verified, owned, read-only file-source snapshots
    diagnostics.py     exact statistics, percentiles, histograms, and label sizes
    snapshots.py       defensively copied graph and workflow snapshots
    execution.py       PipelineRunRequest/Result and headless execution service
    atomic_io.py       atomic UTF-8 JSON/text artifact replacement
    io/
      model.py         ImageDataset, SourceInspection, and series records
      registry.py      format detection and shared read/write entry points
      tiff.py          OME-TIFF, ImageJ TIFF, and conventional TIFF
      raster.py        common PNG/JPEG/BMP/GIF/WebP/TGA/PNM import and 2D export
      ome_zarr.py      OME-Zarr 0.4/0.5 image support
      numpy_io.py      NPY/NPZ support
    tables.py          TableData, TableState, CSV/TSV writer
    batch.py           collection config, planning, execution, and manifests
    batch_setup.py     headless config construction from one workflow snapshot
    batch_demo.py      deterministic paired collection and ground-truth validator
    preview.py         slice/MIP/RGB thumbnail generation
    workflow.py        schema validation and atomic workflow JSON persistence
    export.py          headless Python export
  ui/
    controls.py        reusable generic and image-source parameter controls
    axis_controls.py   semantic slicing, reordering, and table-column controls
    view_dims.py       VIPP-local semantic dimension navigation
    palette.py         node palette widget
    dialogs.py         connection, example, and tunnel dialogs
    plots.py           histogram/scatter presentation widgets
    diagnostic_workers.py typed requests/results and Qt diagnostic runnables
    source_adapter.py  owned, revisioned live napari source boundary
    file_sources.py    Qt worker for verified local file snapshots
    workers.py         thin Qt adapter for core execution
    batch.py           retained collection batch workspace and UI data contracts
    batch_controller.py batch setup/save/load/preview coordination
    batch_navigator.py persistent representative navigation and run progress
    history.py         Qt-free undo/redo session state
    lifecycle.py       external signal and worker shutdown ownership
    examples.py        example workflow registry and resource lookup
  _tests/              pytest and pytest-qt tests

scripts/launch_vipp_sample.py
scripts/launch_vipp_label_workflow.py
examples/otsu-red-channel-labels.json
docs/planning.md
docs/architecture.md
docs/node-roadmap.md
```

## Scientific Integrity Boundaries

The following are architectural contracts, not optional implementation details.
They prevent UI convenience, asynchronous timing, or storage behavior from
quietly changing a scientific result.

| Concern | Owning boundary | Implemented contract |
| --- | --- | --- |
| Local file/store revisions | `core/source_identity.py`, `core/file_sources.py`, `ui/file_sources.py` | VIPP hashes every regular-file path and byte in a file or directory store before inspection/materialization and verifies the same identity afterward. The selected series is copied to an owned, read-only NumPy array. A path-and-series revision stays pinned until explicit `Refresh`; stale in-flight loads cannot repopulate the cache. |
| Live napari revisions | `ui/source_adapter.py`, `_widget.py` | In-memory NumPy layer data and metadata are detached on the GUI thread and tagged with a revision token. Relevant layer events invalidate the token; a background result from an older revision is discarded. Live data that cannot be detached, including lazy arrays, is rejected with an instruction to materialize it or use an immutable file source. Non-axis-aligned napari transforms are rejected rather than ignored. |
| Axis semantics | `core/metadata.py`, `core/pipeline.py` | Every axis carries `explicit` or `shape-inferred` confidence, with `mixed` available at the image-state level. Operations that need semantic auto-selection of spatial rank, channel axis, projection axes, or PSF parameters reject inferred-only axes with `AmbiguousAxisError`; callers must supply explicit metadata or an explicit supported mode/index. Positional kernels also reject explicit noncanonical layouts instead of treating a `ZX` suffix as `YX`; semantic-capable crop, projection, rescaling, and measurement paths resolve named axes directly. Array shape alone never establishes RGB or Z/Y/X meaning. Malformed declared axes, stale carried shapes, and non-finite or non-positive calibration fail instead of being replaced by inferred/default metadata. |
| Graph and execution state | `core/snapshots.py`, `core/workflow.py`, `core/execution.py`, `ui/workers.py` | `GraphSnapshot` and `WorkflowSnapshot` defensively copy persistable state and validate graph materialization. Background work crosses a typed `PipelineRunRequest`/`PipelineRunResult` boundary; the service deep-copies and validates the workflow before execution. The Qt worker only forwards progress and the typed result. Source ownership remains an explicit upstream responsibility. |
| Physical grids | `core/grid.py`, `core/pipeline.py` | Registered multi-image operations compare axis semantics, sizes, scale, unit, and origin for same-shaped inputs. A lower-rank mask is broadcast only through a unique explicit semantic/calibration mapping; coincident dimension sizes are never used to guess omitted axes. Deconvolution separately requires image/PSF axis semantics and sampling to agree while allowing a different PSF extent. Unit aliases are normalized for comparison. VIPP never resamples, reorders, or registers an input implicitly to make grids agree. |
| Diagnostic calculations | `core/diagnostics.py` | Finite statistics, percentiles, histograms, generated-layer extrema, and label-volume summaries use the complete declared population. Chunking bounds temporary memory but is not sampling. Wide integer histogram placement avoids lossy float conversion, and multichannel behavior requires an explicit `channel_axis` rather than a trailing-size RGB guess. |
| Scientific parameters and inputs | `core/operations.py`, operation tests | Invalid, ambiguous, unordered, non-finite, or incomplete parameters are rejected where silently clamping, swapping, defaulting, or dropping values would change the requested method. Dynamic choices have persisted grammars rather than accepting arbitrary non-empty text. RGB/luminance behavior requires an explicit channel declaration. Representative operation tests use read-only inputs and verify that upstream buffers are not mutated. |
| Viewer presentation | `_widget.py`, `core/diagnostics.py` | Inspect and pin layers receive detached array copies, so napari edits cannot mutate pipeline caches. Large generated layers may start with provisional dtype-based contrast solely to keep Qt responsive, followed by exact finite extrema in a stale-safe worker. Viewer contrast, colormaps, and thumbnails never enter operation inputs. |
| Workflow and provenance artifacts | `core/atomic_io.py`, `core/workflow.py`, `core/batch.py` | Each JSON/text artifact is written to a same-directory temporary file, flushed and fsynced, then atomically replaced. Non-finite JSON is rejected. Each file is atomic; a set of related workflow/config/manifest files is not a single filesystem transaction. |
| Headless Python export | `core/export.py`, `core/pipeline.py` | Generated programs embed a validated immutable workflow and reconstruct the shared executor for every call. They carry `ImageState` from `ImageDataset`/`SourcePayload`, preserve output states when saving, reject missing or ambiguous source bindings, and refuse a different VIPP runtime version. Export never recreates metadata-dependent behavior with incomplete direct operation calls. |
| Batch publication | `core/batch.py` | Each item captures every source identity before reading, fully writes available outputs to private staged files, reverifies every source, and only then promotes staged files to final paths. Promotions are atomic per output. A changed source fails the item and removes its staged files, so no apparently complete result is published from a mixed revision. Multi-output promotion can still end in an explicitly recorded partial item if a later promotion fails. |

When a new feature crosses one of these boundaries, extend the focused contract
tests before adding UI affordances. The scientific behavior checklist in
[CONTRIBUTING.md](../CONTRIBUTING.md#scientific-behavior-requirements) is the
review gate for changes to these rules.

## Core Model

`core/pipeline.py` is the centre of the non-UI system.

Important dataclasses:

| Type | Purpose |
| --- | --- |
| `ParameterSpec` | Declares one parameter: name, label, kind, default, range, step, decimals, optional choices, and optional effect-oriented tooltip guidance. |
| `InputSpec` | Declares one named input port of a heterogeneous node: `name`, `input_type`, and optional display `title`. |
| `OutputSpec` | Declares one output port of a node: `name`, `output_type`, and optional display `title`. |
| `OperationSpec` | Declares a node type in `NODE_LIBRARY`: id, title, category, subcategory, input/output types, parameters, function, `max_inputs`, optional named `inputs`, optional static `outputs`, optional dynamic `output_factory`, and whether outputs preserve the connected input type. |
| `GraphNode` | A node instance with a stable id, operation id, resolved display/type fields, mutable `params`, and `max_inputs`. |
| `GraphConnection` | A directed edge from `source_id` to `target_id`, including the target input slot as `target_port`, the source output slot as `source_port`, and optional `tunnel_name` metadata for hidden named-wire connections. |
| `OutputTunnel` | A named source bound to one output port. Input ports can subscribe to it through a tunnel-marked `GraphConnection`. |
| `ConnectionResult` | Returned by `connect()`: success flag, message, created connection, and any replaced/removed connections. |
| `SourcePayload` | Data, optional layer metadata, name, and optional reader-built `ImageState` injected into source nodes. |
| `TableData` / `TableState` | Non-image output data and metadata used by measurement nodes and CSV/TSV export. |

The executable graph is `PrototypePipeline`.

Key behaviours:

- `reset_starter_graph()` creates the starter graph:
  `Image Source -> Gaussian Blur -> Otsu Threshold`.
- `add_node()` creates node instances from `NODE_LIBRARY`.
- `connect(source_id, target_id, target_port=None, source_port=0)` enforces type
  compatibility (using the chosen source output port's type and the specific
  target input port type), rejects cycles, respects `max_inputs`, and stores the
  selected target and source slots. For multi-input nodes, omitted
  `target_port` auto-fills the first free compatible slot; connecting to an
  occupied slot replaces that slot only. `source_port` selects
  which output of a multi-output node feeds the edge.
- `add_output_tunnel(name, source_id, source_port)` labels one output port as a
  reusable named source. `connect_to_tunnel(name, target_id, target_port)`
  creates a normal typed/cycle-checked connection marked with that tunnel name.
  Tunnel connections execute, serialize, and export like ordinary connections,
  but the graph view hides the long wire and shows compact badges at both ports.
- `disconnect(source_id, target_id, target_port=None)` removes either a specific
  slot connection or all matching source-target connections when no slot is
  supplied.
- `trim_invalid_connections(node_id)` removes connections whose stored slot is
  now outside the node's current input count.
- `output_ports(node_id)` returns the node's resolved `OutputSpec`s: static
  `outputs` when declared, one default `out` port otherwise, or a dynamic count
  inferred from the connected source, the last run, or a persisted private
  count hint. Type-preserving multi-output nodes such as `split_channels`
  resolve each port's type from the connected upstream port, allowing a split
  mask channel to feed mask-only operations.
- `trim_invalid_output_connections(node_id)` removes downstream edges whose
  stored `source_port` is now outside a dynamic node's current output count.
- `topological_order()` supports export by returning a source-first order.
- `restore_graph()` validates a complete candidate graph before replacing the
  live model. It rejects unknown operations, duplicate ids, missing nodes,
  cycles, negative/out-of-range ports, multiple connections to one input, and
  invalid or duplicate tunnel sources. Failed restoration is atomic: the
  previous graph remains unchanged.

`core/snapshots.py` provides the typed, detached representation used when graph
state must outlive the editor mutation that captured it. `NodeSnapshot`
deep-copies parameters, `GraphSnapshot` owns ordered nodes/connections/tunnels,
and `WorkflowSnapshot` adds ordered positions, notes, and UI metadata. Restoring
a snapshot always passes through `PrototypePipeline.restore_graph()`; snapshot
construction is not a bypass around graph validation. Interactive undo/redo in
`ui/history.py` stores these workflow snapshots rather than references to live
node dictionaries.

Execution happens in `run(...)`. It repeatedly runs nodes whose upstream sources
are complete. For each node it stores the primary port-0 output in
`outputs[node_id]` and `output_states[node_id]` for convenient single-output
access, plus the full per-port lists in `node_outputs[node_id]` and
`node_output_states[node_id]`. Downstream inputs are resolved by
`(source_id, source_port)` so a node can pull from any output port of its source.
Source nodes use `SourcePayload`s. File sources are loaded through
`core.io.read_image()` inside the verified `core.file_sources` boundary. The
interactive boundary captures and verifies an exact whole-file or
directory-tree identity around inspection and materialization, copies the
selected series into an owned read-only NumPy array, and pins that
path-and-series snapshot until explicit Refresh. OME-Zarr, microscope, and
large-file materialization uses the background queue. A refresh invalidates the
file-load generation as well as the cache, so an older in-flight result cannot
repopulate a stale snapshot. The normalized reader-built state is preserved and
completed against the detached array rather than reparsed from lossy display
metadata.

Live napari sources enter through `LiveLayerSourceAdapter`. An in-memory NumPy
layer is copied, marked read-only, paired with detached display metadata and a
revision token, and checked again before a background result is accepted. VIPP
does not pass a live lazy array through this route: if it cannot make an owned
revision, calculation stops with an explicit error. Axis-aligned layer scale,
translation, and units are carried into `ImageState`; rotation, shear, and
additional affine transforms are rejected until the user explicitly resamples
or registers the data.
Single-input nodes call their pure operation function with one input. Multi-input
nodes gather inputs in `target_port` order and require all ports from
`0..input_count-1` to be connected before running.

The widget can dispatch eligible graph runs to a single background worker
thread. It captures a workflow document, stable source payloads, cache state,
dirty/manual-node policy, source revision tokens, and cancellation state in a
`PipelineRunRequest`. `core.execution.execute_pipeline_request()` deep-copies
and validates the workflow, materializes a detached `PrototypePipeline`, runs
it, and returns a `PipelineRunResult`; `ui.workers.PipelineRunWorker` is only the
Qt signal adapter. The widget accepts the result only if its run id, workflow,
and live-source revisions still match. Small synchronous runs currently call
`PrototypePipeline.run()` directly, which is a remaining unification seam.

The active worker reports node-start events for the global busy indicator and
per-node processing state. Operations that accept a `ProgressContext` can also
report determinate progress and check a worker-owned cancel event between
internal work units. This is currently wired for
rolling-ball/subtract-background block processing, `rescale_axes`, 3D mesh
morphology label loops, and Minimum Threshold's histogram-smoothing loop. Auto
Watershed From Mask is also treated as known-slow even below the large-image
cutoff. The toolbar `Cancel` control clears queued reruns, requeues in-flight
dirty nodes, requests cooperative cancellation, and ignores stale worker
results. It still cannot forcibly interrupt a NumPy/SciPy/scikit-image call
already executing inside that worker.

Dispatch is automatic for the known-slow operation allow-list and whenever an
affected cached input, output, or source payload is at least 32 MiB or four
million values. `Run all in BG` remains a force-all override; leaving it off
does not mean that large work runs on Qt's UI thread. Large selected-node input
histograms and automatic-threshold markers share the worker queue, coalesce
repeated requests, reject stale results, and cache exact summaries. Operational
histograms are accumulated in bounded chunks rather than through a full-stack
finite-value and float64 copy. Chunking limits peak temporary memory; it never
samples or drops finite pixels. Li remains on its raw-value iterative algorithm
and relies on background dispatch for large inputs rather than substituting a
binned approximation.

Special execution cases:

- `save_output` receives the upstream `image_state` so TIFF/ImageJ metadata can
  be written where possible.
- `richardson_lucy_tv_deconvolution` uses fixed, practical slider windows while
  retaining wider valid spinner-entry ranges. Its positive logarithmic bounds
  use geometric interpolation; zero/off values remain spinner-only and do not
  expand the slider. Tooltips from its `ParameterSpec` values are applied to the
  form label and every interactive child of the parameter control.
- `filter_labels_by_volume` parameter controls derive their slider extent from
  the largest incoming object under the resolved 2D/3D spatial mode. The
  logarithmic slider remains practical for wide volume ranges while the spin
  box retains the operation's full hard limit for exact entry. Its inspector
  adds an object-volume distribution computed from the unfiltered label input,
  with live minimum and optional maximum threshold markers. The volume-axis
  logarithmic toggle defaults on and can switch the bins and markers to linear.
- `combine_channels` derives its insertion channel axis from upstream metadata,
  stores that `channel_axis` back into `node.params`, and then stacks inputs in
  slot order.
- `mask_image` is a named two-input node. Port 0 is the image whose metadata
  is carried forward, and port 1 accepts a binary mask or label image. The
  operation broadcasts compatible spatial masks over leading axes and
  channel-last RGB/RGBA axes.
- `measure_objects` receives upstream axis names/types/scales/units when an
  `ImageState` is available and returns `TableData`; the executor builds a
  `TableState` instead of trying to infer image metadata.
- Multi-output nodes call their function once; it returns a sequence of arrays
  that `run` splits across the node's ports, building a per-port `ImageState`
  via `transform_split_output_state`. Ports are either static (declared in
  `OperationSpec.outputs`) or dynamic (built by `OperationSpec.output_factory`
  from the number of returned arrays, e.g. `split_channels` yields one port per
  channel in the image). Graph `PortItem`s carry their own resolved data types,
  so drag compatibility can differ from a multi-output node's fallback type.
- `split_channels` requires a semantic channel axis from metadata or a
  conventional RGB/RGBA channel-last input. `split_axis` is the dynamic-output
  node for arbitrary stack axes such as time, Z, or a leading non-channel axis.

## Node Library

`NODE_LIBRARY` in `core/pipeline.py` is the source of truth for available nodes.
The palette grouping comes from each `OperationSpec.category` and
`OperationSpec.subcategory`.

The current high-level groups are:

- `Image Data`
  - `Source & Output`: Image Source, Save Image, Batch Output
  - `Axes & Regions`: Crop Stack, Select Axis Slice, Split Axis, Reorder Axes
  - `Channels & Composites`: Extract Channel, Combine Channels, Split Channels,
    Composite → RGB
  - `Utilities`: Convert Dtype
  - `Math & Logic`: Calculate New Image, Add, Subtract, Ratio, Mask Image,
    Logical AND, Logical OR, Logical XOR, Invert
- `Intensity & Contrast`: Linear Scale + Offset, Gamma Correction, Rescale
  Intensity, Normalize, Clip
- `Filtering`
  - `Smoothing & Denoising`: Average Blur, Gaussian Blur, Gaussian Blur 3D,
    Median Filter, Bilateral Filtering, Non-Local Means
  - `Edge & Detail`: Difference of Gaussians, Unsharp Mask, Sobel Edges,
    Canny Edges, Laplace Filter
- `Projection`: Maximum Projection, Project Image, Orthogonal Projection
  - `Project Image` presents a contextual dropdown built from the current input
    axes, including automatic Z projection, explicit axis choices, and an
    all-non-YX-spatial option for stack-style reductions.
- `Segmentation`
  - `Global Thresholds`: Otsu, Triangle, Li, Yen, Isodata, Minimum, Binary,
    Hysteresis thresholding
  - `Local Thresholds`: Adaptive Mean, Adaptive Gaussian, Sauvola, Niblack
    thresholding

Automatic global threshold nodes default to
`Threshold uses = Stack histogram`, meaning one cutoff is computed from the
whole grayscale input and applied to the full image. On stack inputs the
inspector exposes `Slice histogram` for per-plane cutoff calculation; that still
produces a full-stack mask, it only changes which histogram is used to compute
each cutoff. On 2D inputs the control is hidden because stack versus slice is
not meaningful. Fixed `Binary Threshold` and local threshold nodes do not expose
this control. Global automatic threshold nodes also show the selected input
histogram with a marker at the computed cutoff.

Otsu, Triangle, Yen, Isodata, and Minimum count every finite input value, with
no data-size-dependent sampling or silent rebinning. Their bin contract is
dtype-aware:

- boolean inputs bypass automatic fitting and are copied unchanged because
  they are already binary segmentations; the inspector's 0.5 marker is a
  display convention rather than a fitted value;
- integer inputs use one bin for every native integer level from the observed
  finite minimum through maximum; `Float histogram bins` is ignored;
- integer spans above 65,536 levels raise an actionable error asking the user to
  convert or rescale instead of silently quantizing the data;
- floating-point inputs use the node's explicit `Float histogram bins`
  (`histogram_bins`) setting, from 2 through 65,536 and defaulting to 256,
  across the observed finite minimum/maximum range.

NaN, positive infinity, and negative infinity do not participate in fitting and
are background in the resulting mask. Li is the exception to the bin rules: its
minimum-cross-entropy iteration uses every finite raw value and has no histogram
bin setting. Integer Li inputs are translated by their exact native minimum
before float64 iteration; relative spans above 2^53 fail explicitly rather than
collapsing levels. Empty/all-nonfinite inputs fail instead of receiving a
fabricated zero threshold. Local thresholds operate plane-wise and may
inherently allocate a local-threshold plane; large runs still use the automatic
background policy.

Minimum exposes a saved `max_iterations` limit (1..10,000, default 10,000) for
histogram smoothing. It reports an explicit failure when the method cannot
resolve two maxima; no mean or alternate-threshold fallback is applied.

`Hysteresis Threshold` uses raw low/high intensity thresholds and displays those
markers on its input histogram. Its spatial processing mode controls whether
connectivity is evaluated per `YX` plane or over a full `ZYX` volume. It lives
with the other thresholding nodes because it is a double-threshold mask
operation. Canny Edges uses quantile thresholds by default, returns an edge mask,
and lives with `Filtering > Edge & Detail` alongside Sobel/Laplace-style edge
operations.

Any node that intentionally processes a stack slice-wise must set
`OperationSpec.stack_processing_note`. The inspector shows that note only when
the connected input has a stack-like spatial axis. This is the general UI rule
for present and future XY-only filters, local thresholds, edge detectors, and
morphology operations: users must be told that the node works on semantic `YX`
planes. With explicit metadata, positional kernels require a canonical `YX`
suffix (or `ZYX` suffix for 3D) and fail clearly otherwise. `Reorder Axes` can
restore canonical storage order, but does not relabel `ZX` as `YX`; such a
reinterpretation would need its own explicit, provenance-recorded operation.
- `Morphology`: Dilation, Erosion, Opening, Closing, Top Hat, Black Hat,
  Morphological Gradient, Fill Holes, Remove Small Objects, Skeletonize,
  Skeleton Keypoints, Skeleton Graph Overlay, Prune Skeleton Branches
- `Label Operations`: Label Connected Components, Filter Labels By Volume,
  Filter Labels By Property, Clear Border Objects, Relabel Sequential, Label
  Skeleton Components, Label Skeleton Branches
- `Measurements`: Measure Objects, Measure Objects + Intensity, Measure 3D
  Mesh Morphology, Analyze Skeleton, Measure Skeleton Branches, Summarize
  Skeleton Branches, Skeleton Graph Tables, Measure Overall Skeleton Network,
  Merge Tables, Select Table Columns, Add Metadata Columns, Summarize
  Measurements

`labels` is a first-class graph type for non-negative integer object IDs with
zero as background. It is distinct from a boolean `mask` and an integer
intensity `image`. Label outputs inspect and pin as napari Labels layers, retain
their integer IDs through TIFF/NumPy saving, and connect only to label-aware or
generic array inputs. Image outputs pin as napari Image layers; pinned mask
outputs remain napari Labels layers so they can be used as overlays.

Connected-component labeling, label-volume/property filtering, relabeling, and
Fill Holes expose metadata-aware 2D/3D spatial processing. Auto mode resolves
from carried axis metadata and stores the resolved spatial dimensionality for
Python export. Leading non-spatial axes are processed independently, so `TCZYX`
data is processed per timepoint and channel while each `ZYX` block is treated
as one volume.

`Clear Border Objects` accepts only masks or labels, preserves the connected
semantic output type and retained label IDs. For 3D inputs it can examine all
`ZYX` boundaries or lateral `YX` boundaries only, with a data-aware border
buffer.

`Fill Holes` accepts masks; a maximum size of `0` fills all enclosed holes.
Positive limits fill only holes up to the selected area or volume.
Auto uses 2D for true `YX` inputs and 3D for z-stacks; an advanced per-slice
mode is available for deliberately slice-wise segmentations. The inspector
hides 3D for true 2D inputs, labels the size control as area or volume, and
warns when per-slice filling is selected for a z-stack.

`Remove Small Objects` accepts masks or labels and preserves the connected
semantic type. It uses metadata-aware 2D/3D spatial blocks, preserves retained
label IDs, and exposes face/full connectivity for mask components. Its
minimum-size control is logarithmic, uses contextual area/volume labeling, and
is bounded by the largest observed input object.

`Filter Labels By Property` declares named `Labels` and `Measurements table`
ports. It filters label IDs using a numeric table column such as physical
volume, intensity, or skeleton/network metrics. Rows with leading index columns
such as `t_index` are matched to the corresponding non-spatial block, so
time-series labels are filtered per frame. The node preserves retained label
IDs; use `Relabel Sequential` when compact IDs are needed after filtering.

`Measure Objects` accepts labels and outputs a `table` rather than an image.
The implementation uses `skimage.measure.regionprops_table` for one row
per labeled object. It measures label ID, pixel/voxel area or volume,
calibrated physical area or volume when spatial scale metadata is available,
centroid, bounding box, equivalent diameter, extent, and Euler number. Leading
time/channel or other non-spatial axes become index columns so repeated label
IDs remain distinguishable across frames. Optional checkbox groups add
shape descriptors, axis/inertia descriptors, 2D boundary descriptors, derived
shape ratios, and 2D shape moments. The 2D-only groups are hidden when the
connected input resolves to true 3D. Derived shape ratios include axis ratios,
bounding-box side lengths/aspect ratios, fill fraction, and inertia eigenvalue
ratios. The 2D shape moments group includes Crofton-based circularity,
perimeter-to-area ratio, and Hu moments. The 3D-safe extensions include
bounding-box volume, filled volume, major/minor axis length, inertia tensor
eigenvalues, and derived 3D shape ratios.

When spatial scale metadata is available, `Measure Objects` and `Measure
Objects + Intensity` also emit calibrated physical variants for extended
non-mesh measurements: centroid and bounding-box coordinates, equivalent
diameter, bounding/fill size, 2D convex area, maximum Feret diameter,
major/minor axis length, bounding-box side lengths, and inertia tensor
eigenvalues. Physical 2D perimeter and Crofton perimeter are emitted only for
isotropic spacing; anisotropic 2D physical perimeter columns remain `NaN`
instead of using an ambiguous scalar scale.

`Measure Objects + Intensity` is the first named heterogeneous-input node. It
declares separate `Labels` and `Intensity image` input slots, requires matching
array shapes in the first implementation, and emits the basic morphology table
plus per-label mean, minimum, maximum, sum, and standard deviation intensity.
It exposes the same optional morphology groups as `Measure Objects`. The same
`InputSpec` mechanism is intended for watershed, colocalization, and other
heterogeneous-input nodes.

`Measure 3D Mesh Morphology` is a manual/cached `labels -> table` node for
surface-based 3D object morphology. It requires true 3D spatial labels, uses
`skimage.measure.marching_cubes` with carried Z/Y/X spacing, computes mesh
surface area with `skimage.measure.mesh_surface_area`, computes mesh volume
with a local signed triangle-volume helper, and optionally computes convex-hull
area/volume with `scipy.spatial.ConvexHull`. It emits one row per object per
leading axis block and preserves identity columns such as `t_index` and
`label_id`. Tiny, flat, or otherwise invalid objects are not dropped; their
mesh columns are `NaN` and `mesh_status` / `mesh_error` explain the failure.

`Merge Tables` accepts a variable number of `table` inputs and emits a joined
`table`. The `input_count` parameter controls visible/required input ports.
`auto` join keys use shared stable identity columns such as `t_index`,
`c_index`, `z_index`, `label_id`, `object_id`, or `component_id`; if no
identity columns overlap and all tables have the same row count, the node falls
back to row-position joining. Duplicate non-key columns from later tables are
suffixes such as `_table2` so no source column is silently overwritten.

`Select Table Columns` accepts one `table` input and emits a table containing
the checked upstream columns in the displayed order. The inspector lists
detected columns with checkboxes, Select all/Deselect all buttons, and row
reordering controls; workflows store the compact ordered `columns` value. It
preserves row order and column units, and errors on missing explicit columns so
batch exports fail clearly when an upstream table schema changes.

`Add Metadata Columns` accepts one `table` input and appends constant
user-supplied `name=value` columns, intended for treatment, replicate, batch,
condition, or source annotations before export. Existing columns are protected
unless the node's overwrite parameter is explicitly set to `yes`.

`Summarize Measurements` accepts one `table` input and emits grouped summary
statistics. Auto grouping prefers metadata and leading index columns such as
`condition`, `replicate`, `source_name`, and `t_index`; users can also enter
explicit group, value, and statistic lists. Numeric value columns are summarized
with count, mean, median, standard deviation, min/max, sum when requested, and
quartiles. Unit metadata is propagated to summary columns when the statistic
retains the source measurement unit.

`Skeletonize` accepts masks and produces a binary skeleton mask in metadata-aware
2D or 3D spatial blocks. `Analyze Skeleton` accepts a skeleton mask and outputs
a `table` with one row per connected skeleton component. It reports skeleton
voxel count, endpoint voxels, junction voxels, isolated nodes, simplified
graph node/edge counts, voxel-graph edge count, cycle count, per-block
connected-component context, and skeleton length in pixels/voxels plus
calibrated physical units when spatial scale metadata is available. `Measure
Skeleton Branches` emits one table row per traced branch with branch type,
length, endpoint-to-endpoint distance, tortuosity, start/end coordinates, and
calibrated physical length when possible. `Summarize Skeleton Branches`
converts those row-per-branch tables into grouped branch-count, length,
tortuosity, and branch-type count/fraction distributions. `Skeleton Graph
Tables` exports explicit graph-node and graph-edge tables. `Measure Overall
Skeleton Network` emits per-block connectedness, fragmentation, branch-count,
branch-length, and normalized per-component/per-length whole-network metrics.
`Skeleton Keypoints`, `Skeleton Graph Overlay`, `Label Skeleton Components`,
`Label Skeleton Branches`, and `Prune Skeleton Branches` provide visual QC
masks/RGB overlays/labels and terminal-spur cleanup using the same skeleton
graph rules. The first graph analyzers deliberately stay generic;
mitochondrial-specific mesh/surface and specialist network indices should be
separate optional nodes.

The longer-term measurement architecture must support selectable measurement
families and merged per-object result tables for exploratory statistics such as
PCA and treatment-group separation. See
[mitomorph-feature-parity.md](mitomorph-feature-parity.md).

Operation dtype policy:

- Preserve the input dtype for intensity-preserving transforms whenever that can
  be done without changing the mathematical meaning of the output. Examples are
  blur, denoise, sharpen, clip, gamma, linear scale/offset, and Sobel edge
  magnitude.
- Use natural output dtypes for semantic type changes: masks are `bool`, labels
  are integer labels, measurement nodes are tables, and RGB composites are
  display floats.
- Keep float outputs when the operation's result has a new numeric domain that
  cannot be represented safely in the original integer dtype, such as ratio,
  weighted image calculation, z-score normalization, Difference of Gaussians, or
  Laplace signed responses.

Normalization internal to a filter must not destroy comparability across planes
or silently redefine input units. Bilateral, unsharp-mask, and non-local-means
filtering use one affine intensity scale for the complete input and invert it on
output, rather than normalizing each plane independently. Bilateral
`sigma_color` is expressed in input intensity units; non-local-means `h` is
explicitly a fraction of the shared normalized span. Invalid scales and
non-finite intensities fail instead of being repaired or replaced. Float
unsharp-mask overshoot is retained in the original intensity units.

To add a node:

1. Define the scientific input/parameter/output contract and implement a pure
   function in `core/operations.py`.
2. Import it in `core/pipeline.py` and add an `OperationSpec` to `NODE_LIBRARY`.
3. Register or implement its physical-grid validation when it combines images.
4. Add or update metadata propagation rules in `core/metadata.py` if the node
   changes axes, dtype semantics, mask/intensity kind, or channel structure.
5. Add numerical, invalid-input, dtype/range, read-only-buffer, metadata,
   workflow, and export tests as applicable.
6. Reuse or add controls under `ui/`. Add only the final composition/viewer
   wiring to `_widget.py`, with a widget test for that boundary.

The fuller recipe and test ladder are in
[developer-notes.md](developer-notes.md#add-or-change-a-scientific-operation).

## Metadata

`core/metadata.py` carries OME-NGFF-inspired state with every graph output. This
is intentionally stricter than "just an array shape" because bioimage workflows
need to know which axes are time, channels, z, y, and x.

`ImageState` records:

- shape and dtype;
- semantic axes as `AxisMetadata`;
- per-axis semantic confidence (`explicit` or `shape-inferred`) and aggregate
  image confidence (`explicit`, `shape-inferred`, or `mixed`);
- axis type, scale, unit, and translation/origin where available;
- optional source-axis indices that map source and derived axes back to the
  correct napari viewer sliders, including right-aligned layers with fewer axes
  than the current viewer;
- kind, such as intensity image, binary mask, RGB image, or multi-channel image;
- bit depth, exact finite value range/binary pattern when materialized, and
  memory estimate;
- metadata source;
- source name and operation history.
- normalized channel names, colors, fluorophore/wavelength fields;
- preserved acquisition description/date and instrument reference strings;
- source URI, format, series index/name, and source UUID.

For registered multi-input operations, `core/pipeline.py` passes these states to
`core/grid.py` before calling the array kernel. Equal array shapes alone are not
accepted as proof of alignment: semantic axis name/type, sample count, scale,
compatible units, and origin must describe the same sampled coordinates.
Image/PSF validation intentionally allows different kernel extents and origins
but still requires matching spatial axis semantics and sample spacing. Shape
broadcasting that an operation deliberately supports remains under that
operation's explicit shape contract; VIPP never inserts a hidden resampling
step.

`TableState` records row and column counts, stable column names, column units
where known, source name, table/measurement kind, metadata source, and operation
history. Node cards use the table summary; the inspector shows table metadata
and a bounded row preview.

Source metadata comes from, in order of preference:

1. carried `vipp_image_state`;
2. OME or OME-NGFF-like `multiscales` metadata;
3. explicit layer axis hints such as `axes`, `axis_order`, or `vipp_axis_order`;
4. inferred axes from shape, explicitly labelled as inferred.

Shape inference is descriptive fallback metadata, not authority to choose a
scientific interpretation. Before operations use names/types to resolve an
automatic spatial mode, channel axis, named projection, or automatic PSF
parameters, `core/pipeline.py` requires the relevant axes to be explicit. A
shape-inferred state raises `AmbiguousAxisError` with the explicit metadata,
mode, or axis-index remedy. This prevents a three-plane volume or
three-column image from silently becoming RGB, and prevents a leading axis from
silently becoming Z.

`Set Pixel Size / Units` is the explicit calibration repair node for files or
napari layers that lack reliable physical metadata. It updates X/Y pixel size,
optional Z step size, and the shared physical unit in carried `AxisMetadata`
without changing pixel values. `Rescale Axes` changes the pixel grid along
X/Y/Z either by scale factors or explicit output pixel counts, optionally locks
X/Y aspect ratio, supports nearest, linear, cubic, and spline interpolation,
and updates `AxisMetadata.scale` from the dimensions actually produced.
Automatic interpolation resolves to linear for intensity images and nearest
neighbor for masks and labels. `Orthogonal Projection` uses calibrated Z/Y/X
spacing to size its XZ and YZ panels, so anisotropic z-stacks display with
physical proportions.

The inspector uses `metadata_table_rows()` and `metadata_history_items()`. Node
cards use `format_compact_metadata()`. Metadata is useful for UI labels,
parameter bounds, preview slicing, histogram slicing, and all shared writers.
Table outputs use the same metadata helpers but do not participate in image
preview slicing or histograms. The inspector therefore hides the per-node
thumbnail toggle for table-only nodes such as measurement outputs.

`core/io/` is the only format-specific boundary. `inspect_image_source()`
discovers selectable image items without loading full TIFF pixel data and also
inspects ordinary raster images through imageio/Pillow. `read_image()` returns
an `ImageDataset`; OME-Zarr data and multiscale levels remain Dask-backed.
`write_image()` dispatches explicit OME-Zarr, OME-TIFF, ImageJ TIFF, TIFF, NPY,
and 2D raster formats. Auto mode selects NPY by suffix, OME-Zarr for `.zarr`,
PNG/JPEG-style raster output by suffix, OME-TIFF for `.ome.tif[f]`,
conventional TIFF for label images, and OME-TIFF for other TIFF outputs.

## Preview And Histograms

`core/preview.py` reduces arbitrary output arrays to displayable thumbnails.

`make_preview(data, mode, current_step, state, channel_colors=None)` supports:

- `Slice`: reduce non-spatial axes by current napari dims or midpoint;
- `MIP`: project spatial axes where appropriate;
- `Off`: skip preview generation;
- state-aware channel handling when `ImageState` is available;
- pseudo-colour fluorescence composites for multichannel data;
- Combine Channels display colours from `channel_colors`.

Slice previews, selected-node histograms, and the metadata panel use
`AxisMetadata.source_axis` when the current napari viewer has more dimensions
than the node output. This matters for workflows like `TCZYX -> Split Channels
-> TZYX`: the derived local `z` axis is output axis 1, but it is still
controlled by the original viewer Z slider at source axis 2.

`Reorder Axes` is a pure transpose: each complete axis record (name, type,
scale, unit, origin, confidence, and source-axis mapping) moves with its pixels.
It never reinterprets original Z samples as Y merely because Z was moved into a
formerly Y-shaped position. State-aware thumbnails keep following the same
napari source sliders. Operations that support arbitrary layouts resolve the
moved names; positional kernels reject a noncanonical explicit suffix instead
of silently processing the wrong plane or volume.

Napari also right-aligns layers with fewer dimensions than the current viewer.
For example, a `CZYX` layer displayed in a 5D viewer is controlled by global
sliders 1, 2, 3, and 4. Napari-layer source payloads therefore rewrite
`source_axis` with this offset before the graph runs. The widget subscribes to
both `dims.events.current_step` and `dims.events.point`, using a bound method
instead of an anonymous callback so the subscription remains alive.

`ViewDimsBar` is the VIPP-local dimension navigator. It builds controls from the
active image context, preferring the pinned node, then the selected image node,
then the input image. It exposes semantic non-XY axes with size greater than one
(`T`, `Z`, `C`, etc.). Controls are keyed by canonical/source axis, not merely
by the currently displayed layer-axis position. When the active node has dropped
an axis, such as Split Channels removing `C`, the bar maps the remaining `Z`
control back to the raw napari axis that still represents source `Z`. For nodes
with a local axis length different from the viewer axis length, it displays the
local range and maps to napari by relative position. The bar writes changes
through `viewer.dims.set_current_step` rather than maintaining a parallel
coordinate state, and has its own responsive behavior: full inline sliders,
compact spin boxes, then a `View dims...` popup menu.

Important nuance: channel pseudo-colour is carried metadata, not pixel data.
OME source colours, Image Source overrides, Combine Channels choices, and the
`Assign Channel Colors` node all write `ChannelMetadata.color`. The underlying
output stays a multichannel array with a channel axis until `Composite → RGB`
is used. Only axes explicitly declared `rgb` or `rgba` preserve encoded colour
order; a generic `C` axis, including a channel-last axis of length three or
four, remains fluorescence data and is blended by carried pseudo-colours. A
yellow channel therefore writes to red and green, while a cyan channel writes
to green and blue. The default mapping retains the native intensity scale
without normalization or clipping, checks conversion and accumulation for
precision/overflow hazards, and rejects unsafe inputs. The legacy per-channel
1st/99th-percentile normalization is available only as an explicitly labelled
lossy choice. Manual red/green/blue selectors still force single-channel RGB
plane mapping. Split Channels exposes a `Thumbnail channel` inspector parameter
that chooses which output port is presented when the downstream graph does not
identify one unambiguous channel.

The presentation-output resolver collects the distinct `source_port` values of
connections leaving a `Split Channels` node. Exactly one distinct used port
becomes the effective presentation output, even when several downstream
connections consume it. Zero used ports or more than one distinct used port
fall back to the saved `preview_channel` (`Thumbnail channel`) parameter. The
same effective output is used consistently for the node thumbnail, napari
inspect and pin layers, the output histogram, metadata, dimension controls, and
`Save selected output...`. Resolving a presentation port never mutates
`preview_channel`, graph connections, cached per-port arrays, or the outputs
supplied to scientific operations.

For generated napari inspect/pin layers, 2D RGB outputs use napari's native
`rgb=True` image layer. Volumetric RGB outputs are displayed as synchronized
additive red/green/blue image layers because napari's scalar-field 3D status
and rendering path does not reliably handle hidden RGB axes.

Generated large image layers receive explicit dtype-based provisional contrast
limits so napari never performs an unannounced full-array scan on Qt's thread.
VIPP then calculates exact full finite extrema in the background, caches them by
array identity, rejects stale results, and preserves a user's intervening manual
contrast edit. This state is display provenance only and does not enter graph
calculations.

Toolbar thumbnail controls are global display settings. `Preview` chooses the
thumbnail reduction (`Slice`, `MIP`, or `Off`); `Off` disables node-card
thumbnail generation globally. The selected-node inspector checkbox still acts
as a per-node opt-out when previews are enabled. `Contrast` chooses how scalar
thumbnail intensities are mapped to display: `Percentile` uses the current
robust 1st/99th percentile stretch, `Min-max` uses the displayed data minimum
and maximum, and `Raw` uses dtype/range display without adaptive stretching.
For `uint16`, raw mode maps 0..65535 to 0..255, so dim microscopy signals may
appear dark by design. The toolbar `Mono` dropdown controls the colormap used
for monochrome thumbnails only. These controls are display settings, not
metadata, and do not alter graph outputs or histogram values.

Scientific diagnostic reductions live in `core/diagnostics.py`; Qt plot widgets
live in `ui/plots.py`. Typed request/result objects and Qt runnables for
thumbnail contrast, input histograms, colocalization scatter, automatic
scale/offset, and generated-layer contrast live in
`ui/diagnostic_workers.py`. The runnables accept narrow calculation callables
and do not import or reach back into `VippWidget`. `_widget.py` still owns
slice/stack selection, cache and stale-result keys, marker policy, worker
submission, and result application. The general selected-output histogram
supports slice-vs-stack and linear-vs-log display.
Multichannel histograms are drawn as separate series using fluorescence-style
colours. The UI resolves a semantic channel axis and passes it explicitly to
`exact_histogram()`; the core does not infer RGB merely because a trailing axis
has length three or four.

Inspector histograms count every finite pixel in the declared slice or stack;
there is no hidden large-array sampling. Bounded chunks limit temporary memory
without changing the population. Exact integer bin placement avoids converting
wide integer offsets to float and collapsing distinct values. Display bins are
separate from the dtype-aware operational bins used to calculate automatic
thresholds.

Cutoff-style nodes listed in `INPUT_HISTOGRAM_OPERATIONS` also show an
`Input Histogram` above the general output histogram. It has its own
`Histogram uses` slice/stack selector, hidden when the connected input has no
meaningful stack axis, and its markers are driven by the node parameters.

Input histogram caching separates two dependency domains. The bounded display
distribution is keyed only by array identity, shape/dtype, semantic axis
signature, normalized slice/stack scope, and the effective slice position. A
second key contains the operation plus only parameters that affect its guide
markers. Manual Binary/Hysteresis and explicit Rescale/Clip markers are rebuilt
synchronously over cached counts; exact percentile and automatic-threshold
markers can run independently in the background. Small and large inputs use
the same cache contract. Pipeline completion invalidates the selected-output
histogram but does not discard an unchanged upstream input distribution; full
workflow/source invalidation clears both caches.

`Rescale Intensity` has an explicit `cutoff_mode`. New nodes default to
`Percentiles`, which computes the requested low/high percentiles from every
finite input value. `Values` uses the stored low/high values directly. The
inactive pair is informative only and cannot silently override the active
mode. Both the mode and cutoff fields are ordinary serialized node parameters.
The percentile markers always describe the full connected input, matching the
operation, even when the inspector is drawing only a slice histogram. Starting
a drag from either percentile marker atomically seeds the explicit-value pair
from the displayed exact cutoffs and changes `cutoff_mode` to `Values`; later
drag events reuse the cached distribution and update the active value parameter.

`Clip Intensity` similarly stores `cutoff_mode = Data range | Values`. New nodes
default to `Data range`. Its data-range markers likewise describe the complete
input; the histogram scope affects only the inspector distribution.

For integer Rescale inputs, percentile cutoffs are exact linear order
statistics over a native-dtype working buffer. The cutoff is retained as an
integer or rational value, then each bounded processing chunk is translated
from the cutoff's integer origin before float64 ratio arithmetic. Exact native
endpoint masks prevent distant saturated values from entering that conversion.
Input and output intervals wider than 2^53 fail explicitly. Integer Clip takes
the simpler fully native path: bounds must be integral, inactive out-of-dtype
sides are clamped to the dtype limit, and `np.clip` never promotes the image to
float. These rules prevent large int64/uint64 offsets from collapsing adjacent
levels.

Colocalization threshold nodes build the inspector's 2-D scatter density by
accumulating a 255 x 255 histogram over every ROI voxel in bounded chunks.
ROI and colocalized counts are accumulated in the same exact pass. Large inputs
run on a stale-safe background worker and cache only the compact density/result;
there is no separate sampled display population. This exact computational
helper still resides in `ui/plots.py`, rather than the Qt-free diagnostics
module, and is a known remaining boundary seam.

When `Filter Labels By Volume` is selected, a second histogram above the
general histogram shows the object-volume distribution from the unfiltered
label input. Its `Log volume axis` toggle defaults on, and it draws live
minimum and enabled maximum threshold markers. Because the distribution uses
the input labels, it remains stable while the filter removes objects. The
per-object volume population is cached independently of the live minimum and
maximum guides, avoiding another full label-image scan while either guide is
dragged.

When a table node is selected, the general histogram is hidden and the inspector
shows a `Table Preview` group with the first rows and unit-annotated headers.
Table nodes do not create napari image/labels layers when inspected or pinned.

## Graph UI

`_graph.py` owns the QGraphicsView/QGraphicsScene node canvas. It is
presentational: it emits signals, and `VippWidget` mutates the pipeline model.

`PipelineGraphView` keeps graph zoom as a calibrated UI percentage. The `100%`
default applies a 3.125x visual multiplier relative to the fitted graph view,
so the previous calibrated `125%` view becomes the current default readable
size. The toolbar slider covers 40% to 250% in this calibrated scale, and the
compact reset icon returns to `100%`. Ctrl/trackpad wheel zoom remains active
with a broader 20% to 400% range. Wheel zoom emits the same zoom signal so the
toolbar label can show out-of-slider values while the slider thumb clamps to its
end.

Main classes:

| Class | Role |
| --- | --- |
| `PipelineGraphView` | Canvas, pan/zoom, drag/drop node creation, wire creation/removal, selection, delete-key handling, and node context menus. |
| `NodeProxy` | Movable graphics item wrapping a `NodeCard`; owns the visual input/output ports (one or more output ports for multi-output nodes). |
| `NodeCard` | Embedded widget with category tint, title, thumbnail, and compact metadata. Pin state is represented by card styling; Pin/Unpin actions live in the inspector and node context menu. |
| `PortItem` | Input/output port circle with hover/drop feedback, optional accent colour/label, and optional tunnel badge. Multi-input and multi-output nodes have several ports. |
| `ConnectionItem` | Curved visible wire storing source id, target id, `target_port`, and `source_port`. Tunnel-marked connections are not drawn as `ConnectionItem`s. |

Right-clicking a node opens a context menu with Delete, Inspect Code, Duplicate
Node, and, for image/mask/label-producing nodes, Pin or Unpin. Duplicate Node
copies the node operation and current parameter values into an unconnected node
near the original; it does not infer connections. Inspect Code opens a
read-only dialog containing node metadata, connected input references, a
single-node call shape, and the pure operation function source when available,
with lightweight Python syntax highlighting.

Right-clicking an output port can create, rename, or remove a named output
tunnel. Right-clicking an input port can subscribe that input to any compatible
named output tunnel, replacing the current input slot connection if needed.
Tunnels are intended for dense workflows where a channel, mask, ROI, or
reference image feeds many downstream nodes. They remain explicit graph
semantics: the connection occupies the target input slot, is type-checked,
rejects cycles, participates in execution, and is saved in workflow JSON.

Signals to `VippWidget`:

- `node_selected(node_id)`
- `node_delete_requested(node_id)`
- `node_duplicate_requested(node_id)`
- `node_code_requested(node_id)`
- `pin_requested(node_id)`
- `node_create_requested(operation_id, scene_position)`
- `connection_requested(source_id, target_id, target_port, source_port)`
- `connection_removed(source_id, target_id, target_port)`
- `port_context_requested(kind, node_id, port_index, global_position)`
- `status_message(text)`

Slot-aware connections are important. The graph must never treat all wires into
a multi-input node as one anonymous target, nor all wires out of a multi-output
node as one anonymous source. `target_port` and `source_port` are the contract
between the visual ports, the pipeline model, workflow JSON, and Python export.
Input ports can also have distinct semantic types on one node; the visual graph
uses these per-port types for colour, tooltip text, and compatible-drop
feedback.
`VippWidget._sync_node_output_ports(node_id)` pushes a node's declared
`OutputSpec`s (labels and accent colours) onto its `NodeProxy` after the graph is
built or a node is added.

## Widget UI

`_widget.py` owns `VippWidget`, the object exposed by `napari.yaml`. Architecturally
it is the `npe2` composition root and compatibility facade: it creates the
pipeline and widgets, receives the napari viewer, injects providers/actions into
controllers, and connects signals across the core, graph, and UI layers. It is
not yet a small passive shell; substantial application orchestration remains
there.

The implemented decomposition is:

| Module | Responsibility moved out of `_widget.py` |
| --- | --- |
| `ui/controls.py`, `ui/axis_controls.py`, `ui/view_dims.py` | Generic parameters, semantic axis controls, and local dimension navigation. |
| `ui/palette.py`, `ui/search.py`, `ui/dialogs.py`, `ui/examples.py` | Palette/search behavior, reusable dialogs, and example-resource catalog. |
| `ui/plots.py` | Histogram and colocalization scatter presentation widgets. |
| `ui/diagnostic_workers.py` | Typed diagnostic requests/results and Qt runnables with injected calculation ports. |
| `ui/source_adapter.py`, `ui/file_sources.py` | Live-layer revision tracking and Qt scheduling for verified file snapshots. |
| `core/execution.py`, `ui/workers.py` | Headless pipeline request/result service and its thin Qt runnable. |
| `core/batch_setup.py`, `ui/batch.py`, `ui/batch_controller.py`, `ui/batch_navigator.py` | Headless batch configuration, retained collection workspace, application controller, and representative/progress navigator. |
| `ui/history.py`, `ui/lifecycle.py` | Undo/redo state and terminal shutdown of external callbacks/background work. |

`_widget.py` still owns the three-pane/toolbar/inspector assembly, graph-editing
callbacks, file inspection and source-cache orchestration, dirty-node and
manual-node scheduling, diagnostic request construction/cache/submission/result
coordination, generated napari layer presentation, and save/load/export command
wiring. Those are candidates for later controller extraction when a cohesive
boundary and a focused test can be named. Moving code solely to reduce line
count is not an architectural goal.

The main route through `run_pipeline()` is:

1. Finish or schedule any missing verified file-source snapshots.
2. Resolve graph-level sources into owned payloads and revision tokens.
3. Calculate the dirty subgraph and manual-node execution policy.
4. Run small work directly or submit a typed background execution request.
5. Accept a result only when the run, graph, and source revisions are current.
6. Refresh dynamic controls and presentation state from the accepted pipeline,
   copying arrays before they enter napari layers.

Most parameter changes go through a 150 ms single-shot debounce timer. Some UI
changes, such as Combine Channels colour identity, also call
`_update_thumbnails()` immediately because they affect display but not numeric
array values.

### Dock Window Integration

Napari hosts plugin widgets inside its `QtViewerDockWidget`, which is a
`QDockWidget`. Qt normally turns a floating dock into a `Qt.Tool` window without
maximize controls, and handles a native title-bar double-click by re-docking the
widget. `VippWidget` installs an event filter on its containing dock and listens
to `topLevelChanged` so a detached VIPP panel is promoted to a standard
`Qt.Window` with minimize/maximize/close controls. A floating title-bar
double-click is intercepted to toggle maximized/normal state while keeping the
panel detached. Keep this behavior plugin-local; do not patch napari internals.

## Input, Output, Persistence, And Export

Input is represented by explicit graph-level Image Source nodes. Image Source
supports:

- existing napari layer;
- file path (`.npy`, `.npz`, OME-TIFF/ImageJ/conventional TIFF, OME-Zarr
  stores, and common raster images such as PNG/JPEG/BMP/GIF/WebP);
- bundled synthetic sample;
- adaptive file/store inspection with a series or image selector when multiple
  items are available.

Quick output saving:

- The inspector has `Save selected output...`; it defaults to OME-TIFF, allows
  `.npy`, and exposes PNG/JPEG-style raster formats only when the selected
  output is 2D intensity or 2D RGB/RGBA.
- `save_array_output()` writes TIFF as ImageJ hyperstacks when metadata allows.
- `save_array_output()` writes ordinary raster formats only for 2D arrays;
  stacks must use TIFF/OME/Zarr/NumPy formats.
- Binary masks are saved as 8-bit `0`/`255`, not `0`/`1`.
- Integer label images use standard TIFF rather than ImageJ TIFF so 32-bit IDs
  are preserved.

Graph-level output:

- `Save Image` passes its input through unchanged.
- If auto-save is enabled, it writes on every graph recompute.

Workflow persistence:

- `core/workflow.py` serializes nodes, params, connections including
  `target_port`, `source_port`, optional tunnel names, output tunnel
  definitions, and canvas positions to JSON.
- Workflow version 3 stores graph notes, VIPP UI metadata, and the required
  scientific controls for threshold/cutoff behavior, explicit channel
  semantics, and composite intensity mapping. Versions 1 and 2 are
  intentionally rejected instead of receiving inferred scientific parameter
  migrations; changing only the JSON version number is not a valid migration.
- Inspector metadata is always written when saving through the widget and
  records the selected node plus right-panel visibility. Per-node thumbnail
  visibility is written only when `Save thumbnail visibility in workflows` is
  enabled.
- Workflow JSON does not serialize thumbnail pixels, cached arrays, cached
  tables, environment provenance, or YAML.
- Image Source paths and layer names are serialized as literal parameters;
  input files are not embedded and paths are not rebased for portable sharing.
- The loader requires the current type and version and rejects unknown
  operations, malformed records, duplicate node ids, invalid positions, and
  dangling or multiply occupied connections. It also rejects duplicate tunnel
  names, tunnel connections without a declared source, tunnel connections whose
  stored source/output does not match the named source, and VIPP workflow
  metadata that references missing nodes.
- `PrototypePipeline.restore_graph()` rebuilds the graph model.
- `workflow_snapshot_from_pipeline()` and
  `workflow_snapshot_from_document()` produce defensively copied typed runtime
  snapshots. Document decoding also materializes a temporary pipeline, so port,
  type, cycle, and duplicate-input validation completes before a snapshot can
  replace live editor state.
- `save_workflow()` writes through `core.atomic_io.atomic_write_json()`: JSON is
  encoded with non-finite values forbidden, flushed and fsynced to a
  same-directory temporary file, then atomically replaced. A serialization or
  replacement failure leaves the previous workflow intact.
- `VippWidget.load_workflow_file()` provides the non-dialog load path used by
  the example launcher.

Python export:

- `core/export.py` emits a runnable script with `run_pipeline()`,
  `batch_process()`, image load/save helpers, and an argparse entry point.
- The generated program embeds validated canonical workflow JSON and creates a
  fresh shared headless executor for every call, so it uses the same graph,
  port, parameter, semantic-axis, and scientific-operation contracts as VIPP.
- `ImageDataset` and `SourcePayload` bindings carry explicit source data,
  metadata, names, and `ImageState`; returned `PipelineResults` preserve output
  states for metadata-aware saving.
- Workflows with several independent sources require an unambiguous binding for
  every source. Missing, duplicate, and unknown bindings fail before execution.
- Generated programs are locked to the VIPP version that created them and fail
  on a different runtime version instead of silently changing behavior.
- The command-line folder helper supplies one primary image source as a simple
  convenience. Saved batch configuration remains the complete multi-source
  collection interface.

Collection batch UI:

- `VippWidget._run_collection_batch()` captures one serialized workflow and
  passes it to the headless setup/runner boundaries, which materialize their own
  pipeline. Folder processing therefore does not replace the live canvas
  outputs with the last processed file.
- `ui.batch.CollectionBatchDialog` owns form presentation and delegates through
  injected `CollectionBatchActions`. `ui.batch_controller.CollectionBatchController`
  coordinates save/load/preview with stable workflow and pipeline providers;
  `core.batch_setup` constructs and validates configuration without Qt.
- A successful preview exposes the complete tuple of `BatchItemPlan` values to
  a UI-only session while keeping the table sample limited. `ui.batch_navigator`
  selects a zero-based representative position and reports one-based runner
  progress. It never becomes part of serialized scientific workflow state.
- Representative selection atomically replaces the transient path mapping for
  every collection-bound Image Source, then uses normal verified source loading
  and graph execution. Fixed Image Sources remain fixed. No list-producing Batch
  Input node is required, and preview paths do not change workflow parameters or
  the scientific workflow hash.
- Requested and committed representative indexes are separate: asynchronous
  source or graph work updates the committed item only after the matching
  generation succeeds. Materialized arrays from older slider positions are
  evicted, while their verified file identities remain pinned to reject changed
  revisions on revisit.
- The batch workspace is retained and modeless so the graph remains visible.
  Preview-table selection and the full-plan navigator stay synchronized;
  execution compares fresh preflight with the reviewed full plan and revalidates
  pinned representative source identities, stopping for review when either
  changes. It reports accurate item-level progress, final manifest statuses,
  validation, and the manifest path in the workspace.
- The dialog lists every `Image Source` node as a possible batch source binding.
  A blank row is accepted only for an existing fixed file-path source;
  napari-layer and sample sources must be collection-bound.
- `core.batch.BatchSourceConfig`, `BatchItemPlan`, and `BatchOutputPlan` are the
  Qt-free source, item, and output contracts. `BatchPreviewResult` exposes a
  limited row sample plus full-plan item and collision totals to the dialog.
- When several sources are bound, matched paths are sorted per source and paired
  by position. All bound sources must match the same number of files. The first
  bound source becomes the primary source for default naming.
- Batch planning is deterministic and separate from graph execution. Preview
  and execution use the same planner instead of calculating names through
  separate UI paths. Execution performs a fresh preflight, then passes that
  exact plan into the runner so changes since an earlier preview are detected.
- `Batch Output` nodes are the authoritative save markers. They pass data
  through during normal graph execution and provide tag, format, subfolder,
  filename-template, and overwrite controls for batch saves.
- Filename templates can use batch-aware fields including `{batch_id}`,
  `{batch_index}`, `{source_name}`, `{source_stem}`, `{primary_source_stem}`,
  `{tag}`, `{node_id}`, and `{node_title}`.
- If a graph has no `Batch Output` nodes, terminal graph outputs are saved as a
  compatibility fallback. Image-like fallback outputs use the selected batch
  image format; table fallback outputs are saved as CSV. Planning exposes a
  warning because terminal membership is less stable than explicit output
  declarations. A terminal with multiple output ports is rejected because the
  fallback cannot represent a port selection.
- `vipp_batch_config.json` is a versioned schema independent of workflow schema
  version 3. It persists source bindings and patterns, output location and
  default format, existing-file policy, the required workflow companion, the
  optional runner choice, the workflow hash, and resolved output declarations.
  Load validates the workflow hash so a configuration cannot silently select
  outputs from a different graph.
- The batch-level existing-file choices are `Error`, `Skip`, and `Overwrite`.
  A `Batch Output` node with an explicit `yes` or `no` overwrite value takes
  precedence over the default. Collision state is part of the plan and shown
  before execution.
- Dialog-started runs write the resolved `vipp_batch_config.json` and required
  `vipp_batch_workflow.json`; headless replays use those files at their existing
  locations. Every individual artifact uses atomic replacement, but the
  workflow/config pair is not one multi-file transaction. Every execution
  writes a `vipp_batch_manifest.json` latest-run view.
  A run-id manifest archive embeds the canonical config and scientific graph,
  while a run-id sidecar directory atomically records each item as it runs and
  after every output. The final manifest contains hashes, software versions,
  input identity and available source metadata, output policy/path/status,
  errors, and summary counts. Output statuses are `pending`, `completed`,
  `skipped`, and `failed`; item statuses additionally include `running` and
  `partial`. After an interrupted process, the sidecars are the recovery
  checkpoints; the canonical latest/archive manifests are finalized on normal
  runner exit rather than reconciled automatically.
- For each item, the runner captures exact identities for all collection-bound
  and fixed file sources before reading. It fully writes every available output
  to a private same-directory staging path, including forcing lazy output bytes,
  then reverifies every input identity. Only a verified item begins final
  publication. If any source changed, all staged files are removed and every
  output is marked failed. Otherwise outputs are promoted atomically one at a
  time under their declared `Error`, `Skip`, or `Overwrite` policy. A later
  promotion failure can therefore produce an explicitly recorded partial item,
  but no output is published from mixed source revisions.
- Batch failures are isolated at item/output boundaries. Successful writes and
  manifest records remain available; later items continue by default or are
  marked skipped when continuation is disabled. The returned summary
  distinguishes completed, partial, skipped, and failed items.
- The dialog can additionally write a thin `vipp_batch_pipeline.py` launcher
  beside the required workflow/config artifacts. The launcher resolves the
  workflow recorded by its config unless an override is supplied and delegates
  to the shared headless batch core; it is distinct from the immutable
  shared-executor workflow program emitted by `Export Python...`.
- Collection execution remains local-folder oriented. Semantic-axis iteration
  and plate/well/field HCS traversal are deliberately deferred rather than
  inferred from array axes or directory names.
- `core.batch_demo` generates a portable three-item, two-source NumPy bundle
  without Qt or napari state. Its bundled graph writes explicit NPY, TIFF, and
  TSV outputs; the validator compares decoded results with exact ground truth
  and checks config/workflow hashes, source identities, final/archive
  manifests, and item sidecars. The batch workspace and flagged example entry use
  this same generator and always choose a new directory.

## Sample Data

`_sample_data.py` contributes:

- `VIPP synthetic volume`: grayscale `ZYX`;
- `VIPP synthetic multichannel volume`: `CZYX`;
- `VIPP synthetic time-lapse multichannel`: `TCZYX`;
- `VIPP synthetic measurement summary`: `TYX` time-series objects with known
  counts and areas;
- `VIPP synthetic object morphology`: `YX` separated circle, ellipse,
  rectangle, and concave objects for derived morphology validation;
- `VIPP synthetic 3D mesh morphology`: anisotropic `ZYX` sphere-like,
  ellipsoid, cuboid, concave dumbbell, and tiny objects for mesh morphology
  validation;
- `VIPP synthetic skeleton network`: sparse `ZYX` network with known endpoints,
  branches, a junction, a short spur, a separate component, and an isolated
  voxel;
- `VIPP synthetic advanced skeleton network`: two-timepoint `TZYX` sparse
  skeleton network with looped components, many junctions, separate fragments,
  terminal spurs, isolated voxels, and anisotropic spatial calibration.
- `VIPP synthetic colocalization`: two-channel `CZYX` volume with partial
  overlap, single-channel objects, offset puncta, gradients, and noise;
- `VIPP synthetic deconvolution image`: blurred/noisy `YX` object image for
  PSF-aware restoration review;
- `VIPP synthetic measured PSF`: compact `YX` measured-PSF-style kernel for
  `Prepare / Validate PSF`.
- `VIPP synthetic 3D deconvolution volume`: blurred/noisy anisotropic `ZYX`
  volume for volumetric PSF-aware restoration review;
- `VIPP synthetic 3D measured PSF`: compact `ZYX` measured-PSF-style kernel for
  3D `Prepare / Validate PSF`.

The samples include OME-NGFF-like `multiscales` metadata and VIPP hint keys.
The time-lapse multichannel sample is preferred as the default when present.

## Tests

Tests live in `src/napari_vipp/_tests/`.

The suite covers pure operations and metadata, graph invariants, workflow
persistence, generated Python execution, I/O round trips, examples, previews,
and focused pytest-qt widget behavior. Keep contract and algorithm tests in the
headless core where possible; reserve widget tests for behavior that genuinely
depends on Qt orchestration.

Useful commands:

```bash
python -m npe2 validate src/napari_vipp/napari.yaml
python -m ruff check .
python -m pytest -q
python scripts/launch_vipp_sample.py
python scripts/launch_vipp_label_workflow.py
```

Focused architecture and scientific-boundary checks:

```bash
python -m pytest -q \
  src/napari_vipp/_tests/test_architecture.py \
  src/napari_vipp/_tests/test_snapshots.py \
  src/napari_vipp/_tests/test_execution.py \
  src/napari_vipp/_tests/test_source_identity.py \
  src/napari_vipp/_tests/test_file_sources.py \
  src/napari_vipp/_tests/test_axis_semantics.py \
  src/napari_vipp/_tests/test_grid.py \
  src/napari_vipp/_tests/test_diagnostics.py \
  src/napari_vipp/_tests/test_diagnostic_workers.py \
  src/napari_vipp/_tests/test_batch.py
```

Add a Qt/widget test only for behavior that genuinely crosses the napari or Qt
boundary. A pure operation change should normally cover numerical reference
behavior, invalid inputs, dtype/range behavior, read-only input preservation,
metadata propagation, and any affected workflow/export contract before a UI
test is considered.

## Known Gaps And Design Notes

Implemented now:

- large pan/zoom graph with draggable nodes;
- node add/connect/delete workflows;
- slot-aware multi-input connections;
- true multi-output graph nodes with per-port wiring, including dynamic port
  counts (e.g. `Split Channels` emits one port per channel);
- Image Data grouping with source, output, axis/region, channel/composite,
  utility, math, and logic nodes, including `Reorder Axes`, `Split Axis`,
  `Assign Channel Colors`, configurable `Composite → RGB`, and `Split
  Channels`;
- OME-NGFF-inspired metadata propagation;
- per-node and global preview controls;
- slice/stack/log histograms;
- guarded `New workflow...` action that resets to one unbound Image Source
  node;
- grouped `Open example...` workflow chooser for bundled templates, plus
  workflow JSON save/load for user files;
- Python export;
- shared OME-TIFF, ImageJ TIFF, conventional TIFF, OME-Zarr 0.4/0.5, NumPy,
  and 2D common raster import/export;
- graph-aware OME-Zarr analysis dataset export with reference image plus label
  groups;
- adaptive TIFF/OME-Zarr image selection and stored collection-binding intent;
- deterministic local collection planning with saved batch configuration,
  explicit output declarations, collision policies, resilient per-item
  execution, and checkpointed provenance records;
- first-class labels, connected-component labeling, volume filtering,
  border-object clearing, axis-aware hole filling and small-object removal,
  sequential relabeling, and label-volume distribution controls;
- first-class table outputs, basic label-object measurement, CSV/TSV table
  saving, and generated-script table output saving;
- manual/cached execution for expensive table-producing nodes via
  `OperationSpec.execution_policy = "manual"`. In the interactive widget,
  manual nodes skip ordinary live recomputation, expose `Calculate`/
  `Recalculate` on the node card and inspector, keep their last cached output
  available downstream when stale, and show `not calculated`, `running`,
  `ready`, `stale`, or `error` state. Manual nodes can opt into the persisted
  private `_vipp_auto_recalculate` setting from the inspector; when enabled,
  the widget includes affected manual nodes in ordinary dirty reruns and hides
  the explicit recalculate button. Private `_vipp_*` settings are filtered out
  before operation functions run. Headless `PrototypePipeline.run()` and
  generated Python export still calculate manual nodes by default so batch
  output is deterministic and does not depend on UI caches. Workflow JSON
  persists node settings but does not serialize cached arrays or tables;
- generic skeletonization, skeleton-network and skeleton-branch measurement
  tables, branch-summary distributions, normalized connectedness summaries,
  keypoint masks, RGB graph overlays, branch/component labels, and short-branch
  pruning;
- first-pass 3D mesh morphology tables using marching cubes and convex hulls;
- standard maximize behavior for detached VIPP windows.

Architectural seams still present:

- `_widget.py` remains a large composition facade, not a completed controller
  decomposition. Inspector assembly, graph command handling, source inspection
  and caching, dirty-run scheduling, diagnostic cache/submission/result
  coordination, generated-layer presentation, and application command wiring
  still live there. Worker adapters themselves now live in
  `ui/diagnostic_workers.py`. Extract only cohesive responsibilities with
  independent tests and explicit dependencies.
- Small synchronous runs call `PrototypePipeline.run()` directly, while
  background runs use `core.execution`. Converging on one application execution
  service would reduce duplicated result/error policy, but must preserve the
  current low-latency path and Qt tests.
- The exact colocalization scatter-density calculation remains in
  `ui/plots.py`; unlike the other exact reductions in `core/diagnostics.py`, it
  is not yet reusable from a headless diagnostic caller.
- `core/operations.py` and `core/pipeline.py` are still large operation and
  registry modules. Splitting them by scientific domain could improve
  discoverability, but only with registry parity, import compatibility,
  metadata, export, workflow-golden, and numerical tests in place.
- Some private names are re-exported through `_widget.py` for transitional test
  and compatibility coverage. Remove those shims only after callers use the
  owning `ui/` or `core/` module directly.
- Atomic persistence is per artifact, not transactional across a workflow,
  batch config, runner, manifests, and sidecars. Batch output promotion is also
  atomic per output rather than as an all-output set; partial publication is
  recorded explicitly when a later promotion fails.

Still incomplete or deliberately future-facing:

- calibrated extended-length variants outside the mesh-specific table;
- specialist domain-specific network metrics and broader cooperative
  cancellation/percentage progress inside long manual calculations;
- OME-Zarr pyramids, label colors/properties, and HCS plate/well/field browsing;
- operation-level lazy execution, remote URI reads, and collection batch
  execution beyond the first-pass local folder UI;
- richer channel/probe naming and colour metadata from real microscopy files;
- semantic-axis batch iteration and HCS traversal;
- plugin/template generation for arbitrary new analysis nodes.

When continuing work, prefer this order: state the scientific contract,
implement and test headless behavior in `core/`, add metadata/grid/provenance
handling, put reusable Qt behavior in `ui/`, and use `_widget.py` only to compose
those pieces with napari and `_graph.py`.
