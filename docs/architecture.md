# napari-vipp Architecture Reference

This document is a developer handoff map for the current `napari-vipp`
prototype.

Last reviewed: 2026-07-01

It reflects the live codebase after slot-aware multi-input and multi-output
graphs, workflow persistence, Python export, metadata panels, first-class label
images, table outputs, label-volume filtering, and detachable-window fixes.

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

The implementation deliberately separates a headless core from the Qt/napari
UI:

```text
napari host
  VippWidget (_widget.py)
    - NodePalette and parameter inspector
    - PipelineGraphView (_graph.py)
    - metadata, histogram, inspect, pin, save/export actions

headless core
  core/pipeline.py    graph model, node library, executor
  core/operations.py  pure NumPy/scikit-image/scipy node functions
  core/metadata.py    OME-NGFF-inspired ImageState propagation
  core/tables.py      TableData/TableState and CSV/TSV table saving
  core/io/            normalized OME/TIFF/Zarr/raster/NumPy readers and writers
  core/preview.py     thumbnail and fluorescence composite reduction
  core/workflow.py    JSON workflow save/load
  core/export.py      runnable Python script export
```

Boundary rule: files under `core/` should stay free of `qtpy` and `napari`
imports. They should remain usable from a headless exported/batch context.

## Repository Map

```text
src/napari_vipp/
  __init__.py          package version
  napari.yaml          npe2 widget and sample-data manifest
  _widget.py           main dock widget and inspector orchestration
  _graph.py            QGraphicsView node canvas, ports, wires, node cards
  _theme.py            category colour tokens
  _sample_data.py      synthetic ZYX, CZYX, TCZYX, TYX, and skeleton samples
  core/
    pipeline.py        graph model, NODE_LIBRARY, executor
    operations.py      pure processing kernels
    metadata.py        ImageState and AxisMetadata propagation
    io/
      model.py         ImageDataset, SourceInspection, and series records
      registry.py      format detection and shared read/write entry points
      tiff.py          OME-TIFF, ImageJ TIFF, and conventional TIFF
      raster.py        common PNG/JPEG/BMP/GIF/WebP/TGA/PNM import and 2D export
      ome_zarr.py      OME-Zarr 0.4/0.5 image support
      numpy_io.py      NPY/NPZ support
    tables.py          TableData, TableState, CSV/TSV writer
    preview.py         slice/MIP/RGB thumbnail generation
    workflow.py        workflow JSON persistence
    export.py          headless Python export
  nodes/               reserved placeholder package
  _tests/              pytest and pytest-qt tests

scripts/launch_vipp_sample.py
scripts/launch_vipp_label_workflow.py
examples/otsu-red-channel-labels.json
docs/planning.md
docs/architecture.md
docs/node-roadmap.md
```

## Core Model

`core/pipeline.py` is the centre of the non-UI system.

Important dataclasses:

| Type | Purpose |
| --- | --- |
| `ParameterSpec` | Declares one parameter: name, label, kind, default, range, step, decimals, and optional choices. |
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
- `output_ports(node_id)` returns the node's resolved `OutputSpec`s: the static
  `outputs` when declared, a single default `out` port otherwise, or — for nodes
  with an `output_factory` — one port per channel discovered on the last run
  (defaulting to three before the node has processed an image). Type-preserving
  multi-output nodes such as `split_channels` resolve each port's type from the
  connected upstream port, allowing a split mask channel to feed mask-only
  operations.
- `trim_invalid_output_connections(node_id)` removes downstream edges whose
  stored `source_port` is now outside a dynamic node's current output count.
- `topological_order()` supports export by returning a source-first order.
- `restore_graph()` rebuilds the model from workflow JSON.

Execution happens in `run(...)`. It repeatedly runs nodes whose upstream sources
are complete. For each node it stores the primary port-0 output in
`outputs[node_id]` and `output_states[node_id]` for convenient single-output
access, plus the full per-port lists in `node_outputs[node_id]` and
`node_output_states[node_id]`. Downstream inputs are resolved by
`(source_id, source_port)` so a node can pull from any output port of its source.
Source nodes use `SourcePayload`s or the toolbar-selected napari layer. File
sources are loaded through `core.io.read_image()` and inject the normalized
reader-built state directly, avoiding metadata reparsing.
Single-input nodes call their pure operation function with one input. Multi-input
nodes gather inputs in `target_port` order and require all ports from
`0..input_count-1` to be connected before running.

The widget can dispatch eligible graph runs to a single background worker
thread. The active worker reports node-start events for the global busy
indicator and per-node processing state. Operations that accept a
`ProgressContext` can also report determinate progress and check a worker-owned
cancel event between internal work units. This is currently wired for
rolling-ball/subtract-background block processing, `rescale_axes`, and 3D mesh
morphology label loops. The toolbar `Cancel` control clears queued reruns,
requeues in-flight dirty nodes, requests cooperative cancellation, and ignores
stale worker results. It still cannot forcibly interrupt a NumPy/SciPy/
scikit-image call already executing inside that worker.

Special execution cases:

- `save_output` receives the upstream `image_state` so TIFF/ImageJ metadata can
  be written where possible.
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

Histogram-based global threshold nodes default to
`Threshold uses = Stack histogram`, meaning one cutoff is computed from the
whole grayscale input and applied to the full image. On stack inputs the
inspector exposes `Slice histogram` for per-plane cutoff calculation; that still
produces a full-stack mask, it only changes which histogram is used to compute
each cutoff. On 2D inputs the control is hidden because stack versus slice is
not meaningful. Fixed `Binary Threshold` and local threshold nodes do not expose
this control. Global automatic threshold nodes also show the selected input
histogram with a marker at the computed cutoff.

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
morphology operations: users must be told that the node works in the current
`YX` plane and that `Reorder Axes` should be used first when a different plane
or slice axis is intended.
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

To add a node:

1. Implement a pure function in `core/operations.py`.
2. Import it in `core/pipeline.py`.
3. Add an `OperationSpec` to `NODE_LIBRARY`.
4. Add or update metadata propagation rules in `core/metadata.py` if the node
   changes axes, dtype semantics, mask/intensity kind, or channel structure.
5. Add dynamic bounds or custom inspector UI in `_widget.py` only if the generic
   controls are insufficient.
6. Add tests, usually in `test_operations.py`, `test_widget.py`, and possibly
   `test_preview.py`, `test_workflow.py`, or `test_export.py`.

## Metadata

`core/metadata.py` carries OME-NGFF-inspired state with every graph output. This
is intentionally stricter than "just an array shape" because bioimage workflows
need to know which axes are time, channels, z, y, and x.

`ImageState` records:

- shape and dtype;
- semantic axes as `AxisMetadata`;
- axis type, scale, unit, and translation/origin where available;
- optional source-axis indices that map source and derived axes back to the
  correct napari viewer sliders, including right-aligned layers with fewer axes
  than the current viewer;
- kind, such as intensity image, binary mask, RGB image, or multi-channel image;
- bit depth, value range/pattern, and memory estimate;
- metadata source;
- source name and operation history.
- normalized channel names, colors, fluorophore/wavelength fields;
- preserved acquisition description/date and instrument reference strings;
- source URI, format, series index/name, and source UUID.

`TableState` records row and column counts, stable column names, column units
where known, source name, table/measurement kind, metadata source, and operation
history. Node cards use the table summary; the inspector shows table metadata
and a bounded row preview.

Source metadata comes from, in order of preference:

1. carried `vipp_image_state`;
2. OME or OME-NGFF-like `multiscales` metadata;
3. explicit layer axis hints such as `axes`, `axis_order`, or `vipp_axis_order`;
4. inferred axes from shape, explicitly labelled as inferred.

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

`Reorder Axes` transposes the array and then reinterprets spatial axis names by
output position. This makes downstream nodes treat a Y/Z swap like a rotated
volume: the moved original Z data can become the effective Y axis, and
state-aware thumbnails keep following napari slice sliders. Channel and time
axes still move with their data, and physical scale/unit metadata follows the
moved data axis.

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
is used. In auto mode, `Composite → RGB` preserves declared or unlabelled
channel-last RGB/RGBA images as true RGB, but ordinary fluorescence channel
stacks are blended by carried pseudo-colours. A yellow channel therefore writes
to red and green, while a cyan channel writes to green and blue. Manual
red/green/blue selectors still force single-channel RGB plane mapping. Split
Channels exposes a `Thumbnail channel` inspector parameter that chooses which
output port appears on the node card; it does not alter the generated channel
outputs or downstream port wiring.

For generated napari inspect/pin layers, 2D RGB outputs use napari's native
`rgb=True` image layer. Volumetric RGB outputs are displayed as synchronized
additive red/green/blue image layers because napari's scalar-field 3D status
and rendering path does not reliably handle hidden RGB axes.

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

Histograms live mostly in `_widget.py` through `HistogramPlot` and helper
functions near the bottom of the file. The general selected-output histogram
supports slice-vs-stack and linear-vs-log display. Multichannel histograms are
drawn as separate series using fluorescence-style colours.

Cutoff-style nodes listed in `INPUT_HISTOGRAM_OPERATIONS` also show an
`Input Histogram` above the general output histogram. It has its own
`Histogram uses` slice/stack selector, hidden when the connected input has no
meaningful stack axis, and its markers are driven by the node parameters.

When `Filter Labels By Volume` is selected, a second histogram above the
general histogram shows the object-volume distribution from the unfiltered
label input. Its `Log volume axis` toggle defaults on, and it draws live
minimum and enabled maximum threshold markers. Because the distribution uses
the input labels, it remains stable while the filter removes objects.

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

`_widget.py` owns the `VippWidget` dock widget and is the orchestration layer.
It is currently large because it contains:

- the 3-pane layout and side-panel toggles;
- the node palette and fuzzy search;
- generic parameter controls;
- custom `ImageSourceControl`;
- custom `AxisSliceControl` for keep/remove/range axis subsetting;
- custom `ReorderAxesControl` for drag-reordering output axes while storing a
  compact workflow order string;
- custom Combine Channels controls for input count and channel colour identity;
- source resolution from napari layers, files, or samples;
- graph editing callbacks;
- pipeline execution and debouncing;
- thumbnails, metadata table, history, and histogram updates;
- inspect and pin layer management;
- quick save, Save Image node support, workflow save/load, and Python export.

The main update loop is `run_pipeline()`:

1. Resolve graph-level Image Source nodes.
2. Run `PrototypePipeline.run(...)`.
3. Refresh dynamic parameter bounds and re-run once if those bounds changed.
4. Hide/restores managed input layers as needed for inspection.
5. Refresh node thumbnails, inspect layer, pinned layer, metadata panel,
   histogram, and status text.

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
- Workflow version 1 also stores graph notes and VIPP UI metadata. Inspector
  metadata is always written when saving through the widget and records the
  selected node plus right-panel visibility. Per-node thumbnail visibility is
  written only when `Save thumbnail visibility in workflows` is enabled.
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
- `VippWidget.load_workflow_file()` provides the non-dialog load path used by
  the example launcher.

Python export:

- `core/export.py` emits a runnable script with `run_pipeline()`,
  `batch_process()`, image load/save helpers, and an argparse entry point.
- The generated folder batch helper supplies one primary image source; workflows
  with additional independent sources require manual binding in the script.
- Multi-input calls use input sources ordered by `target_port`.
- Multi-output sources are assigned a list; downstream calls index the right
  port (for example `split_channels_1[1]` for the second channel) using
  `source_port`.
- `combine_channels` exports using stored `channel_axis`; if a composite node
  has not run yet and lacks that derived axis, export emits a NOTE comment.

Collection batch UI:

- `VippWidget._run_collection_batch()` clones the current workflow model before
  execution, so folder processing does not replace the live canvas outputs with
  the last processed file.
- `CollectionBatchDialog` lists every `Image Source` node as a possible batch
  source binding. Blank rows keep their normal fixed layer/file/sample source.
- `BatchSourceBinding` stores the node id, display title, folder, and glob
  pattern for one collection-bound source. `BatchItem` stores a stable item
  index/id plus the source path assigned to every bound source. `BatchPreviewRow`
  is the non-executing dry-run representation shown by `Preview batch`.
- When several sources are bound, matched paths are sorted per source and paired
  by position. All bound sources must match the same number of files. The first
  bound source becomes the primary source for default naming.
- `Batch Output` nodes are the authoritative save markers. They pass data
  through during normal graph execution and provide tag, format, subfolder,
  filename-template, and overwrite controls for batch saves.
- Filename templates can use batch-aware fields including `{batch_id}`,
  `{batch_index}`, `{source_name}`, `{source_stem}`, `{primary_source_stem}`,
  `{tag}`, `{node_id}`, and `{node_title}`.
- If a graph has no `Batch Output` nodes, terminal graph outputs are saved as a
  compatibility fallback. Image-like fallback outputs use the selected batch
  image format; table fallback outputs are saved as CSV.
- The dialog can write `vipp_batch_workflow.json` and
  `vipp_batch_pipeline.py` beside the results for reproducibility.
- This is a local-folder first pass. Per-item provenance manifests,
  plate/well/field identities, HCS traversal, and semantic-axis iteration remain
  future work.

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

Current test split:

- `test_operations.py`: processing functions and save behaviour.
- `test_graph.py`: Qt graph canvas behaviour, node movement, ports,
  connection feedback, deletion.
- `test_preview.py`: slice/MIP/multichannel thumbnail generation, including
  requested Combine Channels colours.
- `test_sample_data.py`: sample data shapes and metadata hints.
- `test_widget.py`: pytest-qt tests for widget workflows, metadata, histogram,
  pinning, source controls, graph interactions, Combine Channels UI, and save
  actions.
- `test_workflow.py`: workflow JSON save/load and target-port preservation.
- `test_export.py`: generated Python syntax and execution.
- `test_multi_output.py`: multi-output ports, `source_port` routing, split
  metadata, and persistence/export of multi-output sources.
- `test_example_workflow.py`: checked-in label workflow loading and execution.

Useful commands:

```bash
python -m npe2 validate src/napari_vipp/napari.yaml
python -m ruff check .
python -m pytest -q
python scripts/launch_vipp_sample.py
python scripts/launch_vipp_label_workflow.py
```

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

Still incomplete or deliberately future-facing:

- calibrated extended-length variants outside the mesh-specific table;
- specialist domain-specific network metrics and broader cooperative
  cancellation/percentage progress inside long manual calculations;
- OME-Zarr pyramids, label colors/properties, and HCS plate/well/field browsing;
- operation-level lazy execution, remote URI reads, and collection batch
  execution beyond the first-pass local folder UI;
- richer channel/probe naming and colour metadata from real microscopy files;
- richer batch execution UI for paired source collections, output templates, and
  per-item provenance;
- plugin/template generation for arbitrary new analysis nodes.

When continuing work, prefer this order: implement data behaviour in `core/`,
add metadata propagation, add tests, and only then expose controls in `_widget.py`
or visual affordances in `_graph.py`.
