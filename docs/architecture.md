# napari-vipp Architecture Reference

This document is a developer handoff map for the current `napari-vipp`
prototype.

Last reviewed: 2026-06-16

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
pin mask or label outputs as overlays.

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
  core/io/            normalized OME/TIFF/Zarr/NumPy readers and writers
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
  _sample_data.py      synthetic ZYX, CZYX, and TCZYX fluorescence samples
  core/
    pipeline.py        graph model, NODE_LIBRARY, executor
    operations.py      pure processing kernels
    metadata.py        ImageState and AxisMetadata propagation
    io/
      model.py         ImageDataset, SourceInspection, and series records
      registry.py      format detection and shared read/write entry points
      tiff.py          OME-TIFF, ImageJ TIFF, and conventional TIFF
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
| `GraphConnection` | A directed edge from `source_id` to `target_id`, including the target input slot as `target_port` and the source output slot as `source_port`. |
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

## Node Library

`NODE_LIBRARY` in `core/pipeline.py` is the source of truth for available nodes.
The palette grouping comes from each `OperationSpec.category` and
`OperationSpec.subcategory`.

The current high-level groups are:

- `Image Data`
  - `Source & Output`: Image Source, Save Image
  - `Axes & Regions`: Crop Stack, Select Axis Slice
  - `Channels & Composites`: Extract Channel, Combine Channels, Split Channels,
    Composite → RGB
  - `Type & Scaling`: Convert Dtype, Rescale Intensity, Normalize, Clip
  - `Math & Logic`: Calculate New Image, Add, Subtract, Ratio, Mask Image,
    Logical AND, Logical OR, Logical XOR, Invert
- `Contrast`: Contrast Stretching, Gamma Correction
- `Filtering`: Average Blur, Gaussian Blur, Gaussian Blur 3D, Median Filter,
  Bilateral Filtering
- `Projection`: Maximum Projection
- `Segmentation`: Otsu, Triangle, Binary, Adaptive Mean, Adaptive Gaussian
  thresholding
- `Morphology`: Dilation, Erosion, Opening, Closing, Top Hat, Black Hat,
  Morphological Gradient, Fill Holes, Remove Small Objects, Skeletonize
- `Label Operations`: Label Connected Components, Filter Labels By Volume,
  Clear Border Objects, Relabel Sequential
- `Measurements`: Measure Objects, Measure Objects + Intensity, Analyze
  Skeleton, Merge Tables, Add Metadata Columns

`labels` is a first-class graph type for non-negative integer object IDs with
zero as background. It is distinct from a boolean `mask` and an integer
intensity `image`. Label outputs inspect and pin as napari Labels layers, retain
their integer IDs through TIFF/NumPy saving, and connect only to label-aware or
generic array inputs.

Connected-component labeling, label-volume filtering, relabeling, and Fill
Holes expose metadata-aware 2D/3D spatial processing. Auto mode resolves from
carried axis metadata and stores the resolved spatial dimensionality for Python
export. Leading non-spatial axes are processed independently, so `TCZYX` data
is processed per timepoint and channel while each `ZYX` block is treated as one
volume.

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

`Measure Objects` accepts labels and outputs a `table` rather than an image.
The first implementation uses `skimage.measure.regionprops_table` for one row
per labeled object. It measures label ID, pixel/voxel area or volume,
calibrated physical area or volume when spatial scale metadata is available,
centroid, bounding box, equivalent diameter, extent, and Euler number. Leading
time/channel or other non-spatial axes become index columns so repeated label
IDs remain distinguishable across frames.

`Measure Objects + Intensity` is the first named heterogeneous-input node. It
declares separate `Labels` and `Intensity image` input slots, requires matching
array shapes in the first implementation, and emits the basic morphology table
plus per-label mean, minimum, maximum, sum, and standard deviation intensity.
The same `InputSpec` mechanism is intended for watershed, colocalization, and
other heterogeneous-input nodes.

`Merge Tables` accepts a variable number of `table` inputs and emits a joined
`table`. The `input_count` parameter controls visible/required input ports.
`auto` join keys use shared stable identity columns such as `t_index`,
`c_index`, `z_index`, `label_id`, `object_id`, or `component_id`; if no
identity columns overlap and all tables have the same row count, the node falls
back to row-position joining. Duplicate non-key columns from later tables are
suffixes such as `_table2` so no source column is silently overwritten.

`Add Metadata Columns` accepts one `table` input and appends constant
user-supplied `name=value` columns, intended for treatment, replicate, batch,
condition, or source annotations before export. Existing columns are protected
unless the node's overwrite parameter is explicitly set to `yes`.

`Skeletonize` accepts masks and produces a binary skeleton mask in metadata-aware
2D or 3D spatial blocks. `Analyze Skeleton` accepts a skeleton mask and outputs
a `table` with one row per connected skeleton component. It reports skeleton
voxel count, endpoint voxels, junction voxels, isolated nodes, simplified
graph node/edge counts, voxel-graph edge count, cycle count, per-block
connected-component context, and skeleton length in pixels/voxels plus
calibrated physical units when spatial scale metadata is available. The first
graph analyzer deliberately stays generic; mitochondrial-specific
fragmentation, mesh/surface, and domain-normalized connectivity measurements
should be separate specialist nodes.

The longer-term measurement architecture must support selectable measurement
families and merged per-object result tables for exploratory statistics such as
PCA and treatment-group separation. See
[mitomorph-feature-parity.md](mitomorph-feature-parity.md).

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

The inspector uses `metadata_table_rows()` and `metadata_history_items()`. Node
cards use `format_compact_metadata()`. Metadata is useful for UI labels,
parameter bounds, preview slicing, histogram slicing, and all shared writers.
Table outputs use the same metadata helpers but do not participate in image
preview slicing or histograms. The inspector therefore hides the per-node
thumbnail toggle for table-only nodes such as measurement outputs.

`core/io/` is the only format-specific boundary. `inspect_image_source()`
discovers selectable image items without loading full TIFF pixel data.
`read_image()` returns an `ImageDataset`; OME-Zarr data and multiscale levels
remain Dask-backed. `write_image()` dispatches explicit OME-Zarr, OME-TIFF,
ImageJ TIFF, TIFF, and NPY formats. Auto mode selects NPY by suffix, OME-Zarr
for `.zarr`, OME-TIFF for `.ome.tif[f]`, conventional TIFF for label images,
and OME-TIFF for other TIFF outputs.

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

Napari also right-aligns layers with fewer dimensions than the current viewer.
For example, a `CZYX` layer displayed in a 5D viewer is controlled by global
sliders 1, 2, 3, and 4. Napari-layer source payloads therefore rewrite
`source_axis` with this offset before the graph runs. The widget subscribes to
both `dims.events.current_step` and `dims.events.point`, using a bound method
instead of an anonymous callback so the subscription remains alive.

Important nuance: Combine Channels colour assignment is display intent. The
underlying output stays a multichannel array with a channel axis. The assigned
colours affect thumbnails and graph-port presentation; RGB conversion remains a
separate operation (`Composite → RGB`) when a downstream RGB image is needed.

Histograms live mostly in `_widget.py` through `HistogramPlot` and helper
functions near the bottom of the file. The general selected-output histogram
supports slice-vs-stack and linear-vs-log display. Multichannel histograms are
drawn as separate series using fluorescence-style colours.

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
| `PipelineGraphView` | Canvas, pan/zoom, drag/drop node creation, wire creation/removal, selection, delete-key handling. |
| `NodeProxy` | Movable graphics item wrapping a `NodeCard`; owns the visual input/output ports (one or more output ports for multi-output nodes). |
| `NodeCard` | Embedded widget with category tint, title, thumbnail, compact metadata, and pin button when applicable. |
| `PortItem` | Input/output port circle with hover/drop feedback and optional accent colour/label. Multi-input and multi-output nodes have several ports. |
| `ConnectionItem` | Curved wire storing source id, target id, `target_port`, and `source_port`. |

Signals to `VippWidget`:

- `node_selected(node_id)`
- `node_delete_requested(node_id)`
- `pin_requested(node_id)`
- `node_create_requested(operation_id, scene_position)`
- `connection_requested(source_id, target_id, target_port, source_port)`
- `connection_removed(source_id, target_id, target_port)`
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
- custom Combine Channels controls for input count and channel colour identity;
- source resolution from napari layers, files, or samples;
- graph editing callbacks;
- pipeline execution and debouncing;
- thumbnails, metadata table, history, and histogram updates;
- inspect and pin layer management;
- quick save, Save Image node support, workflow save/load, and Python export.

The main update loop is `run_pipeline()`:

1. Resolve toolbar input and any graph-level Image Source nodes.
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

Input is represented both by the toolbar input selector and by explicit
graph-level Image Source nodes. Image Source supports:

- existing napari layer;
- file path (`.npy`, `.npz`, TIFF and other `skimage.io.imread`-readable files);
- bundled synthetic sample.

Quick output saving:

- The inspector has `Save selected output...`; it defaults to TIFF but allows
  `.npy`.
- `save_array_output()` writes TIFF as ImageJ hyperstacks when metadata allows.
- Binary masks are saved as 8-bit `0`/`255`, not `0`/`1`.
- Integer label images use standard TIFF rather than ImageJ TIFF so 32-bit IDs
  are preserved.

Graph-level output:

- `Save Image` passes its input through unchanged.
- If auto-save is enabled, it writes on every graph recompute.

Workflow persistence:

- `core/workflow.py` serializes nodes, params, connections including
  `target_port` and `source_port`, and canvas positions to JSON.
- Workflow version 1 does not yet store preview visibility, inspector state,
  graph notes, environment provenance, or YAML.
- Image Source paths and layer names are serialized as literal parameters;
  input files are not embedded and paths are not rebased for portable sharing.
- The loader requires the current type and version and rejects unknown
  operations, malformed records, duplicate node ids, invalid positions, and
  dangling or multiply occupied connections.
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

## Sample Data

`_sample_data.py` contributes:

- `VIPP synthetic volume`: grayscale `ZYX`;
- `VIPP synthetic multichannel volume`: `CZYX`;
- `VIPP synthetic time-lapse multichannel`: `TCZYX`.

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
- Image Data grouping with source, output, channels, conversion, math, and logic
  nodes, including the configurable `Composite → RGB` and `Split Channels`;
- OME-NGFF-inspired metadata propagation;
- per-node and global preview controls;
- slice/stack/log histograms;
- workflow JSON save/load;
- Python export;
- shared OME-TIFF, ImageJ TIFF, conventional TIFF, OME-Zarr 0.4/0.5, and NumPy
  import/export;
- graph-aware OME-Zarr analysis dataset export with reference image plus label
  groups;
- adaptive TIFF/OME-Zarr image selection and stored collection-binding intent;
- first-class labels, connected-component labeling, volume filtering,
  border-object clearing, axis-aware hole filling and small-object removal,
  sequential relabeling, and label-volume distribution controls;
- first-class table outputs, basic label-object measurement, CSV/TSV table
  saving, and generated-script table output saving;
- generic skeletonization and skeleton-network measurement tables;
- standard maximize behavior for detached VIPP windows.

Still incomplete or deliberately future-facing:

- intensity-aware measurements and property-based label filtering;
- skeleton QC feature masks, branch labels, pruning, and graph export;
- OME-Zarr pyramids, label colors/properties, and HCS plate/well/field browsing;
- operation-level lazy execution, remote URI reads, and collection batch
  execution;
- richer channel/probe naming and colour metadata from real microscopy files;
- batch execution UI beyond the exported Python script;
- plugin/template generation for arbitrary new analysis nodes.

When continuing work, prefer this order: implement data behaviour in `core/`,
add metadata propagation, add tests, and only then expose controls in `_widget.py`
or visual affordances in `_graph.py`.
