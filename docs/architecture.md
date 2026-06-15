# napari-vipp Architecture Reference

This document is a developer handoff map for the current `napari-vipp`
prototype.

Last reviewed: 2026-06-15

It reflects the live codebase after slot-aware multi-input and multi-output
graphs, workflow persistence, Python export, metadata panels, first-class label
images, label-volume filtering, and detachable-window fixes.

For product framing and longer-range ideas, see [README.md](../README.md) and
[planning.md](planning.md).

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
  core/preview.py     thumbnail and fluorescence composite reduction
  core/workflow.py    JSON workflow save/load
  core/export.py      runnable Python script export
```

Boundary rule: files under `core/` should stay free of `qtpy` and `napari`
imports. They should remain usable from a headless exported/batch context.

## Repository Map

```text
src/napari_vipp/
  __init__.py          version shim
  napari.yaml          npe2 widget and sample-data manifest
  _widget.py           main dock widget and inspector orchestration
  _graph.py            QGraphicsView node canvas, ports, wires, node cards
  _theme.py            category colour tokens
  _sample_data.py      synthetic ZYX, CZYX, and TCZYX fluorescence samples
  core/
    pipeline.py        graph model, NODE_LIBRARY, executor
    operations.py      pure processing kernels
    metadata.py        ImageState and AxisMetadata propagation
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
| `OutputSpec` | Declares one output port of a node: `name`, `output_type`, and optional display `title`. |
| `OperationSpec` | Declares a node type in `NODE_LIBRARY`: id, title, category, subcategory, input/output types, parameters, function, `max_inputs`, optional static `outputs`, optional dynamic `output_factory`, and whether outputs preserve the connected input type. |
| `GraphNode` | A node instance with a stable id, operation id, resolved display/type fields, mutable `params`, and `max_inputs`. |
| `GraphConnection` | A directed edge from `source_id` to `target_id`, including the target input slot as `target_port` and the source output slot as `source_port`. |
| `ConnectionResult` | Returned by `connect()`: success flag, message, created connection, and any replaced/removed connections. |
| `SourcePayload` | Data, metadata, and name injected into source nodes. |

The executable graph is `PrototypePipeline`.

Key behaviours:

- `reset_starter_graph()` creates the starter graph:
  `Image Source -> Gaussian Blur -> Otsu Threshold`.
- `add_node()` creates node instances from `NODE_LIBRARY`.
- `connect(source_id, target_id, target_port=None, source_port=0)` enforces type
  compatibility (using the chosen source output port's type), rejects cycles,
  respects `max_inputs`, and stores the selected target and source slots. For
  multi-input nodes, omitted `target_port` auto-fills the first free slot;
  connecting to an occupied slot replaces that slot only. `source_port` selects
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
are complete. For each node it stores the primary output in `outputs[node_id]`
and `output_states[node_id]` (the port-0 value, for backward compatibility) plus
the full per-port lists in `node_outputs[node_id]` and
`node_output_states[node_id]`. Downstream inputs are resolved by
`(source_id, source_port)` so a node can pull from any output port of its source.
Source nodes use `SourcePayload`s or the toolbar-selected napari layer.
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
  Morphological Gradient, Fill Holes, Volume Filter
- `Label Operations`: Label Connected Components, Filter Labels By Volume,
  Clear Border Objects, Relabel Sequential

`labels` is a first-class graph type for non-negative integer object IDs with
zero as background. It is distinct from a boolean `mask` and an integer
intensity `image`. Label outputs inspect and pin as napari Labels layers, retain
their integer IDs through TIFF/NumPy saving, and connect only to label-aware or
generic array inputs.

The spatial label/mask operations expose `Auto from axes`, `2D YX`, and
`3D ZYX` spatial modes. Auto mode resolves from carried axis metadata and stores
the resolved spatial dimensionality for Python export. Leading non-spatial axes
are processed independently, so `TCZYX` labels are calculated per timepoint and
channel while each `ZYX` block is treated as one volume. When a connected input
has only two spatial axes, the inspector omits the invalid `3D ZYX` choice.

`Clear Border Objects` accepts only masks or labels, preserves the connected
semantic output type and retained label IDs, and supports a pixel/voxel border
buffer.

The current `Fill Holes` morphology node predates this shared spatial contract:
it calls `scipy.ndimage.binary_fill_holes` on the complete input array, fills
all enclosed holes, and has no parameters. The planned revision keeps the same
operation name/id for workflow compatibility while adding a maximum hole size,
2D-per-slice versus 3D-volume processing, and input-aware hiding of the 3D
choice for true 2D inputs.

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
- kind, such as intensity image, binary mask, RGB image, or multi-channel image;
- bit depth, value range/pattern, and memory estimate;
- metadata source;
- source name and operation history.

Source metadata comes from, in order of preference:

1. carried `vipp_image_state`;
2. OME or OME-NGFF-like `multiscales` metadata;
3. explicit layer axis hints such as `axes`, `axis_order`, or `vipp_axis_order`;
4. inferred axes from shape, explicitly labelled as inferred.

The inspector uses `metadata_table_rows()` and `metadata_history_items()`. Node
cards use `format_compact_metadata()`. Metadata is useful for UI labels,
parameter bounds, preview slicing, histogram slicing, ImageJ TIFF export, and
future OME-Zarr work.

## Preview And Histograms

`core/preview.py` reduces arbitrary output arrays to displayable thumbnails.

`make_preview(data, mode, current_step, state, channel_colors=None)` supports:

- `Slice`: reduce non-spatial axes by current napari dims or midpoint;
- `MIP`: project spatial axes where appropriate;
- `Off`: skip preview generation;
- state-aware channel handling when `ImageState` is available;
- pseudo-colour fluorescence composites for multichannel data;
- Combine Channels display colours from `channel_colors`.

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

## Graph UI

`_graph.py` owns the QGraphicsView/QGraphicsScene node canvas. It is
presentational: it emits signals, and `VippWidget` mutates the pipeline model.

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

Input is represented both by the historical toolbar input selector and by
explicit graph-level Image Source nodes. Image Source supports:

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
- Unknown operations are skipped on load.
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
- ImageJ-oriented TIFF output for images/masks and standard TIFF for integer
  labels;
- first-class labels, connected-component labeling, volume filtering,
  border-object clearing, sequential relabeling, and label-volume distribution
  controls;
- standard maximize behavior for detached VIPP windows.

Still incomplete or deliberately future-facing:

- non-image outputs such as measurement tables;
- OME-Zarr/OME-NGFF import/export as first-class IO, beyond internal metadata
  inspiration;
- richer channel/probe naming and colour metadata from real microscopy files;
- batch execution UI beyond the exported Python script;
- plugin/template generation for arbitrary new analysis nodes.

When continuing work, prefer this order: implement data behaviour in `core/`,
add metadata propagation, add tests, and only then expose controls in `_widget.py`
or visual affordances in `_graph.py`.
