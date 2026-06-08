# napari-vipp Architecture Reference

This document is a developer handoff map for the current `napari-vipp`
prototype. It was reviewed against the live codebase after the slot-aware
Channel Composite work, workflow persistence, Python export, metadata panels,
histograms, and channel-colour preview updates.

For product framing and longer-range ideas, see [README.md](../README.md) and
[planning.md](planning.md).

## Current Shape

`napari-vipp` is a napari `npe2` plugin that embeds a node-graph image
processing workflow inside a dock widget. The graph is the main work surface:
users add processing nodes, wire outputs to inputs, tune parameters in the
inspector, view live thumbnails, inspect full-resolution outputs in napari, and
pin mask outputs as overlays.

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
docs/planning.md
docs/architecture.md
```

## Core Model

`core/pipeline.py` is the centre of the non-UI system.

Important dataclasses:

| Type | Purpose |
| --- | --- |
| `ParameterSpec` | Declares one parameter: name, label, kind, default, range, step, decimals, and optional choices. |
| `OperationSpec` | Declares a node type in `NODE_LIBRARY`: id, title, category, subcategory, input/output types, parameters, function, and `max_inputs`. |
| `GraphNode` | A node instance with a stable id, operation id, resolved display/type fields, mutable `params`, and `max_inputs`. |
| `GraphConnection` | A directed edge from `source_id` to `target_id`, including the target input slot as `target_port`. |
| `ConnectionResult` | Returned by `connect()`: success flag, message, created connection, and any replaced/removed connections. |
| `SourcePayload` | Data, metadata, and name injected into source nodes. |

The executable graph is `PrototypePipeline`.

Key behaviours:

- `reset_starter_graph()` creates the starter graph:
  `Image Source -> Gaussian Blur -> Otsu Threshold`.
- `add_node()` creates node instances from `NODE_LIBRARY`.
- `connect(source_id, target_id, target_port=None)` enforces type
  compatibility, rejects cycles, respects `max_inputs`, and stores the selected
  target slot. For multi-input nodes, omitted `target_port` auto-fills the first
  free slot; connecting to an occupied slot replaces that slot only.
- `disconnect(source_id, target_id, target_port=None)` removes either a specific
  slot connection or all matching source-target connections when no slot is
  supplied.
- `trim_invalid_connections(node_id)` removes connections whose stored slot is
  now outside the node's current input count.
- `topological_order()` supports export by returning a source-first order.
- `restore_graph()` rebuilds the model from workflow JSON.

Execution happens in `run(...)`. It repeatedly runs nodes whose upstream sources
are complete and stores both `outputs[node_id]` and `output_states[node_id]`.
Source nodes use `SourcePayload`s or the toolbar-selected napari layer.
Single-input nodes call their pure operation function with one input. Multi-input
nodes gather inputs in `target_port` order and require all ports from
`0..input_count-1` to be connected before running.

Special execution cases:

- `save_output` receives the upstream `image_state` so TIFF/ImageJ metadata can
  be written where possible.
- `channel_composite` derives its insertion channel axis from upstream metadata,
  stores that `channel_axis` back into `node.params`, and then stacks inputs in
  slot order.

## Node Library

`NODE_LIBRARY` in `core/pipeline.py` is the source of truth for available nodes.
The palette grouping comes from each `OperationSpec.category` and
`OperationSpec.subcategory`.

The current high-level groups are:

- `Image Data`
  - `Source & Output`: Image Source, Save Image
  - `Axes & Regions`: Crop Stack, Select Axis Slice
  - `Channels & Composites`: Extract Channel, Channel Composite, RGB Composite
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
- Channel Composite display colours from `channel_colors`.

Important nuance: Channel Composite colour assignment is display intent. The
underlying output stays a multichannel array with a channel axis. The assigned
colours affect thumbnails and graph-port presentation; RGB conversion remains a
separate operation (`RGB Composite`) when a downstream RGB image is needed.

Histograms live mostly in `_widget.py` through `HistogramPanel` and helper
functions near the bottom of the file. The inspector supports slice-vs-stack
histograms and linear-vs-log display. Multichannel histograms are drawn as
separate series using fluorescence-style colours.

## Graph UI

`_graph.py` owns the QGraphicsView/QGraphicsScene node canvas. It is
presentational: it emits signals, and `VippWidget` mutates the pipeline model.

Main classes:

| Class | Role |
| --- | --- |
| `PipelineGraphView` | Canvas, pan/zoom, drag/drop node creation, wire creation/removal, selection, delete-key handling. |
| `NodeProxy` | Movable graphics item wrapping a `NodeCard`; owns the visual input/output ports. |
| `NodeCard` | Embedded widget with category tint, title, thumbnail, compact metadata, and pin button when applicable. |
| `PortItem` | Input/output port circle with hover/drop feedback. Multi-input nodes have several input ports. |
| `ConnectionItem` | Curved wire storing source id, target id, and `target_port`. |

Signals to `VippWidget`:

- `node_selected(node_id)`
- `node_delete_requested(node_id)`
- `pin_requested(node_id)`
- `node_create_requested(operation_id, scene_position)`
- `connection_requested(source_id, target_id, target_port)`
- `connection_removed(source_id, target_id, target_port)`
- `status_message(text)`

Slot-aware connections are important. The graph must never treat all wires into
a multi-input node as one anonymous target. `target_port` is the contract
between the visual port, the pipeline model, workflow JSON, and Python export.

## Widget UI

`_widget.py` owns the `VippWidget` dock widget and is the orchestration layer.
It is currently large because it contains:

- the 3-pane layout and side-panel toggles;
- the node palette and fuzzy search;
- generic parameter controls;
- custom `ImageSourceControl`;
- custom `AxisSliceControl` for keep/remove/range axis subsetting;
- custom Channel Composite controls for input count and channel colour identity;
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
changes, such as Channel Composite colour identity, also call
`_update_thumbnails()` immediately because they affect display but not numeric
array values.

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

Graph-level output:

- `Save Image` passes its input through unchanged.
- If auto-save is enabled, it writes on every graph recompute.

Workflow persistence:

- `core/workflow.py` serializes nodes, params, connections including
  `target_port`, and canvas positions to JSON.
- Unknown operations are skipped on load.
- `PrototypePipeline.restore_graph()` rebuilds the graph model.

Python export:

- `core/export.py` emits a runnable script with `run_pipeline()`,
  `batch_process()`, image load/save helpers, and an argparse entry point.
- Multi-input calls use input sources ordered by `target_port`.
- `channel_composite` exports using stored `channel_axis`; if a composite node
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
  requested Channel Composite colours.
- `test_sample_data.py`: sample data shapes and metadata hints.
- `test_widget.py`: pytest-qt tests for widget workflows, metadata, histogram,
  pinning, source controls, graph interactions, Channel Composite UI, and save
  actions.
- `test_workflow.py`: workflow JSON save/load and target-port preservation.
- `test_export.py`: generated Python syntax and execution.

Useful commands:

```bash
python -m npe2 validate src/napari_vipp/napari.yaml
python -m ruff check .
python -m pytest -q
python scripts/launch_vipp_sample.py
```

## Known Gaps And Design Notes

Implemented now:

- large pan/zoom graph with draggable nodes;
- node add/connect/delete workflows;
- slot-aware multi-input connections;
- Image Data grouping with source, output, channels, conversion, math, and logic
  nodes;
- OME-NGFF-inspired metadata propagation;
- per-node and global preview controls;
- slice/stack/log histograms;
- workflow JSON save/load;
- Python export;
- ImageJ-oriented TIFF output for saved arrays.

Still incomplete or deliberately future-facing:

- true multi-output graph nodes;
- non-image outputs such as measurement tables;
- OME-Zarr/OME-NGFF import/export as first-class IO, beyond internal metadata
  inspiration;
- richer channel/probe naming and colour metadata from real microscopy files;
- batch execution UI beyond the exported Python script;
- plugin/template generation for arbitrary new analysis nodes.

When continuing work, prefer this order: implement data behaviour in `core/`,
add metadata propagation, add tests, and only then expose controls in `_widget.py`
or visual affordances in `_graph.py`.
