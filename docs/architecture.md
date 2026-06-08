# napari-vipp Architecture & Codebase Reference

This document is a developer/agent-oriented map of the `napari-vipp` codebase.
It explains how the pieces fit together so that anyone (human or AI) can quickly
get their bearings and continue building. For product framing and roadmap, see
[README.md](../README.md) and [docs/planning.md](planning.md).

> Status: research prototype. The graph model, metadata propagation, and Qt UI
> are functional. Workflow persistence (JSON save/load) and pipeline export
> (graph → runnable Python with batch processing) are implemented; true
> multi-output nodes are not yet implemented.

---

## 1. Big Picture

`napari-vipp` is a napari (`npe2`) plugin that provides an interactive
**node-graph image-processing workflow** inside a single dock widget. Users drag
processing nodes onto a pan/zoom canvas, wire outputs to inputs, tune
parameters, and see live thumbnails, metadata, and histograms. Selected outputs
can be "pinned" or "inspected" as real napari layers.

The design separates a **headless core** (pure data model + NumPy operations +
metadata) from the **Qt UI layer**. The core has no Qt or napari dependency,
which keeps the future "export to runnable Python" and batch-execution goals
achievable.

```
┌──────────────────────────────────────────────────────────────┐
│ napari (npe2 host)                                             │
│   └─ VippWidget  (dock widget, _widget.py)  ── orchestrator    │
│        ├─ PipelineGraphView (_graph.py)     ── Qt canvas/cards │
│        ├─ NodePalette / ParameterControls   ── Qt inspector    │
│        └─ HistogramPanel / metadata panels                     │
└───────────────────────────┬──────────────────────────────────┘
                            │ calls (no Qt below this line)
┌───────────────────────────▼──────────────────────────────────┐
│ core/ (headless)                                              │
│   pipeline.py   ── graph model + executor + node library      │
│   operations.py ── pure NumPy/scikit-image/scipy functions    │
│   metadata.py   ── OME-NGFF-inspired ImageState propagation   │
│   preview.py    ── thumbnail reduction (slice/MIP/composite)  │
└──────────────────────────────────────────────────────────────┘
```

**Key boundary rule:** anything under `core/` must stay free of `qtpy` and
`napari` imports. UI concerns live in `_widget.py`, `_graph.py`, `_theme.py`.

---

## 2. Repository Layout

```
src/napari_vipp/
├── __init__.py          Plugin version shim only.
├── napari.yaml          npe2 manifest: widget + sample-data contributions.
├── _widget.py           (~3000 lines) Main dock widget + all inspector UI.
├── _graph.py            (~810 lines) Qt graph canvas, node cards, ports, edges.
├── _theme.py            Category color/tint tokens shared by UI.
├── _sample_data.py      Synthetic ZYX / CZYX / TCZYX fluorescence samples.
├── core/
│   ├── pipeline.py      Graph model (nodes/connections), executor, NODE_LIBRARY.
│   ├── operations.py    Pure image-processing functions (the node "kernels").
│   ├── metadata.py      ImageState / AxisMetadata + propagation rules.
│   └── preview.py       Reduce arbitrary arrays to 2D/RGB thumbnail sources.
├── nodes/               Empty placeholder package (reserved for future nodes).
└── _tests/              pytest + pytest-qt tests (see §8).
scripts/launch_vipp_sample.py   Standalone launcher (napari + samples + widget).
docs/planning.md                Design intent and migration notes.
```

---

## 3. Core Layer (headless)

### 3.1 `core/pipeline.py` — model + executor + library

This is the heart of the system. It defines the data model, the node catalogue,
and the topological executor.

**Dataclasses (the model):**

| Type | Role |
| --- | --- |
| `ParameterSpec` | Declares one tunable parameter: name, label, `kind` (`int`/`float`/`choice`/`text`), default, min/max/step/decimals, optional `choices`. |
| `OperationSpec` | A node *type* in the library: `id`, `title`, `category`, `subcategory`, `input_type`/`output_type`, `parameters`, the `function` from `operations.py`, and `max_inputs` (`1`, an int, or `None` for unbounded). |
| `GraphNode` | A node *instance* in the graph: `id`, `operation_id`, resolved spec fields, and a mutable `params` dict. |
| `GraphConnection` | A directed edge `source_id -> target_id`, including the target input slot as `target_port`. |
| `ConnectionResult` | Returned by `connect()`: success flag, message, the created `connection`, and any auto-`removed` edges (single-input targets and occupied multi-input slots replace their edge). |
| `SourcePayload` | Carries `data`, optional `metadata`, and `name` injected for source nodes. |

**`NODE_LIBRARY`** is a tuple of `OperationSpec`s — this is the single source of
truth for available nodes. `NODE_LIBRARY_BY_ID` indexes it. Categories/
subcategories drive the palette grouping (e.g. `IMAGE_DATA_CATEGORY` with
`SOURCE_OUTPUT_GROUP`, `AXES_REGIONS_GROUP`, etc.). **To add a node:** write the
function in `operations.py`, import it, and append an `OperationSpec` here.

**`PrototypePipeline`** is the executable graph:

- Holds `nodes: dict[id, GraphNode]`, `connections: list[GraphConnection]`,
  and the last computed `outputs` / `output_states` (per node).
- `reset_starter_graph()` seeds the demo graph
  (`Image Source -> Gaussian Blur -> Otsu Threshold`, see `PROTOTYPE_NODES`).
- Editing API: `add_node`, `remove_node`, `connect`, `disconnect`, `set_param`.
  `connect()` enforces type compatibility (`_types_compatible`), rejects cycles
  (`_would_create_cycle`), respects `max_inputs`, and stores the target input
  slot for multi-input nodes.
- `run(input_data, input_metadata, input_name, source_payloads)` executes the
  graph by **Kahn-style topological iteration**: repeatedly run nodes whose
  inputs are all `completed`. Each node's result and its `ImageState` are stored.
- `_run_node` dispatches three cases: **source nodes** (no input — pull from
  `source_payloads` or the toolbar input), **multi-input nodes**
  (`max_inputs != 1`, e.g. `channel_composite`, `calculate_weighted_image`), and
  **single-input nodes**. It calls `spec.function(...)` then asks `metadata.py`
  to compute the output `ImageState`.

Special wiring inside `_run_node`: `save_output` receives the upstream
`image_state` as a kwarg (for ImageJ hyperstack metadata); `channel_composite`
gets a computed `channel_axis`.

### 3.2 `core/operations.py` — the node kernels

Pure functions, one per processing node, all taking array-like `data` plus
keyword params and returning a NumPy array. Grouped by theme: crop/contrast,
blur/filter (`gaussian_blur`, `median_filter`, `bilateral_filter`, ...),
thresholding (`otsu_threshold`, `adaptive_*`, ...), morphology (`dilate`,
`erode`, `opening`, `top_hat`, `fill_holes`, `volume_filter`, ...), math/logic
(`add_images`, `ratio_image`, `mask_image`, `logical_*`, `invert`,
`calculate_weighted_image`), channels/composites (`extract_channel`,
`channel_composite`, `rgb_composite`), type/scale (`convert_dtype`,
`rescale_intensity`, `normalize_image`, `clip_intensity`), and IO
(`save_output`, `save_array_output`).

Shared private helpers handle axis logic so operations stay
dimension-agnostic: `_xy_axes` / `_spatial_axes` find the trailing Y/X (and Z)
axes, `_apply_plane_wise` maps a 2D function across stacks, `_float_if_bool`,
`_restore_numeric_dtype`, `_to_grayscale`, `_global_threshold`, etc. When adding
ops, prefer these helpers so behaviour matches existing nodes on `TCZYX` data.

### 3.3 `core/metadata.py` — OME-NGFF-inspired `ImageState`

Every node output carries an `ImageState` describing the array *semantically*,
not just numerically. This is what makes the graph axis-aware.

- `AxisMetadata`: one axis — `name` (`t/c/z/y/x/...`), `type`
  (`time/channel/space`), `unit`, `scale`, `translation`.
- `ImageState`: `shape`, `dtype`, `kind`, `axes` (tuple of `AxisMetadata`),
  `bit_depth`, `value_range`, `value_pattern`, `memory`, `metadata_source`,
  `source_name`, and a `history` tuple of applied steps. Both have
  `to_dict`/`from_dict` (groundwork for save/load).
- `image_state_from_array(...)` builds state for source data. If a napari layer
  exposes OME-NGFF `multiscales` axes (`_axes_from_multiscales`), those are used;
  otherwise axes are inferred from shape (`infer_axis_metadata*`) and the
  `metadata_source` explicitly records the fallback.
- `transform_image_state(...)` / `transform_multi_input_image_state(...)`
  compute the output state given the input state(s), `operation_id`, and params.
  Per-op rules live in `_transformed_axes` (e.g. threshold ops collapse channels
  via `CHANNEL_COLLAPSE_OPERATIONS`, crop/slice shift axis ranges, composites
  add/restructure a channel axis).
- Formatting helpers feed the UI: `format_compact_metadata` (node-card line),
  `metadata_table_rows`, `metadata_history_items`, `format_detailed_metadata`.

### 3.4 `core/preview.py` — thumbnail reduction

`make_preview(data, mode, current_step, state)` reduces an arbitrary array to a
2D or RGB array for display. Modes: `slice` (take an index along non-spatial
axes — follows the napari dims `current_step` when available), `mip` (max along
spatial axes), `off`. When an `ImageState` is provided, `_state_aware_preview`
uses axis *types* to keep Y/X, build a fluorescence pseudo-color composite from a
channel axis (`_fluorescence_composite` with `FLUORESCENCE_COLORS`), and reduce
the rest. `normalize_thumbnail` converts the reduced array to display-ready
uint8 RGB at a fixed size.

---

## 4. UI Layer (Qt)

### 4.1 `_graph.py` — the canvas

A `QGraphicsView`/`QGraphicsScene`-based node editor.

| Class | Role |
| --- | --- |
| `PipelineGraphView` | The canvas. Owns node proxies, ports, and edges; handles pan/zoom, drag-to-connect, drag-drop from palette (`OPERATION_MIME`), and context menus. Emits signals (see below). |
| `NodeProxy` | A `QGraphicsProxyWidget` wrapping a `NodeCard`, positioned/moved on the scene; keeps connected edges attached. |
| `NodeCard` | The embedded node widget: title, category accent, thumbnail (`ClickablePreview`), compact metadata line, and Pin button. |
| `PortItem` | Input/output port circles on a node; the start/end of connections. |
| `ConnectionItem` / `PendingConnectionItem` | Curved Bézier edges (committed vs. in-progress while dragging). |

`PipelineGraphView` is purely presentational — it does **not** mutate the
pipeline. Instead it emits signals that `VippWidget` connects to:
`node_selected`, `node_delete_requested`, `pin_requested`,
`node_create_requested(op_id, pos)`,
`connection_requested(src, tgt, target_port)`,
`connection_removed(src, tgt, target_port)`, `status_message`.

### 4.2 `_widget.py` — `VippWidget` (orchestrator)

The single dock widget that ties everything together. It owns one
`PrototypePipeline` and reacts to graph/inspector events by mutating the model,
re-running, and refreshing the views.

Layout (3-pane `QSplitter`, toggleable via `SidePanelToggleButton`):

- **Left:** `NodePalette` (categorized, fuzzy-filterable `QTreeWidget`;
  drag source for new nodes).
- **Center:** toolbar (input selector, preview mode `Slice/MIP/Off`, follow-dims
  checkbox, reset) + the `PipelineGraphView` + status label.
- **Right:** inspector — selected-node parameter controls, metadata table/history,
  and `HistogramPanel` (slice/stack × linear/log).

Parameter controls are small reusable widgets: `ParameterControl`
(slider + numeric box with soft range expansion), `ChoiceControl`, `TextControl`,
`ImageSourceControl` (mode/layer/file/sample), and `AxisSliceControl` /
`AxisIntervalSlider` for the generic axis-subsetting node. `ParameterBounds`
carries dynamic, data-aware min/max for a control.

**The central data-flow method is `run_pipeline()`** (≈ line 2390):

1. Resolve inputs: toolbar layer + per-source-node `SourcePayload`s
   (`_source_payloads_for_pipeline`, including bundled samples).
2. Call `self.pipeline.run(...)`.
3. If dynamic parameter bounds changed
   (`_refresh_selected_parameter_controls`), re-run once so controls reflect
   real data ranges.
4. Refresh everything: `_update_thumbnails`, `_inspect_selected_node`,
   `_refresh_pinned_layer_if_active`, `_update_metadata_panel`,
   `_update_histogram`, and the status label.

Re-runs are **debounced** through `self._debounce_timer` (150 ms, single-shot →
`run_pipeline`) so dragging a slider does not thrash the executor.

napari integration touchpoints:
- **Inspect** (`_inspect_selected_node`): pushes the selected node's full-res
  output into napari as a managed layer for review; input layers may be hidden
  (`_hide_input_layer_for_inspection`) to avoid clutter and restored later.
- **Pin** (`pin_requested` → pinned overlay): keeps a node output (typically a
  mask) as a persistent napari layer via `_pinned_layer_name`.
- **Follow dims:** when checked, thumbnails/metadata/histogram track the napari
  dims slider via `_current_step()`.

### 4.3 `_theme.py`

`CATEGORY_COLORS` / `CATEGORY_TINTS` + `category_color()` / `category_tint()`
helpers. Category names here must match the `category` strings used in
`NODE_LIBRARY` (e.g. `Filtering`, `Segmentation`, `Morphology`).

---

## 5. Plugin Wiring

`napari.yaml` declares two contributions, both resolved from `pyproject.toml`'s
`napari.manifest` entry point:

- `napari-vipp.make_widget` → `napari_vipp._widget:VippWidget` (the dock widget).
- `napari-vipp.sample_data` → `napari_vipp._sample_data:make_sample_data`.

`make_sample_data()` returns synthetic `ZYX`, `CZYX`, and `TCZYX` stacks, each
tagged with OME-NGFF-style `multiscales` metadata (so `metadata.py` reads real
axes) and `napari_vipp_*` hint keys (`preferred_input` selects the `TCZYX`
sample as the default graph input).

---

## 6. End-to-End Data Flow (one update cycle)

```
user edits param / drags slider
   └─ ParameterControl emits value → VippWidget mutates GraphNode.params
        └─ _debounce_timer (150 ms) → run_pipeline()
             ├─ gather toolbar layer + SourcePayloads (samples/files/layers)
             ├─ PrototypePipeline.run() topologically executes nodes:
             │     for each node → operations.fn(data, **params)
             │                   → metadata.transform_image_state(...)
             ├─ (maybe) re-run once if dynamic bounds changed
             └─ refresh views:
                   preview.make_preview/normalize_thumbnail → NodeCard thumbnails
                   metadata.format_* → node cards + inspector table/history
                   HistogramPanel → selected node output
                   napari layers   → inspect / pinned outputs
```

---

## 7. Extending the System (common tasks)

**Add a processing node:**
1. Implement a pure function in `core/operations.py` (use the `_xy_axes` /
   `_apply_plane_wise` helpers for dimension safety).
2. Import it in `core/pipeline.py` and append an `OperationSpec` to
   `NODE_LIBRARY` with the right `category`/`subcategory`, `input_type`/
   `output_type`, `parameters`, and `function`.
3. If the output changes axes (drops/adds/collapses), add a rule in
   `metadata.py` `_transformed_axes` (and `CHANNEL_COLLAPSE_OPERATIONS` if it
   reduces a channel axis).
4. If it needs a bespoke inspector control, extend `_render_parameters` /
   `_parameter_bounds_for` in `_widget.py`. Otherwise the generic controls work.
5. Add tests under `_tests/` (operation correctness + metadata propagation).

**Add a multi-input node:** set `max_inputs` to an int or `None` on the
`OperationSpec`; the executor's multi-input path and
`transform_multi_input_image_state` handle the rest.

**Add a new category color:** add matching entries to `_theme.py`.

---

## 8. Tests, Build, and Run

Tests live in `src/napari_vipp/_tests/` (configured via `pyproject.toml`
`testpaths`):

- `test_operations.py` — operation correctness (largest).
- `test_graph.py` — pipeline model: connect/disconnect/cycle/type rules.
- `test_preview.py` — thumbnail reduction.
- `test_sample_data.py` — sample shapes/metadata.
- `test_widget.py` — `pytest-qt` UI behaviour (largest; needs a Qt backend).

Common commands (from the README):

```bash
python -m npe2 validate src/napari_vipp/napari.yaml   # manifest check
python -m ruff check .                                 # lint (E,F,I,UP,B; line 88)
python -m pytest                                       # tests
python scripts/launch_vipp_sample.py                   # standalone app
```

Runtime deps: `numpy`, `qtpy`, `scikit-image`, `scipy`, `tifffile`. Dev extras
add `napari[pyqt6]`, `npe2`, `pytest`, `pytest-qt`, `ruff`, `build`.

---

## 9. Known Gaps / Where Active Work Is Likely

These are intentionally not implemented yet (see [planning.md](planning.md) and
the README roadmap):

- **True multi-output nodes:** Split/Merge Channels, Channel Overlap, and
  non-image outputs (histograms/morphology tables) need model + UI support.
- **OME-Zarr I/O:** metadata is OME-NGFF-*inspired* internally; no real
  import/export.
- `nodes/` is an empty reserved package.

### Recently implemented

- **Persistence:** workflow save/load via [core/workflow.py](../src/napari_vipp/core/workflow.py).
  Serializes nodes (operation id + params), connections, and canvas node
  positions to a versioned JSON document. The toolbar exposes *Save workflow...*
  and *Load workflow...*; load uses `PrototypePipeline.restore_graph()` and
  skips any unknown operation ids.
- **Pipeline export:** graph → runnable Python via
  [core/export.py](../src/napari_vipp/core/export.py) and the *Export Python...*
  toolbar button. The generated script imports the pure functions from
  `core/operations.py`, rebuilds the graph as a `run_pipeline()` function, and
  ships a `batch_process()` helper plus an `argparse` entry point so a tuned
  pipeline can run headlessly over a single image or a whole folder.

When continuing work, prefer adding capability in `core/` first (keeping it Qt-
and napari-free) and then surfacing it in `_widget.py`.
