# Known Bugs / Issues

## BUG-001 — `channel_colors` on single-channel inputs raises hard pipeline error instead of being silently ignored

**Reported:** 2026-07-30  
**Status:** Open  
**Version:** 0.12.0a1

### Description

Setting `channel_colors` on an `input` node (e.g. `"channel_colors": "Magenta"`) when loading a single-channel grayscale TIFF z-stack causes the pipeline to abort with:

```
Pipeline error: Channel colours require a declared channel axis in the image metadata.
```

### Root cause

`with_channel_colors` in `src/napari_vipp/core/metadata.py` (line ~1423) calls `_channel_axis_index(state.axes)` which returns `None` for images that have no declared channel axis (i.e. all standard single-channel TIF/z-stacks). The function then raises a `ValueError` unconditionally instead of skipping the colour assignment.

```python
# metadata.py  ~line 1420
channel_index = _channel_axis_index(state.axes)
if channel_index is None or channel_index >= len(state.shape):
    raise ValueError(
        "Channel colours require a declared channel axis in the image metadata."
    )
```

### Expected behaviour

If there is no channel axis, `channel_colors` should be silently ignored (or at most emit a warning). A single-channel image is a perfectly valid input and a user-supplied display colour hint should not abort the pipeline.

### Suggested fix

Return `state` unchanged when `channel_index is None`, rather than raising:

```python
channel_index = _channel_axis_index(state.axes)
if channel_index is None or channel_index >= len(state.shape):
    return state  # no channel axis — colour hint cannot be applied, ignore silently
```

### Workaround

Leave `channel_colors` empty (`""`) on input nodes that load single-channel TIFF stacks. The channel colour is cosmetic only and does not affect any computation.

---

## BUG-002 — `logical_and` (and other multi-input operations) hard-fail when inputs have different physical axis units saved by different software

**Reported:** 2026-07-30  
**Status:** Open  
**Version:** 0.12.0a1

### Description

When two mask stacks originating from different TIFF files are combined with `logical_and`, the pipeline aborts with:

```
Pipeline error: Logical AND cannot combine Input 1 and Input 2: their physical grids
are incompatible (axis 0 (z) units are incompatible ('index/pixel' versus 'micrometer');
axis 1 (y) units are incompatible ('index/pixel' versus 'micrometer'); axis 2 (x) units
are incompatible ('index/pixel' versus 'micrometer')).
Reorder or explicitly resample one input onto the other grid before combining them.
```

This occurs even when both arrays are identical in shape and represent the same physical space. In this case the MAGENTA channel was saved by DeconvolutionLab2 (no physical unit metadata → `index/pixel`) and the BLUE channel was a raw ImageJ save (with microscope-embedded pixel sizes → `micrometer`).

### Root cause

The grid-compatibility check in the multi-input combiner operations (at minimum `logical_and`, likely also `logical_or`, `logical_xor`, `subtract_images`, `add_images`) treats axis-unit mismatches as hard errors. There is no way to opt out or force-combine arrays that are known to be spatially registered.

### Expected behaviour

When arrays are the same shape and the user has not requested physical resampling, the pipeline should either:

1. Warn and proceed (units are metadata, not data), or
2. Offer a boolean parameter such as `ignore_grid_mismatch` to allow the user to override the check.

### Suggested fix

Allow an explicit override flag on all multi-input operations, or treat `index/pixel` as a wildcard unit that is compatible with any physical unit of the same scale.

### Workaround

Insert a `set_pixel_size` node (unit = `"pixel"`, all scales = 1.0) immediately before each input to the combining operation. This strips the conflicting physical metadata and forces both inputs onto the same nominal grid.

---

## BUG-003 — Scatter plot disappears and shows "calculating exact counts" while dragging threshold lines

**Reported:** 2026-07-30  
**Status:** Open  
**Version:** 0.12.0a3  
**Affected node:** `masked_colocalization_metrics` (and likely `colocalization_metrics`)

### Description

When manually dragging a threshold line on the colocalization scatter plot inspector panel, the scatter plot image disappears entirely and is replaced by the message "calculating exact counts" for the duration of the drag. The plot only reappears once the drag is released and the computation completes.

### Expected behaviour

The scatter plot should remain visible at all times during a threshold drag. The threshold guide lines should update their visual position immediately and responsively as the user drags. Any heavy downstream computation (exact voxel counts, Manders fractions) should run asynchronously in the background and update the numeric display once complete, without blocking or hiding the scatter plot itself.

This is a standard pattern in interactive visualisation tools: the lightweight visual (moving a line) should never be gated on a heavy computation.

### Suggested fix

Decouple the scatter plot render from the count recalculation. The scatter plot image and the threshold line positions are pure display operations that require only the pre-computed 2D histogram (already available). Exact counts should be recalculated on a debounced timer or background thread after the drag ends, leaving the plot visible throughout.

### Workaround

None. The plot must be released before it reappears.

---

## BUG-004 — Missing CPU parity target for `masked_colocalization_metrics` blocks exact validation during optimization

**Reported:** 2026-07-30  
**Status:** Open  
**Version:** 0.12.0a3  
**Affected node:** `masked_colocalization_metrics`

### Description

During optimization/parity validation, the pipeline reports:

```
masked_colocalization_metrics_1: Private current parity requested 'cpu-masked_colocalization_metrics-v1' but reported 'missing'; exact validation cannot continue.
```

This causes exact validation to stop instead of handling the missing parity implementation gracefully.

### Expected behaviour

If a requested parity target is unavailable, the system should degrade gracefully:

1. Emit a clear warning identifying the missing target, and
2. Fall back to a supported execution/validation path (or allow non-exact validation) so pipeline execution can continue.

### Suggested fix

Add/restore registration for `cpu-masked_colocalization_metrics-v1`, or implement robust fallback logic in parity selection so missing private parity targets do not hard-stop validation.

### Relationship to other issues

Likely related to the broader CPU/GPU optimization behavior in batch workflows: this appears to be a specific optimizer/parity-path failure that may contribute to optimization non-functionality.

### Workaround

No robust workaround identified in UI. Temporary mitigation may require disabling exact parity validation for affected nodes/workflows until parity registration is fixed.

---

## FEATURE-002 — Input nodes do not display the filename or sample name on the node tile, making multi-input graphs hard to read

**Reported:** 2026-07-30  
**Status:** Open  
**Version:** 0.12.0a3  
**Affected node:** `input`

### Description

When a workflow contains multiple `input` nodes (e.g. the four-channel cytoplasm ROI workflow: MAGENTA, BLUE, GREEN, RED), each node tile in the graph canvas shows only the generic label "Input" with no indication of which file or sample it is bound to. The only way to identify a node is to click it and read the file path in the inspector panel.

This makes the graph difficult to read at a glance, especially for workflows with four or more inputs, and increases the risk of wiring inputs to the wrong downstream nodes.

### Expected behaviour

The node tile should display a short identifier derived from the current binding, for example:

- **File path mode:** the filename stem (e.g. `P-tau_B48_R2_N2_Cell2_16_Deconvolved_MAGENTA`) or at minimum the file extension and a truncated path.
- **Collection mode:** the folder name and the glob pattern (e.g. `Deconvolved_MAGENTA/  *.tif`).
- **Sample mode:** the sample name string (e.g. `VIPP synthetic colocalization`).
- **Layer name mode:** the layer name.

The display should update live as the user changes the binding in the inspector, and should truncate gracefully (e.g. show only the last path component, ellipsis-truncated to fit the tile width).

### Notes

The `layer_name` parameter already exists on the input node for this purpose but is not surfaced on the tile. Populating the tile subtitle from whichever of `layer_name`, `file_path` (stem), or `sample_name` is non-empty would resolve this with minimal new logic.

---

## FEATURE-003 — Inserting a node into an existing connection and re-routing tunnel sources via drag

**Reported:** 2026-07-30  
**Status:** Open  
**Version:** 0.12.0a3

### Description

Two related graph-editing interactions are currently not supported:

**3a — Insert a node into an existing wire by dragging**

When a new node is dragged from the node palette and dropped onto an existing connection wire (the line between two nodes), it should be inserted into that connection automatically. The incoming wire should connect to the new node's first input port and the outgoing wire should connect from the new node's first output port to the original downstream node. This avoids the current workflow of: manually disconnect → place node → reconnect both ends.

This is especially useful when inserting preprocessing steps (e.g. a `subtract_background` or `set_pixel_size` node) into an already-wired pipeline.

**3b — Re-route a tunnel by dragging its source**

Tunnels (the long-distance labelled connections that skip the canvas) currently have a fixed source node and port defined in the workflow. There is no way to change which node a tunnel originates from without editing the JSON directly. A drag handle on the tunnel source end (or in the tunnel label chip on the canvas) should allow the user to re-route the tunnel to a different node's output port interactively.

### Expected behaviour

- Dropping a node onto a wire inserts it in-line; the original connection is split and re-wired automatically.
- Dragging the source end of a tunnel re-routes it to the newly released output port, updating the tunnel definition live.
- Both operations should be undoable.

### Notes

Both interactions are standard in node-graph editors (e.g. Blender Compositor, Unreal Blueprint, DaVinci Resolve Fusion). The insert-on-drop interaction in particular would significantly reduce the friction of iteratively refining a workflow that is already connected.

---

## FEATURE-004 — No tab bar for switching between multiple open workflows

**Reported:** 2026-07-30  
**Status:** Open  
**Version:** 0.12.0a3

### Description

When multiple workflows are loaded or created during a session, there is no way to switch between them quickly. Each workflow occupies the entire canvas and the only way to work with a second workflow is to close the current one and re-open the other, losing any unsaved canvas state.

This becomes a significant friction point when:

- Comparing two workflow variants side by side (e.g. with and without background subtraction)
- Running one workflow while editing another
- Keeping a reference workflow open alongside an experimental one

### Expected behaviour

A tab bar above (or below) the canvas should list all currently open workflows by name, with the filename stem or a user-editable label as the tab title. Clicking a tab switches the canvas to that workflow instantly. Standard tab interactions should be supported:

- **New tab** — open a blank workflow or load a file into a new tab
- **Close tab** — with an unsaved-changes prompt if applicable
- **Rename tab** — double-click the tab label to set a display name
- **Reorder tabs** — drag to reorder

The run/preview state of each workflow should be preserved independently when switching tabs so that a running batch in one tab is not interrupted by switching to another.

### Notes

This is analogous to the tab model used in VS Code, Jupyter, and most modern IDEs. Even a minimal implementation (open N workflows, click to switch, no persistent state between sessions) would substantially improve the multi-workflow development experience.

---

## FEATURE-001 — Scatter plot resolution is too low for wide-range intensity data; no high-resolution export or interactive pop-out

**Reported:** 2026-07-30  
**Status:** Open  
**Version:** 0.12.0a3  
**Affected node:** `colocalization_scatter_plot` (and the scatter plot panel in `masked_colocalization_metrics`)

### Description

The scatter plot is currently rendered at a fixed 512×512 px output with a default of 128 histogram bins. For images with wide native intensity ranges (e.g. 0–720 ADU for 16-bit data, as encountered in this project), most of the scatter plot area is empty space and the populated region occupies only a small fraction of the axes, making the colocalization distribution visually unreadable at display size.

Two capabilities are missing:

1. **High-resolution export.** There is no way to save the scatter plot at a resolution suitable for publication or detailed inspection (e.g. 1024×1024 or 2048×2048 with proportionally more histogram bins). The 512×512 image is insufficient for a figure panel.

2. **Interactive pop-out window.** There is no way to open the scatter plot in a larger, resizable window that still supports interactive threshold adjustment. The current in-panel view is too small to see the structure of the scatter distribution when the intensity range exceeds ~300 ADU.

### Expected behaviour

- A **configurable bin count and output size** parameter on `colocalization_scatter_plot` so that high-resolution renders can be produced as a batch output node or via the inspector.
- An **"Open in window"** or **"Pop out"** button on the scatter plot inspector panel that opens a resizable, interactive window showing the scatter plot at full resolution with draggable threshold lines and live count display.
- The pop-out window must still support **export to image file** (PNG/TIFF) at the display resolution of the window.

### Notes

The axis range should automatically zoom to the populated data region (data min–max or a configurable percentile clip) rather than spanning the full 0–display_max range, which is the primary cause of the small-dot problem on wide-range data. This would benefit all users regardless of whether the pop-out is implemented.
