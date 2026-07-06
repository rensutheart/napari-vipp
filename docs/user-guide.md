# VIPP User Guide

This guide explains everyday usage of the VIPP widget from an end-user point of
view.

## Toolbar Controls

### View dims

When the active image has non-XY axes such as `T`, `Z`, or `C`, VIPP shows a
`View dims` bar above the graph. These controls mirror napari's dimension
sliders and remain usable when napari's own Z slider is hidden in 3D view.

Moving a VIPP slider updates the napari viewer. Moving a napari slider updates
the VIPP slider. For downstream nodes whose axis length differs from the source
image, for example after `Rescale Axes`, VIPP shows the node's local axis range
and maps it to the equivalent relative napari position. For nodes that drop an
axis, for example `Split Channels`, VIPP still maps the remaining axes back to
their original source dimensions, so the `Z` control continues to scrub the same
stack axis downstream.

In narrow dock layouts, the bar first hides the full sliders, then collapses to
a `View dims...` menu containing the same controls.

### Follow napari dims

- On (default): previews, slice histograms, and the Current view metadata track
  the current napari/VIPP dims position (for example T and Z).
- Off: these panels use a stable/default sampling context instead of following
  napari or VIPP sliders.

Use On for normal interactive work. Use Off when you want a stable reference
view while scrubbing dims or comparing parameter changes.

### Run all in BG

- Off (default): only known slower operations use background processing.
- On: all pipeline recomputes run in background mode.

Background mode shows progress in the toolbar and node graph. This is useful
for long pipelines and large images because you can see progress while updates
run.

For small or very fast edits, Run all in BG can add overhead. If updates feel
slower for simple operations, switch it off.

## Named Port Tunnels

Named port tunnels are hidden wires for outputs that are reused many times,
such as split fluorescence channels, masks, ROIs, or the original reference
image. They keep dense graphs readable without changing the calculation.

To create one:

1. Right-click an output port.
2. Choose `Create output tunnel...`.
3. Give it a short name such as `Ch1`, `DAPI`, `Mask`, or `Reference`.

To use one:

1. Right-click a compatible input port.
2. Choose `Use tunnel`.
3. Select the named source.

The graph shows compact tunnel badges on the source output and subscribed input
ports instead of drawing the long wire. A tunnel still behaves like a real
connection: it occupies that input slot, replaces any previous connection to
that slot, respects data-type compatibility, rejects cycles, runs in batch mode,
and is saved with the workflow.

To inspect tunnel usage:

1. Click a tunnel badge to highlight its source and all subscribed inputs.
2. Right-click a tunneled port to reveal subscribers, rename the tunnel, remove
   it, or open the tunnel manager.
3. Use the toolbar `Tunnels...` button to filter, audit, focus, reveal, rename,
   or delete named sources from one table.

## Graph Notes

Use `Add note` to place a movable annotation on the graph canvas. Notes are for
workflow reasoning, reminders, interpretation comments, or marking alternatives;
they do not run as pipeline nodes and do not affect outputs.

Notes can be moved like nodes. Double-click a note, or right-click it and choose
`Edit note...`, to change the text. Select a note and press Delete, or use the
note context menu, to remove it. Note creation, movement, text edits, and
deletion are included in undo/redo.

Notes are saved in workflow JSON and reload at the same canvas position.

## When To Use Which Mode

Recommended default for most users:

- Follow napari dims: On
- Run all in BG: Off

Recommended for large images or long graphs:

- Follow napari dims: On
- Run all in BG: On

Recommended for fixed-reference comparisons while navigating dims:

- Follow napari dims: Off
- Run all in BG: choose based on pipeline size

## Manual Calculation Nodes

Some table-producing nodes are intentionally not recalculated on every
parameter change. Current manual nodes are `Measure Objects`, `Measure Objects
+ Intensity`, `Measure 3D Mesh Morphology`, `Analyze Skeleton`,
`Measure Skeleton Branches`, `Skeleton Graph Tables`, `Measure Overall
Skeleton Network`, `Colocalization Metrics`, `Masked Colocalization Metrics`,
`RACC Index`, and `Masked RACC Index`.

When selected, these nodes show an `Execution` panel with `Calculate` or
`Recalculate`. The same action is available on the node card. If upstream data
or parameters change after a calculation, the node keeps its last table output
available downstream but marks it as stale until recalculated. Workflow files do
not store cached tables, so loading a workflow starts from the node settings and
the table must be calculated again in the UI. Exported Python scripts calculate
manual nodes normally during headless runs.

The toolbar `Calculate all` button recalculates every manual node that is not
current, including never-calculated, stale, or errored manual nodes.

The `Execution` panel also has `Auto Recalculate`. This is off by default. When
enabled for a manual node, VIPP recalculates that node automatically when
upstream data or relevant parameters change, and hides the manual
`Recalculate` button because the node no longer waits for an explicit click.
Use it only when the node is fast enough for the current image size.

Manual node cards use status colours:

- gray: not calculated;
- green: calculated and current;
- orange: cached result is stale;
- red: calculation failed.

## Object And Mesh Morphology

Use `Measure Objects` for standard region/object measurements from a label
image. Use `Measure Objects + Intensity` when you also need per-object intensity
statistics from a separate image input.

Use `Measure 3D Mesh Morphology` only for true 3D label images. It extracts
per-object surfaces with marching cubes, applies carried Z/Y/X scale metadata,
and reports mesh surface area, mesh volume, sphericity, 3D solidity, convex-hull
metrics, and status/error columns for objects that are too small or geometrically
invalid. The node is manual/cached because these calculations are more expensive
than ordinary regionprops. The reference workflow is
`examples/synthetic-3d-mesh-morphology.json`.

The broader object, mesh, skeleton, and table-composition contract is documented
in [measurement-workflows.md](measurement-workflows.md). The bundled workflow
examples are indexed in [../examples/README.md](../examples/README.md).

## Colocalization And RACC Nodes

First-pass colocalization nodes live under `Colocalization & Spatial Analysis`.
Connect two same-shaped channel images, usually from `Split Channels`, into the
named `Channel 1 image` and `Channel 2 image` ports.

Manual thresholds are normalized `0..255` values. VIPP jointly scales the two
input channels into this range before calculating metrics, inspector scatter
views, colocalized-voxel views, or RACC. `Costes auto` can be selected instead
of manual thresholds; the calculated Costes thresholds are written back into the
threshold controls so the values are visible.

When a colocalization threshold node is selected, the inspector shows a scatter
density panel with red/green threshold guide lines. Dragging a guide line
switches that node to manual thresholds and updates the corresponding threshold
value. Masked variants add a third `ROI mask` input and restrict metrics,
scatter display, colocalized voxels, and RACC output to that mask.

`Colocalization Metrics`, `Masked Colocalization Metrics`, `RACC Index`, and
`Masked RACC Index` are manual/cached nodes. `Colocalized Voxels` and `Masked
Colocalized Voxels` are live visual feedback nodes for threshold tuning. The
pixel/RACC reference workflow is `examples/synthetic-colocalization-racc.json`.

Object-aware colocalization nodes are also available in the same category.
Use `Object Colocalization Metrics` when you have object labels plus two
matching channel images and want one table row per object. Use `Label Overlap
Association`, `Nearest Object Distance`, and `Event Localization` for
label-label overlap, nearest-neighbor association, and puncta/event assignment
against labels, masks, or ROIs. These nodes output tables designed to merge
with `Measure Objects` and `Measure Objects + Intensity` through `label_id`
and leading axis index columns. The object-table reference workflow is
`examples/synthetic-object-colocalization-association.json`.

## Skeleton Analysis Nodes

The skeleton/network nodes are documented in
[skeleton-nodes.md](skeleton-nodes.md). That guide explains which nodes expect
binary masks, which expect already skeletonized masks, which nodes produce
visual QC outputs, and which nodes produce measurement tables. In brief,
`Measure Skeleton Branches` produces detailed branch rows, `Summarize Skeleton
Branches` converts those rows into branch-length/tortuosity distributions and
branch-type fractions, and `Measure Overall Skeleton Network` measures
whole-network graph metrics directly from a skeleton mask. Use
`examples/synthetic-skeleton-qc.json` for a compact skeleton check and
`examples/synthetic-advanced-skeleton-network.json` for a richer time-indexed
3D skeleton graph/table stress test.
