# VIPP User Guide

This guide explains everyday usage of the VIPP widget from an end-user point of
view.

## Toolbar Controls

### Follow napari dims

- On (default): previews, slice histograms, and the Current view metadata track
  the current napari dims position (for example T and Z).
- Off: these panels use a stable/default sampling context instead of following
  napari sliders.

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
+ Intensity`, `Analyze Skeleton`, and `Measure Skeleton Branches`.

When selected, these nodes show an `Execution` panel with `Calculate` or
`Recalculate`. The same action is available on the node card. If upstream data
or parameters change after a calculation, the node keeps its last table output
available downstream but marks it as stale until recalculated. Workflow files do
not store cached tables, so loading a workflow starts from the node settings and
the table must be calculated again in the UI. Exported Python scripts calculate
manual nodes normally during headless runs.

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

## Skeleton Analysis Nodes

The skeleton/network nodes are documented in
[skeleton-nodes.md](skeleton-nodes.md). That guide explains which nodes expect
binary masks, which expect already skeletonized masks, which nodes produce
visual QC outputs, and which nodes produce measurement tables.
