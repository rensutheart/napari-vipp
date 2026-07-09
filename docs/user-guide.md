# VIPP User Guide

This guide explains everyday usage of the VIPP widget from an end-user point of
view.

## Toolbar Controls

### Graph search

Use `Search graph` above the canvas to highlight matching graph elements.
Search checks node titles, operation IDs, named tunnel names, and Batch Output
tags. Press Enter or `Focus` to jump to the next match. Tunnel matches reveal
the tunnel source and subscribed inputs.

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

### Thumbnail Contrast Range

`Preview` chooses whether graph thumbnails show the current slice, a maximum
projection, or no thumbnail. `Contrast` chooses the intensity mapping:
percentile, min-max, or raw. `Range` chooses where that mapping is measured.
Use `Stack` (default) for stable brightness while moving through Z/T/C, which
is usually best for PSFs and restored float images. Use `Slice` when you want
each viewed slice to stretch itself locally.

### Run all in BG

- Off (default): only known slower operations use background processing.
- On: all pipeline recomputes run in background mode.

Background mode shows progress in the toolbar and node graph. This is useful
for long pipelines and large images because you can see progress while updates
run.

For small or very fast edits, Run all in BG can add overhead. If updates feel
slower for simple operations, switch it off.

## Axis And Channel Splitting

Use `Split Channels` when the input has a semantic channel axis, such as OME
`C` metadata, VIPP sample metadata, `Combine Channels` output, or a conventional
RGB/RGBA channel-last image. The node creates one graph output port per channel.

Use `Split Axis` when you want one output per index of another stack axis, such
as timepoints, Z slices, or a leading non-channel axis. This keeps accidental
Z/time splitting separate from fluorescence channel splitting.

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

Right-click a node and choose `Add note` to attach a movable annotation to that
node. Notes are for workflow reasoning, reminders, interpretation comments, or
marking alternatives; they do not run as pipeline nodes and do not affect
outputs.

Attached notes move with their node during dragging, insertion layout changes,
and `Auto structure graph`. Dragging a note only moves the note. Double-click a
note, or right-click it and choose `Edit note...`, to change the text. Select a
note and press Delete, or use the note context menu, to remove it. Note creation,
movement, text edits, and deletion are included in undo/redo.

Notes are saved in workflow JSON and reload with their attached node.

## Workflow Save And Load

Use `Open example...` for bundled workflow templates. The example chooser is
grouped by task, such as segmentation, measurements, colocalization, skeletons,
and restoration. Example workflows use `Image Source` nodes set to bundled
samples, so they run without loading sample layers into napari first.

`Save workflow...` writes the graph, node parameters, connections, canvas
positions, named tunnels, graph notes, and selected inspector state to workflow
JSON. The selected node and whether the right inspector panel is visible are
restored when the workflow is loaded.

Use `Load workflow...` for a workflow JSON file that is not in the bundled
example set.

Per-node thumbnail visibility is optional workflow UI metadata. Enable
`Save thumbnail visibility in workflows` in Settings when hidden/shown thumbnail
choices should be restored with the workflow. VIPP stores only the visibility
preference; thumbnail image pixels, cached arrays, and cached tables are never
embedded in workflow JSON.

Old workflows without this metadata still load. When thumbnail metadata is
absent, VIPP starts with normal thumbnail visibility.

## When To Use Which Mode

Recommended default for most users:

- Follow napari dims: On
- Run all in BG: Off

Recommended for large images or long graphs:

- Follow napari dims: On
- Run all in BG: On
- Cache mode: Smart interactive cache, or Low-memory mode when RAM is tight

Recommended for fixed-reference comparisons while navigating dims:

- Follow napari dims: Off
- Run all in BG: choose based on pipeline size

The Settings menu also exposes `Auto memory guard` and `Cache limit`. If
keep-all caching uses too much reclaimable memory, VIPP switches to Smart
interactive cache and warns you. `Cache limit` is the allowed percentage of
`free RAM + current VIPP cache`, and defaults to 90%. Mark important
intermediate nodes with `Keep output cached` in the inspector when they should
survive Smart or Low-memory pruning. See
[cache-and-memory.md](cache-and-memory.md) for details.

## Manual Calculation Nodes

Some table-producing nodes are intentionally not recalculated on every
parameter change. Current manual nodes are `Measure Objects`, `Measure Objects
+ Intensity`, `Measure 3D Mesh Morphology`, `Analyze Skeleton`,
`Measure Skeleton Branches`, `Skeleton Graph Tables`, `Measure Overall
Skeleton Network`, `Colocalization Metrics`, `Masked Colocalization Metrics`,
`RACC Index`, `Masked RACC Index`, `Richardson-Lucy Deconvolution`, and
`Richardson-Lucy TV Deconvolution`.

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

## Restoration And PSF Workflows

PSF-aware deconvolution lives under `Filtering -> Restoration & PSF`. The first
supported path is explicit: the image and the PSF are separate graph images, and
the PSF is prepared before it is reused.

`Born-Wolf PSF` can generate scalar 2D or 3D PSFs from the connected image's
normalized metadata. Keep `Auto from metadata` on when wavelength, objective
NA, refractive index, and pixel spacing are present; the disabled fields show
the resolved values used for calculation. For multi-channel images, `Channel =
-1` generates one output port per metadata channel, such as `488 PSF` and
`561 PSF`. Set `Channel` to a specific index to generate only one
channel-specific PSF. Missing required auto values are marked red and the node
will not produce a PSF until the metadata is supplied or `Auto from metadata`
is turned off. In manual mode the physical fields become editable exact
numeric inputs and are initialized to non-zero defaults before calculation. PSF
size and pupil samples are always explicit numeric entries, not sliders,
because they are generation settings.

For a measured PSF workflow:

1. Add an `Image Source` for the microscopy image.
2. Add a second `Image Source` for the measured PSF image, such as a bead PSF
   saved as TIFF, OME-TIFF, NumPy, or a napari layer.
3. Connect the PSF source to `Prepare / Validate PSF`.
4. Connect the microscopy image to the deconvolution node's `Image` input.
5. Connect the prepared PSF to the deconvolution node's `PSF` input.
6. Choose `2D YX` for plane-wise restoration or `3D ZYX` for volumetric
   restoration, then click `Calculate`.

Use `Split Channels` first when each fluorescence channel needs its own
deconvolution branch. Born-Wolf can expose one generated PSF output per
channel, but Richardson-Lucy nodes still consume one scalar PSF per branch;
connect the matching channel-specific PSF output to that branch's
`Prepare / Validate PSF` or deconvolution node.

`Prepare / Validate PSF` converts the PSF to `float32`, replaces non-finite
values with zero, optionally clips negatives, centers by peak or centroid,
optionally forces odd shape, and normalizes the sum. Use `Peak` centering for
generated or clean measured PSFs. Use `Centroid` when a bead PSF is broader or
slightly asymmetric. Keep `Normalize sum` on for Richardson-Lucy deconvolution.
An empty or all-invalid PSF raises an error instead of producing a misleading
restoration.

`Richardson-Lucy Deconvolution` is the baseline comparator. It is useful for
checking whether the PSF and iteration count are plausible. `Richardson-Lucy TV
Deconvolution` adds total-variation regularization to reduce noise
amplification while preserving stronger edges. Start with a modest iteration
count and increase slowly; ordinary RL can sharpen features and noise together.
For RL-TV, start with the default `TV regularization` and adjust in small steps.
Too little behaves like ordinary RL. Too much can flatten fine structure.

Important caveats:

- The PSF dimensionality must match the selected spatial mode: 2D PSF for
  `2D YX`, 3D PSF for `3D ZYX`.
- Deconvolution output is always `float32`, not the input integer dtype.
- Output metadata follows the image input, not the PSF input.
- PSFs are normalized defensively inside deconvolution even when the preparation
  node is skipped.
- `Preserve input scale` keeps restored intensities near the original image
  scale after internal normalization.
- The first implementation uses same-size convolution boundary behavior. Edges
  can show restoration artifacts; crop margins or interpret image borders with
  care.
- GPU acceleration, blind deconvolution, spatially variant PSFs, and
  vendor-specific file import are outside this first restoration pass.

Reference review workflows:

- `examples/synthetic-deconvolution-rl-tv.json`: compact 2D measured-PSF
  workflow for quick inspection.
- `examples/synthetic-3d-deconvolution-rl-tv.json`: true ZYX measured-PSF
  workflow for the more common volumetric microscopy case.

Both examples use a blurred/noisy synthetic image sample plus a separate
measured-PSF sample, prepare the PSF, then run ordinary RL and RL-TV side by
side.

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
