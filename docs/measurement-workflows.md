# Measurement Workflow Guide

Last reviewed: 2026-07-02

This guide is the practical wrap-up for the current measurement and morphology
phase. It explains which nodes produce tables, how those tables are meant to be
combined, what units are currently carried, and which scientific assumptions
need to be kept visible.

## Core Table Contract

VIPP table-producing nodes emit `TableData`, not images. Tables carry:

- stable column order;
- row count and column count;
- a table kind, such as `Object measurements` or `3D mesh morphology`;
- a source name when available;
- per-column units where the calculation has a clear unit.

Table composition nodes preserve units when possible:

- `Merge Tables` keeps units from upstream columns and suffixes units along
  with duplicated column names.
- `Select Table Columns` preserves units for selected columns.
- `Add Metadata Columns` preserves existing units and leaves new metadata
  columns unitless.
- `Summarize Measurements` propagates units to numeric summary columns except
  `count` statistics.

Stable identity columns are intentionally simple: leading axis indices such as
`t_index`, `c_index`, and `z_index`, plus object or graph identifiers such as
`label_id`, `component_id`, `branch_id`, `node_id`, and `edge_id`. The current
workflows use `Add Metadata Columns` for experimental fields such as condition,
replicate, treatment, or batch.

## Manual Calculation

Expensive table nodes are manual/cached by default. They expose
`Calculate`/`Recalculate` on the node card and inspector, and the toolbar
`Calculate all` action runs all manual nodes that are missing, stale, or in an
error state.

Current manual table nodes:

- `Measure Objects`;
- `Measure Objects + Intensity`;
- `Measure 3D Mesh Morphology`;
- `Analyze Skeleton`;
- `Measure Skeleton Branches`;
- `Skeleton Graph Tables`;
- `Measure Overall Skeleton Network`.

Manual nodes can opt into `Auto Recalculate`, but the default should remain off
for analysis steps that may become expensive on large z-stacks.

## Object Measurements

Use `Measure Objects` when the input is a label image and the output should be
one row per object. The baseline columns include object ID, pixel/voxel
area/volume, centroid, bounding box, equivalent diameter, extent, and Euler
number.

Optional groups add:

- shape descriptors;
- axis/inertia descriptors;
- derived shape ratios;
- 2D boundary descriptors;
- 2D shape moments.

Physical columns are emitted when pixel-size metadata is available. In addition
to physical area or volume, VIPP reports calibrated centroid and bounding-box
coordinates, equivalent diameter, bounding/fill size, maximum Feret diameter,
major/minor axis length, bounding-box side lengths, and inertia eigenvalues
where those values are available from spacing-aware region measurements.
Isotropic 2D inputs also report physical perimeter, physical Crofton perimeter,
and physical perimeter-to-area ratio. For anisotropic 2D pixels, physical
perimeter columns remain `NaN` rather than using a misleading scalar scale.
The analytical phantom validation report is
[analytical-phantom-validation.md](analytical-phantom-validation.md).

Use `Measure Objects + Intensity` when the same labels should also be measured
against a matching intensity image. It uses named `Labels` and `Intensity image`
ports and adds mean, min, max, sum, and standard deviation intensity per label.

## 3D Mesh Morphology

Use `Measure 3D Mesh Morphology` only for true 3D label images. It extracts one
surface per object with marching cubes and uses carried Z/Y/X scale metadata for
anisotropic microscopy data.

Implemented columns include:

- `voxel_count`;
- `voxel_volume_physical`;
- `mesh_volume_physical`;
- `mesh_surface_area_physical`;
- `surface_area_to_volume`;
- equivalent sphere radius and diameter;
- `sphericity`;
- physical mesh extents and extent ratios;
- optional convex-hull volume and area;
- `solidity_3d`;
- `surface_area_to_convex_hull_area`;
- `mesh_status` and `mesh_error`.

Tiny or invalid objects remain in the table with `NaN` mesh metrics and an
explanatory status instead of failing the whole node.

Scientific cautions:

- mesh measurements depend strongly on the segmentation threshold;
- anisotropic voxel spacing must be correct before measurement;
- marching-cubes surfaces are approximations of voxelized objects;
- very small or flat objects may not support stable surface or convex-hull
  metrics;
- mesh export, mesh preview, and specialist mesh repair remain future work.

## Skeleton And Network Measurements

Skeleton nodes are documented in more detail in
[skeleton-nodes.md](skeleton-nodes.md). The measurement phase currently supports
three table levels:

- `Analyze Skeleton`: per-component network measurements from a skeleton mask.
- `Measure Skeleton Branches`: one row per traced branch.
- `Skeleton Graph Tables`: explicit graph-node and graph-edge tables.
- `Measure Overall Skeleton Network`: compact per-block whole-network summary.

`Summarize Skeleton Branches` converts row-per-branch tables into grouped
length/tortuosity distributions and branch-type counts/fractions.

Scientific cautions:

- skeleton metrics assume the input skeleton represents the biology of interest;
- branch counts and endpoints are sensitive to segmentation and skeletonization;
- pruning removes terminal spurs, so before/after tables should be kept when
  quality control matters;
- generic skeleton metrics are useful for mitochondria, neurites, vessels,
  fibers, and other curvilinear objects, but specialist mitochondrial indices
  remain future optional nodes.

## Analysis-Ready Tables

The intended measurement workflow is composable:

```text
labels
  -> Measure Objects
  -> optional Measure Objects + Intensity / Measure 3D Mesh Morphology
  -> Merge Tables
  -> Select Table Columns
  -> Add Metadata Columns
  -> Summarize Measurements or CSV/TSV export
```

This is the current path for PCA/treatment-group style analysis. The table
nodes are deliberately generic so colocalization/localization tables can join
through the same identity and metadata columns.

Colocalization is a separate analysis family. The current workflow outputs
whole-image and ROI-masked metrics, inspector scatter-density threshold
controls, thresholded colocalized-voxel images, RACC index images, and
object-aware tables. `Object Colocalization Metrics`, `Label Overlap
Association`, `Nearest Object Distance`, and `Event Localization` are designed
to join through the same table-composition path described above.

## Reference Workflows

See [../examples/README.md](../examples/README.md) for all bundled workflows.
The key measurement examples are:

- `red-channel-object-intensity-measurements.json`;
- `red-channel-merged-measurement-table.json`;
- `synthetic-measurement-summary.json`;
- `synthetic-derived-object-morphology.json`;
- `synthetic-3d-mesh-morphology.json`;
- `synthetic-skeleton-qc.json`;
- `synthetic-advanced-skeleton-network.json`;
- `synthetic-colocalization-racc.json`;
- `synthetic-object-colocalization-association.json`.

## Current Wrap-Up Status

Completed for this phase:

- table output type and table previews;
- table CSV/TSV saving;
- object, intensity, skeleton, branch, graph, summary, and 3D mesh morphology
  measurements;
- calibrated physical variants for extended non-mesh object morphology;
- analytical phantom validation for calibrated morphology;
- table merge, metadata annotation, column selection, and grouped summaries;
- first-pass pixel colocalization/RACC metrics, masked/ROI-restricted variants,
  and visual outputs;
- per-object colocalization, object association, nearest-distance, and
  event-localization tables;
- deterministic synthetic validation samples;
- example workflows for each major measurement family;
- full example-workflow tests.

Remaining non-blocking follow-up:

- optional mesh export/preview;
- specialist mitochondrial network indices;
- broader cancellation and percentage progress coverage for long manual
  calculations.
