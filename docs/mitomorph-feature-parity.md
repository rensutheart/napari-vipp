# MitoMorph Feature Parity And Measurement Roadmap

Last reviewed: 2026-06-16

This document captures the MitoMorph-derived capabilities that should become
first-class VIPP workflows. The goal is not to copy the old implementation
directly. The goal is to preserve the scientific intent: extract broad,
well-defined object, intensity, surface, and network measurements so users can
compare treatment groups with downstream statistics such as PCA, clustering,
classification, and regression.

## Core Measurement Goal

VIPP should support high-dimensional feature extraction from segmented objects.
A user should be able to choose which measurement families to compute, then
combine the selected measurements into one final per-object results table.

The final table must preserve stable object identity columns so different
measurement nodes can be joined reliably:

- source item or batch item id;
- time index, channel index, z/block context where applicable;
- label id;
- optional object UUID if labels are relabeled or tracked later.

This object table should be suitable for CSV/TSV export, OME-Zarr table export
later, and direct use in PCA/treatment-group analysis.

## Recommended UX Model

Use both focused nodes and selectable measurement groups.

- Keep cheap, common measurements in `Measure Objects`.
- Add optional checkboxes or measurement-set selectors inside measurement nodes
  for feature groups that are conceptually related.
- Put expensive or dimension-specific computations in separate nodes so users
  can opt into them deliberately.
- Avoid presenting 2D-only measurements as meaningful in 3D. The UI should hide
  or clearly label dimension-specific metrics.

Proposed measurement families:

1. **Basic Object Morphology**
   Label id, area/volume, physical area/volume, centroid, bounding box,
   equivalent diameter, extent, Euler number. This is already implemented as
   the first `Measure Objects` set.

2. **Extended Region Properties**
   Filled area/volume, bounding-box area/volume, convex area/volume where
   valid, solidity, major/minor axis length, inertia tensor eigenvalues,
   orientation, eccentricity for 2D, perimeter for 2D, Feret diameter where
   available, Hu moments for 2D, and form-factor/circularity-style metrics.

3. **Intensity Measurements**
   Requires named inputs: `labels + intensity image -> table`. Include mean,
   minimum, maximum, sum/integrated intensity, standard deviation, selected
   percentiles, and optionally background-corrected versions.

4. **3D Mesh-Based Morphology**
   Convert each 3D label object to a surface mesh when requested. Measure mesh
   surface area, mesh volume, mesh extents, convex hull surface area, convex
   hull volume, convexity, solidity, sphericity/form factor, principal inertia
   components, and mesh-derived axis-length ratios. Preserve physical scale and
   anisotropic z spacing.

5. **Skeleton And Network Measurements**
   The base `Skeletonize` and `Analyze Skeleton` nodes are implemented. Future
   work should add branch length distributions, tortuosity, branch labels,
   endpoint/junction QC masks, short-branch pruning, explicit graph export, and
   domain-normalized connectedness metrics.

6. **Mitochondrial Time-Lapse Events**
   MitoMorph-specific fission, fusion, depolarization, event localization, and
   duplicate suppression should become a later specialist workflow family. This
   likely requires object tracking/association and should not be folded into
   generic static object measurement.

## Table Combination Requirement

VIPP needs a table-combination path before broad feature extraction becomes
pleasant:

- `Merge Tables`: implemented; joins tables by stable object identity columns
  and falls back to row-position joining for equal-length tables.
- `Select Table Columns`: keep/drop/reorder measurement columns before export.
- `Add Metadata Columns`: implemented; adds treatment, replicate, condition,
  timepoint, or batch metadata columns.
- `Summarize Measurements`: group by condition/time/source and calculate
  mean, median, standard deviation, count, and quantiles.

These nodes are required for the intended PCA/treatment-group workflow because
users will compute different measurement families in separate graph branches
and then assemble them into one analysis-ready table.

## Implementation Order

1. Add named heterogeneous input ports so nodes can accept `labels`, `image`,
   `mask`, and `table` inputs explicitly. Implemented for graph execution,
   visual ports, workflow JSON, and Python export.
2. Add intensity-aware object measurements. Implemented as `Measure Objects +
   Intensity` with per-object mean, minimum, maximum, sum, and standard
   deviation intensity.
3. Add table merge and metadata annotation. Implemented as `Merge Tables` and
   `Add Metadata Columns`.
4. Expand `Measure Objects` with selectable extended region-property groups.
5. Add table column selection and grouped summaries.
6. Add 3D mesh morphology as an opt-in node because it is more expensive and
   has stronger assumptions about anisotropy, surface extraction, and object
   size.
7. Add skeleton QC/pruning/branch-label outputs.
8. Add mitochondrial event analysis after tracking/association design is in
   place.

## Scientific Cautions

- Mesh-derived measurements depend on thresholding, voxel anisotropy, surface
  extraction method, and small-object handling.
- Convexity, solidity, and form factor are useful exploratory features, but
  their biological interpretation should be workflow-specific.
- PCA and treatment-group separation require reproducible preprocessing and
  consistent feature definitions across all samples.
- Optional feature groups should record their settings in table metadata and
  exported workflow provenance.
