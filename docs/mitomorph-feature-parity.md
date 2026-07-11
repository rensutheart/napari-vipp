# MitoMorph Feature Parity And Measurement Roadmap

Status: current specialist gap tracker; implemented items are marked in place

Last reviewed: 2026-07-10

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
- For expensive feature families, use the implemented manual calculation model
  rather than live recomputation: show `Calculate`/`Recalculate`, stale-state
  warnings after upstream changes, and the last valid cached result until
  recomputed. Add cancellation and percentage progress when individual
  libraries expose that information.
- Avoid presenting 2D-only measurements as meaningful in 3D. The UI should hide
  or clearly label dimension-specific metrics.

Proposed measurement families:

1. **Basic Object Morphology**
   Label id, area/volume, physical area/volume, centroid, bounding box,
   equivalent diameter, extent, Euler number. This is already implemented as
   the first `Measure Objects` set.

2. **Extended Region Properties**
   Baseline selectable groups are implemented in `Measure Objects` and
   `Measure Objects + Intensity`: filled and bounding-box area/volume,
   major/minor axis length, inertia tensor eigenvalues, 2D orientation,
   2D eccentricity, 2D perimeter, 2D Crofton perimeter, 2D convex area,
   2D solidity, 2D maximum Feret diameter, derived shape ratios, 2D Hu
   moments, Crofton-based circularity, perimeter-to-area ratio, and calibrated
   physical variants for supported extended length/shape columns. Robust 3D
   mesh morphology is implemented as a separate table node.
   The concrete implementation plan is tracked in
   [object-mesh-morphology-plan.md](object-mesh-morphology-plan.md).

3. **Intensity Measurements**
   Baseline `labels + intensity image -> table` support is implemented in
   `Measure Objects + Intensity` with mean, minimum, maximum, sum/integrated
   intensity, and standard deviation. Remaining intensity work includes
   selected percentiles and optionally background-corrected variants.

4. **3D Mesh-Based Morphology**
   Convert each 3D label object to a surface mesh when requested. Measure mesh
   surface area, mesh volume, mesh extents, convex hull surface area, convex
   hull volume, convexity, solidity, sphericity/form factor, principal inertia
   components where clearly defined, and mesh-derived axis-length ratios.
   Preserve physical scale and anisotropic z spacing. The first VIPP version
   should produce a table only and use the existing `scikit-image`/`scipy`
   stack; `trimesh`, `porespy`, mesh export, and visual panels should come
   after the numeric metrics are stable and their need is concrete.

5. **Skeleton And Network Measurements**
   `Skeletonize`, `Analyze Skeleton`, skeleton keypoint masks,
   skeleton graph overlays, component/branch label images, short-branch
   pruning, row-per-branch measurements, explicit graph node/edge table export,
   branch-summary distributions, and per-block normalized network summaries
   are implemented. Future work should add specialist mitochondrial
   connectedness indices when those metrics have clear biological definitions
   beyond the generic graph summaries.

6. **Localization And Colocalization Measurements**
   Add table-producing nodes for pixel-based colocalization, object-based
   association, nearest-neighbor distance, object/event localization relative
   to labels or masks, and optional ROI-restricted measurements. These should
   preserve object identity columns so they can be merged with morphology,
   intensity, mesh, and skeleton features.

7. **Mitochondrial Time-Lapse Events**
   MitoMorph-specific fission, fusion, depolarization, event localization, and
   duplicate suppression should become a later specialist workflow family. This
   likely requires object tracking/association and should not be folded into
   generic static object measurement.

## Table Combination Requirement

VIPP needs a table-combination path before broad feature extraction becomes
pleasant:

- `Merge Tables`: implemented; joins tables by stable object identity columns
  and falls back to row-position joining for equal-length tables.
- `Select Table Columns`: implemented; keep/drop/reorder measurement columns
  before export while preserving row order and column units.
- `Add Metadata Columns`: implemented; adds treatment, replicate, condition,
  timepoint, or batch metadata columns.
- `Summarize Measurements`: group by condition/time/source and calculate
  mean, median, standard deviation, count, and quantiles. Implemented.

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
4. Add property-based label cleanup. Implemented as `Filter Labels By
   Property`, using existing measurement-table columns to keep or remove label
   IDs before downstream measurement or export.
5. Expand `Measure Objects` with selectable extended region-property groups.
   Implemented as checkbox groups for shape descriptors, axis/inertia
   descriptors, and 2D boundary descriptors.
6. Add table column selection and grouped summaries. Implemented as
   `Select Table Columns` and `Summarize Measurements`.
7. Add 3D mesh morphology as an opt-in node because it is more expensive and
   has stronger assumptions about anisotropy, surface extraction, and object
   size. Implemented as `Measure 3D Mesh Morphology`, separate from the cheaper
   regionprops-derived object measurements.
8. Use the implemented manual/cached expensive-node execution model for broad
   mesh, graph, colocalization, or restoration calculations that would feel
   sluggish if run live.
9. Add colocalization/localization table nodes that can merge with object
   measurement tables. Implemented as object colocalization, label-overlap,
   nearest-distance, and event-localization table nodes.
10. Add specialist mitochondrial network metrics beyond the implemented
    skeleton branch-summary tables and normalized overall-network summaries.
11. Add mitochondrial event analysis after tracking/association design is in
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
