# Object And Mesh Morphology Plan

Status: implementation record; phases 1-3 complete, phase 4 deferred

Last reviewed: 2026-07-10

This plan covers the next morphology milestone after the skeleton/network
work. It is informed by the old MitoMorph `morphology.py` implementation, but
the VIPP version should use explicit, composable nodes rather than one
monolithic "calculate all parameters" function.

## MitoMorph Reference

The old MitoMorph morphology code did four main things:

- label and filter binary stacks with `scipy.ndimage.label`;
- measure objects with `skimage.measure.regionprops`;
- convert 3D object coordinates to meshes with `trimesh`;
- calculate exploratory 3D morphology such as mesh surface area, mesh volume,
  convex-hull area/volume, solidity, sphericity/form factor, mesh extents, and
  inertia-derived axis ratios.

Useful ideas to preserve:

- broad feature extraction for PCA/treatment-group comparison;
- explicit 3D surface and convex-hull measurements;
- physical scaling, especially anisotropic Z spacing;
- optional expensive feature families, not always-on live recalculation.

Things not to copy directly:

- global function that mixes labeling, measurement, mesh export, plotting, and
  Excel writing;
- hard-coded paths and reporting panels;
- broad `except` blocks that hide failed geometry calculations;
- ambiguous column names such as "relative" when the reference frame is really
  physical voxel spacing;
- axis-length formulas that are not clearly tied to a standard metric.

## VIPP Current Baseline

Already implemented:

- `Measure Objects`: labels -> table using `skimage.measure.regionprops_table`;
- `Measure Objects + Intensity`: labels + image -> table;
- basic morphology: label id, area/volume, physical area/volume, centroid,
  bounding box, equivalent diameter, extent, Euler number;
- optional regionprops groups: bounding/fill size, 2D convex area, 2D solidity,
  maximum Feret diameter, major/minor axis length, inertia eigenvalues,
  eccentricity/orientation, 2D perimeter and Crofton perimeter;
- optional derived shape ratios: axis ratio, bounding-box side lengths,
  bounding-box aspect ratios, fill fraction, and inertia eigenvalue ratios;
- optional 2D shape moments: Crofton-based circularity, perimeter-to-area
  ratio, and Hu moments;
- `Merge Tables`, `Select Table Columns`, `Summarize Measurements`,
  `Add Metadata Columns`, and `Filter Labels By Property`.

The current gap is not "run regionprops". The gap is:

- optional mesh export/preview and later specialist mesh metrics beyond the
  implemented first-pass table measurements.

## Implementation Strategy

Use two complementary paths.

1. Extend existing measurement nodes for cheap derived regionprops features.
   These should remain table-producing manual nodes with the existing checkbox
   model.
2. Use the dedicated `Measure 3D Mesh Morphology` node for expensive
   marching-cubes and convex-hull features. It is manual/cached by default and
   does not run live unless the user explicitly enables auto recalculation.

This keeps everyday object measurements fast while making the expensive 3D
surface feature family explicit.

## Phase 1: Richer Regionprops-Derived Object Morphology

Status: derived shape ratios, 2D shape moments, and calibrated physical variants
are implemented as checkbox groups/metadata-aware columns on `Measure Objects`
and `Measure Objects + Intensity`.

Implemented groups:

- **Derived shape ratios**
  - `axis_ratio_major_minor`;
  - 3D inertia/eigenvalue ratios where available, using explicit names such as
    `inertia_eigval_ratio_0_1`, not "main/middle" unless axis ordering is
    documented;
  - bounding-box side lengths in pixels/voxels;
  - bounding-box aspect ratios;
  - fill fraction inside the bounding box.
- **2D boundary shape**
  - circularity/form factor using Crofton perimeter:
    `4*pi*area/perimeter_crofton^2`;
  - perimeter-to-area ratio;
  - Hu moments as separate columns `hu_moment_0` ... `hu_moment_6`;
  - keep 2D-only features hidden/disabled for true 3D spatial blocks.

Implemented calibrated physical variants:

- **Calibrated physical variants**
  - physical centroid coordinates;
  - physical bounding-box min/max coordinates and side lengths;
  - physical equivalent diameter;
  - physical bounding-box and filled area/volume;
  - physical 2D convex area and maximum Feret diameter;
  - physical major/minor axis lengths;
  - physical inertia tensor eigenvalues;
  - physical perimeter, Crofton perimeter, and perimeter-to-area ratio for
    isotropic 2D pixels.

Implementation notes:

- Preserve stable identity columns: source, leading axis indices, and
  `label_id`.
- Avoid silently reporting misleading calibrated lengths under anisotropic
  spacing. Physical 2D perimeter columns remain `NaN` for anisotropic pixels
  because `skimage.measure.regionprops_table` only supports perimeter with
  isotropic spacing.
- Keep invalid measurements as `NaN` plus clear documentation, rather than
  dropping rows or columns.

## Phase 2: New `Measure 3D Mesh Morphology` Node

Status: implemented as a manual table node:

```text
labels -> Measure 3D Mesh Morphology -> table
```

Purpose:

- measure 3D object surface/mesh features that regionprops either does not
  expose or does not calibrate clearly enough for anisotropic microscopy data.

Implemented backend:

- first implementation should use the existing VIPP dependency stack:
  `scikit-image`, `scipy`, and local NumPy helpers only;
- use `skimage.measure.marching_cubes` with `spacing=(z, y, x)` for surface
  extraction;
- use `skimage.measure.mesh_surface_area` for mesh surface area;
- use a small local helper for signed triangle mesh volume;
- use `scipy.spatial.ConvexHull` for convex hull area/volume;
- do not add `trimesh` for the first table-metric implementation. Reconsider it
  only when mesh export, mesh preview/rendering, oriented bounding boxes,
  principal mesh inertia, or mesh repair become first-class features;
- do not add `porespy` for this milestone. Treat it as a later specialist
  porous/network analysis dependency if pore-size, local-thickness, chord, or
  pore-network extraction metrics become explicit goals.

Implemented parameters:

- `Spatial processing`: auto/3D only. The node should clearly explain that it
  needs true 3D spatial data.
- `Include convex hull metrics`: default on, but failure-tolerant.
- `Minimum voxel count`: skip very small labels that cannot produce stable
  meshes.
- `Smoothing`: not implemented yet. Add only if the measurement effect is
  documented.
- `Auto Recalculate`: inherited manual-node option, off by default.

Implemented columns:

- identity/context:
  - leading axis indices such as `t_index` and `c_index`;
  - `label_id`;
  - `mesh_status`;
  - `mesh_error` for failed objects;
- size:
  - `voxel_volume`;
  - `voxel_volume_physical`;
  - `mesh_volume_physical`;
  - `mesh_surface_area_physical`;
  - `surface_area_to_volume`;
  - `equivalent_sphere_diameter_physical`;
- extents:
  - `mesh_extent_z_physical`;
  - `mesh_extent_y_physical`;
  - `mesh_extent_x_physical`;
  - axis-aligned extent ratios with explicit names;
- convex hull:
  - `convex_hull_volume_physical`;
  - `convex_hull_surface_area_physical`;
  - `solidity_3d = mesh_volume_physical / convex_hull_volume_physical`;
  - `surface_area_to_convex_hull_area`;
  - optionally `convex_hull_area_to_surface_area` if the old MitoMorph-style
    ratio is useful, but name it explicitly;
- shape:
  - `sphericity = pi^(1/3) * (6V)^(2/3) / A`;
  - optional mesh inertia components only if the definition and units are clear.

Failure handling:

- objects with too few voxels, flat geometry, invalid marching cubes, or failed
  convex hulls should still produce a table row;
- failed metric groups should be `NaN` and `mesh_status` should state the
  reason;
- convex-hull failure should not discard surface-area or mesh-volume results.

## Phase 3: Synthetic Validation Data And Example Workflow

Status: implemented as `VIPP synthetic 3D mesh morphology` plus
`examples/synthetic-3d-mesh-morphology.json`.

The deterministic sample is designed for morphology validation:

- a sphere-like object;
- an ellipsoid with anisotropic Z spacing;
- a cuboid or rectangular prism;
- a concave object or dumbbell shape where convex-hull metrics differ from
  mesh metrics;
- a tiny object that should trigger the minimum-voxel/unstable-mesh path.

Implemented workflow:

```text
synthetic 3D morphology image
  -> Binary Threshold
  -> Label Connected Components
  -> Measure Objects
  -> Measure 3D Mesh Morphology
  -> Merge Tables
  -> Select Table Columns
```

Tests assert:

- expected object count;
- calibrated physical volume follows axis scale;
- sphere-like object has higher sphericity than cuboid/concave objects;
- concave object has lower 3D solidity than convex objects;
- the tiny object is reported with a clear non-success status when below the
  threshold;
- the workflow loads and runs.

## Phase 4: Optional Mesh Export And Visualization

Do this only after table metrics are stable.

Possible nodes:

- `Object Mesh Preview`: labels -> RGB/points/surface-like QC output where
  feasible in napari;
- `Export Object Meshes`: labels -> files, probably OBJ/PLY/STL;
- optional `trimesh` dependency for robust export and oriented bounding boxes.

This should not block the first mesh-morphology table node.

## Deferred

- `trimesh`-backed mesh export, mesh preview/rendering, oriented bounding
  boxes, principal mesh inertia, and mesh repair;
- `porespy`-backed pore-network, pore-size, local-thickness, chord-length, or
  porous-media metrics;
- object tracking and fission/fusion event features;
- mesh repair/smoothing as default behavior;
- treating 2D boundary measurements and 3D surface measurements as if they were
  directly interchangeable.
