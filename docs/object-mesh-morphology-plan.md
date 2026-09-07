# Object And Mesh Morphology Plan

Status: implementation record; phases 1-3 complete; object-aware phase-4 workflow implemented, unreleased

Last reviewed: 2026-09-07

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

The object-aware surface viewing/export path is implemented (phase 4), including
per-label identity, colours, collection operations, refinement and 3MF.
Specialist mesh metrics, Boolean union and repair remain future work.

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

The initial implementation after 0.15.0a1 was **Mask to 3D Mesh**
(`mask_to_3d_mesh`), originally under Morphology and now under **3D Meshes**.
Extraction uses the existing scikit-image dependency; napari owns Surface
presentation; `core/meshes.py` owns immutable geometry and the shared writer.
The subsequent object-aware extension and its dependencies are detailed below.

- Input: one Boolean 3D volume with explicit Z/Y/X spatial axes, in any order.
  Canonicalize axes and calibration together to ZYX. Reject 2D, singleton
  spatial dimensions and unresolved/channel/time dimensions. Raw function
  calls without metadata use documented ZYX.
- Extraction: Lewiner marching cubes, level 0.5, step 1, no smoothing/decimation.
  Default border closure pads one background voxel and shifts vertices back;
  the open choice makes no outside-background assumption. Do not label, merge,
  repair or derive quantitative metrics here.
- Output: immutable `MeshData` (owned read-only N×3 float64 vertices and M×3
  int64 triangle indices) plus `MeshState`. Coordinates are voxel-centre ZYX;
  calibration and source history remain separate. It is not an ImageState or
  an image-compatible graph port. Bypass cannot substitute a mask for a mesh.
- Execution: manual/cached CPU; full-volume extraction can consume substantial
  memory. Progress and cancellation checkpoints surround the opaque extraction
  call. Input pixels are unchanged. Empty surfaces carry zero counts; napari
  presentation removes old geometry rather than constructing dummy vertices.
- Presentation: native Surface with independent buffers, source scale/origin,
  3D inspection and pinning. Smooth lighting is not geometric smoothing. No
  image histogram or thumbnail is generated for a mesh.
- OBJ: stream calibrated XYZ coordinates and 1-based triangle indices. Reverse
  winding when swapping coordinate handedness; outward orientation is tested.
  Convert compatible length units to the X unit. Reject incompatible units.
  Embed units, calibration, boundary and history as JSON comments in the same
  atomically published file. OBJ has no standard unit field. Keep absolute
  origin; never silently adapt scientific dimensions for printing.
- Manual saving, Batch Output (default OBJ for mesh), and generated Python
  publication share the writer. Existing batch skip/overwrite and source
  verification boundaries remain in force. Empty mesh exports fail explicitly.

Tests: `test_meshes.py` covers geometry, axis permutations, calibration, border
behavior, empty/invalid inputs, orientation, cancellation, atomic publication,
workflow round trips, native Surface inspection/pinning, batch and generated
Python. These regression checks do not establish biological or print validity.

### Existing-mesh measurement (unreleased)

`measure_3d_mesh_morphology` accepts `mask_or_labels_or_mesh`. Integer label
measurement is unchanged; a Boolean mask is a single foreground ID per spatial
block. Mesh inputs use the supplied vertices/faces directly on CPU, one row per
explicit object, with no marching cubes, voxel reconstruction or minimum
voxel filter. The mesh row uses `mesh_id`, vertex/triangle counts and
`watertight`, not label IDs or invented voxel counts. Compatible spatial units
are converted to the X-axis unit; incompatible units are rejected.

Triangle cross products give area; signed tetrahedral sums give enclosed
volume, preserving cavity orientation. Boundary/non-manifold edges,
inconsistent winding, duplicate faces and zero-area triangles suppress
volume-dependent metrics, retaining area/extents and explicit status/error.
These checks do not prove absence of self-intersections. No geometry is
repaired. Optional convex-hull metrics use the same SciPy helper as the label
path. `test_mesh_input_measurements.py` covers analytical geometry, calibration,
cavities, input immutability, invalid surfaces, workflow/export and graph cards.

### Object-aware mesh workflow (unreleased)

Implemented under **3D Meshes**, immediately after Morphology. Subgroups:
**Create surfaces**, **Objects & colours**, **Refine geometry**. Measurement
stays under Measurements. All geometry/object nodes are manual/cached CPU.

- **Mask to 3D Mesh** retains its single-object default; explicit **Connected
  objects** uses 6-connected foreground voxels. **Labels to 3D Mesh** extracts
  positive integer IDs independently, preserving int64 identities and cavity
  winding. Both require one explicit non-singleton Z/Y/X volume, no leading
  dimensions, and use full-resolution Lewiner marching cubes.
- `MeshData` adds immutable face IDs and sorted `MeshObject` records: name,
  RGBA, parent/source ID, source axes/scale/units/origin and history. Empty
  collections contain no objects. **Colour Mesh Objects** chooses distinct ID
  colours or a fresh volume, area, sphericity or triangle-count gradient.
  Missing measurements are grey; colouring changes no coordinates/metrics.
  Gradients are normalized per collection, not comparable scales across runs.
- **Combine Meshes** accepts 2–16 inputs and converts compatible units into
  the first input's coordinate frame. Duplicate IDs are remapped with retained
  lineage and colours. It never welds vertices or unions overlapping solids.
  Mixed physical/uncalibrated units are rejected rather than guessed.
- **Split Mesh Objects** separates shared-edge connected triangle components
  within each object; child IDs retain their parent and colour. Cavity shells
  can become separate objects: this operation is not biological segmentation.
  **Filter Mesh Objects** uses current volume, area, sphericity, triangle count
  or ID; inside range is inclusive, outside is exclusive. Missing/non-finite
  values are excluded in both modes. No external measurement-table input yet.
- **Smooth Mesh** applies a two-pass Taubin-style filter in physical space
  with fixed inverse initial-edge-length weights. Controls: iterations, strength
  and boundary preservation. Lambda is half the strength; mu uses passband
  0.1. Zero strength is an exact no-op. **Simplify Mesh** uses the native
  `fast-simplification` QEM provider, with percentage of triangles to keep,
  aggressiveness and boundary preservation. Each disconnected component is
  handled separately, so small shells are not silently lost; reduction targets
  are not guaranteed under topology/boundary constraints.
- Refinement creates new immutable buffers and records parameters/provider.
  Degenerate triangles and inconsistent/non-manifold edges are rejected; output topology,
  winding and boundary checks reject unsafe changes rather than repairing them.
  Volume, shape and self-intersection freedom are not guaranteed. Source mesh
  remains available for comparison; measurements always use current geometry.
- Native napari Surface receives detached object-coloured presentation buffers
  and common-unit scale/origin. Shared vertices are duplicated for display only.
  Cached scientific arrays and IDs cannot be mutated by viewer colour changes.
- **3MF** uses core surface objects, base-material RGBA, components and one build
  assembly retaining positions. Calibrated physical units are mandatory; tiny
  units are converted to supported micron units. Scientific int64 IDs, original
  float colours, calibration and processing/source history are embedded as JSON.
  No automatic scale-to-print, packing or repair. Surface export is not a
  printability certificate. Core material channels are quantized to 8-bit.
  OBJ retains geometry/object groups and embedded metadata; standard OBJ alone
  does not carry VIPP colours or units. Both use atomic publication and existing
  source-revision, skip/overwrite and cancellation boundaries.

Dependencies: Matplotlib for colour maps and
[`fast-simplification`](https://pyvista.github.io/fast-simplification/) 0.2.x
for QEM. Standard-library ZIP/XML implements
[3MF Core](https://github.com/3MFConsortium/spec_core); no trimesh dependency.
The 3MF writer was checked against the official Core XSD, including open and
closed multi-object surfaces. This establishes structure, not printer support.

Regression modules: `test_mesh_objects.py`, `test_mesh_refinement.py`,
`test_mesh_3mf.py`, `test_mesh_workflow_integration.py` and
`test_mesh_objects_example.py`. The synthetic five-object workflow exercises
colours, disjoint size filtering/recombination, refinement, per-object
measurement and 3MF output without modifying its source.

## Deferred

- Geometric union, external measurement-table colouring, PLY/STL, marching tetrahedra, oriented
  bounding boxes, principal mesh inertia and mesh repair; introduce `trimesh`
  or another dependency only for a concrete capability gap;
- `porespy`-backed pore-network, pore-size, local-thickness, chord-length, or
  porous-media metrics;
- object tracking and fission/fusion event features;
- mesh repair/smoothing as default behavior;
- treating 2D boundary measurements and 3D surface measurements as if they were
  directly interchangeable.
