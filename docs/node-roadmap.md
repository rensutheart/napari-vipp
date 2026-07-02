# Bioimage Node Roadmap

Status: working discussion document  
Last reviewed: 2026-07-01

This document prioritizes future VIPP nodes for bioimage analysis. It is not a
commitment to reproduce every function in OpenCV, scikit-image, or SciPy.
Instead, the goal is to support complete, understandable analysis workflows
while preserving image axes, physical scale, data type, and provenance.

The current recommendation is to build a strong classical segmentation and
measurement workflow before adding a large catalogue of specialized filters.
MitoMorph-derived feature extraction requirements, including selectable
measurement families and final per-object table combination for PCA/treatment
group analysis, are tracked in
[mitomorph-feature-parity.md](mitomorph-feature-parity.md).

## Confirmed Product Scope

The initial bioimage scope should support all of these workflow families:

- nuclei and cell segmentation;
- puncta and spot detection;
- mitochondrial morphology and network analysis;
- colocalization analysis.

Spatial processing must support both:

- true 3D fluorescence `ZYX` volumes, including anisotropic voxel spacing;
- 2D images, including flow-cytometry imaging data.

Registration and deconvolution remain important later milestones, but they are
not part of the current implementation focus.

The immediate focus is:

```text
binary mask
  -> label structures
  -> remove/filter labeled structures by volume
  -> preserve cleaned integer labels for inspection and later measurement
```

Implementation status:

- implemented: first-class `labels` graph/display type;
- implemented: Label Connected Components with 2D/3D spatial modes and
  configurable face/full connectivity;
- implemented: Filter Labels By Volume using pixel/voxel counts while
  preserving retained IDs, with a data-aware slider and input-volume
  distribution;
- implemented: Clear Border Objects for masks and labels, with a border buffer
  and all-volume or lateral-only 3D boundaries;
- implemented: Fill Holes with metadata-aware 2D/3D processing, an advanced
  per-slice mode, connectivity, and optional maximum hole area/volume;
- implemented: Remove Small Objects for masks and labels, with type-preserving
  minimum-size filtering and contextual area/volume controls;
- implemented: Relabel Sequential;
- implemented: table-driven property-based label filtering;
- implemented: selectable extended region-property groups;
- implemented: table column selection/reordering;
- implemented: skeleton QC masks, component labels, branch labels, and
  short-branch pruning.

## Priority Definitions

| Priority | Meaning |
| --- | --- |
| P0 | Platform capability required by several important nodes |
| P1 | Implement next; high value for common bioimage workflows |
| P2 | Useful after the core segmentation and measurement path works |
| P3 | Specialized, expensive, or dependent on additional product decisions |
| Defer | Do not implement now unless a concrete workflow requires it |

## Decision Principles

1. Prefer complete workflows over library feature parity.
2. Treat `T`, `C`, `Z`, `Y`, and `X` axes explicitly.
3. Make slice-wise versus volumetric processing a visible choice.
4. Use physical spacing when an operation represents distance, size, or volume.
5. Keep binary masks, integer labels, intensity images, tables, and points
   semantically distinct.
6. Prefer SciPy and scikit-image while they cover the requirement. OpenCV
   should become a dependency only for a demonstrated capability or performance
   gap.
7. Every node must remain deterministic, testable headlessly, serializable in a
   workflow, and exportable to Python.
8. Avoid several nearly identical nodes when one well-designed node with a
   method selector is clearer.
9. Expensive nodes should be explicit about execution state. If recomputing
   live would make the UI stutter or hide long-running work, prefer a
   manual/cached node with progress and stale-state indicators.

## Current Coverage

VIPP already has useful coverage in these areas:

- image sources, saves, workflow persistence, and Python export;
- channel extraction, splitting, combination, and RGB display conversion;
- crop and axis slicing;
- dtype conversion, normalization, clipping, and intensity rescaling;
- typed Mask Image, image arithmetic, and logical operations;
- Gaussian, mean, median, and bilateral filtering;
- fixed, Otsu, Triangle, Li, Yen, Isodata, Minimum, Hysteresis, Sauvola,
  Niblack, adaptive mean/Gaussian, and local thresholding;
- Sobel, Canny, Laplace, Difference of Gaussians, Unsharp Mask, Non-Local
  Means, and other filtering/detail nodes;
- binary erosion, dilation, opening, closing, fill holes, and small-component
  removal;
- maximum projection, general axis-aware projection, and orthogonal projection;
- metadata calibration repair with `Set Pixel Size / Units`;
- X/Y/Z scale-factor resampling with `Rescale Axes`;
- image metadata, thumbnails, intensity histograms, label-volume histograms,
  image/mask/label inspection, and image/mask/label pinning;
- connected-component labels, pixel/voxel-volume filtering, sequential
  relabeling, table outputs, basic and intensity-aware label-object
  measurements, table merge/annotation, skeletonization, skeleton-network
  measurement tables, skeleton keypoint masks, branch/component labels, and
  short-branch pruning.

The remaining object-analysis gaps are difficult segmentation, richer
measurement utilities, calibrated physical filtering, and broader expensive
measurement execution control. VIPP can now label separated
foreground objects, clean them by size or measured properties, measure basic
and selected extended label morphology plus intensity, skeletonize masks,
measure skeleton components with generic graph metrics, export skeleton graph
node/edge tables, summarize skeleton branches, visualize skeleton
keypoints/branches/components, and prune short terminal branches, but it
cannot yet:

- provide robust seeded segmentation presets beyond the current watershed
  building blocks and defaults;
- use calibrated physical area/volume directly as a filter unit;
- provide cancellation and percentage progress for long cached metrics or
  restorations.

## Existing Nodes To Clarify

Before growing the catalogue, several current names or contracts should be
cleaned up.

| Current node | Issue | Recommended direction |
| --- | --- | --- |
| Linear Scale + Offset | Dtype preservation is implemented, but users may still confuse it with contrast stretching. | Keep the explicit math name; use `Rescale Intensity` when the desired output range is the main control. |
| Top Hat / Black Hat | These are binary operations, while grayscale white top-hat is a common bioimage background-removal tool. | Rename current nodes `Binary Top Hat` and `Binary Black Hat`; add explicit grayscale morphology later. |
| Maximum Projection | Axis is numeric and only the maximum reducer is available. | Keep as a quick MIP shortcut; prefer `Project Image` for semantic axes and reducer choice. |
| 2D filtering behavior | Several nodes infer XY and process other axes plane by plane. | Add an explicit processing scope: `XY per plane` or `spatial volume`. |

## P0: Platform Foundations

These are not glamorous nodes, but they prevent important algorithms from being
implemented with misleading types or awkward parameters.

### First-Class Label Images: Implemented

The `labels` port/output type represents non-negative integer object IDs where
`0` is background. Labels inspect and pin as napari Labels layers and are not
treated as ordinary intensity data. Image outputs can also be pinned as napari
Image layers; boolean mask pins remain Labels overlays.

Implemented behavior:

- labels connect to label-aware nodes and generic array nodes;
- labels do not connect silently to binary-mask-only nodes;
- connected-component and filter operations record their parameters in
  operation history;
- 2D-per-plane versus volumetric processing is explicit and exportable;
- save and inspect actions preserve integer IDs.

Nearest-neighbor interpolation remains a requirement for future geometry and
registration nodes that accept labels.

### Named Heterogeneous Input Ports: Implemented

Nodes can declare named, typed input ports and keep those slots through visual
wiring, workflow JSON, Python export, and execution. Current examples include
`Measure Objects + Intensity` and `Mask Image`.

Marker-controlled watershed can now use ports such as:

| Port | Type |
| --- | --- |
| Elevation | image |
| Markers | labels |
| Mask | mask, optional |

Optional input slots are not implemented yet; watershed should initially use
required elevation/marker inputs and add optional masks once optional ports are
designed.

### Table Outputs: Implemented

The first `table` output type is implemented through `TableData` and
`TableState`. It has:

- stable column names and units;
- one row per object or observation;
- an inspector table view;
- CSV/TSV save support;
- workflow/export support;
- no image thumbnail, histogram, or napari layer inspection behavior.

Pandas is intentionally not required for core measurement or export. Optional
linkage from a label column back to a displayed Labels layer remains future UI
work.

### Slow Node Execution And Progress State: Baseline Implemented

Some nodes should not execute continuously on every upstream or parameter
change. The baseline manual/cached model is now implemented for
`Measure Objects`, `Measure Objects + Intensity`, `Analyze Skeleton`, and
`Measure Skeleton Branches`, `Skeleton Graph Tables`, and `Summarize Skeleton
Network`. Future examples include 3D mesh morphology, colocalization/
localization over large stacks, deconvolution, and expensive background
estimation.

The implemented UX model is:

- the node card and inspector expose `Calculate` or `Recalculate`;
- the inspector exposes `Auto Recalculate`, off by default, for manual nodes
  that are fast enough to update live for the current data;
- the node keeps its last valid result as its current output;
- changing an upstream input or relevant parameter marks the result as stale;
- stale nodes remain usable downstream but show a visible warning state;
- node-card state colours are gray for not calculated, green for current,
  orange for stale, and red for error;
- active calculations show the existing background busy ring and inspector
  status;
- the UI remains responsive while the operation runs;
- exported Python and batch execution recompute deterministically rather than
  depending on an interactive cached value.

Remaining TODO: cancellation, percentage progress for operations that expose
progress, polished failure recovery, and broader adoption by future expensive
feature families. Workflow JSON persists node settings and graph structure,
including the auto-recalculate preference, but deliberately does not store large
cached arrays or tables.

### Spatial Scope And Units: Partially Implemented

Label operations currently expose `Auto from axes`, `2D YX`, and `3D ZYX`.
The broader shared contract should become:

- `XY per plane`: repeat independently over non-spatial dimensions;
- `spatial volume`: operate over the recognized spatial axes;
- size units: `pixels/voxels` initially, with `physical units` enabled when
  axis scale metadata is reliable;
- connectivity: explicit and dimension-aware;
- anisotropic spacing: passed to distance, expansion, and measurement functions
  where supported.

### Points Output, Later In P0: Not Implemented

Spot detection and peak finding naturally produce coordinates. A first-class
`points` type should eventually map to a napari Points layer and a table. It is
not required for the first label/watershed milestone because maxima can
initially be emitted as a marker mask or marker-label image.

## P1: Implement Next

### P1A: Binary Masks And Labels

Implement the remaining nodes in the order shown by the planned statuses.

| Status | Node | Input -> Output | Suggested backend | Why it matters |
| --- | --- | --- | --- | --- |
| Implemented | Label Connected Components | mask -> labels | `scipy.ndimage.label` | Converts segmentation masks into distinct objects. |
| Implemented | Filter Labels By Volume | labels -> labels | NumPy label counts | Removes labels outside minimum and optional maximum pixel/voxel volume. |
| Implemented | Relabel Sequential | labels -> labels | `skimage.segmentation.relabel_sequential` | Normalizes sparse IDs after filtering. |
| Implemented | Clear Border Objects | mask/labels -> same semantic type | `skimage.segmentation.clear_border` | Removes partial objects touching image or ROI boundaries. |
| Implemented | Fill Holes | mask -> mask | `scipy.ndimage.binary_fill_holes` and connected-hole sizing | Fills all or size-limited holes using metadata-aware 2D/3D processing. |
| Implemented | Remove Small Objects | mask/labels -> same semantic type | SciPy connected components and NumPy label counts | Removes objects below a metadata-aware 2D/3D minimum size while preserving mask/label type. |
| Implemented | Euclidean Distance Transform | mask -> image | `scipy.ndimage.distance_transform_edt` | Foundation for separating touching objects and measuring thickness; metadata-aware 2D/3D processing. |
| Implemented | H-Maxima Markers | image -> labels | `skimage.morphology.h_maxima` / `local_maxima` | Produces robust labeled watershed seeds; `h=0` falls back to local maxima. |
| Implemented | Marker-Controlled Watershed | image + markers + mask -> labels | `skimage.segmentation.watershed` | Core method for separating touching nuclei, cells, and particles; defaults to inverted distance-map mode. |
| Implemented | Expand Labels | labels -> labels | `skimage.segmentation.expand_labels` | Approximates cell regions from nuclear seeds without label overlap; supports 2D slice-wise or 3D expansion. |
| Planned 5 | Find Label Boundaries | labels -> mask | `skimage.segmentation.find_boundaries` | Useful for QC, overlays, and boundary measurements. |

`Filter Labels By Volume` is also implemented as the label-preserving cleanup
operation. It supports both minimum and optional maximum size, a data-aware
logarithmic slider, and a volume-distribution inspector.

`Remove Small Objects` is the simpler cleanup node when only a minimum cutoff
is needed or when the input is still a mask. `Filter Labels By Volume` remains
the labels-only choice when a maximum cutoff or the object-volume distribution
is needed.

### What These Segmentation Terms Mean

#### Connected Components

Connected-component labeling takes a binary mask and assigns a distinct
positive integer to each spatially connected foreground structure:

```text
binary mask values: 0, 1
label image values: 0, 1, 2, 3, ...
```

`0` remains background. In 3D, connectivity is evaluated across the `Z`, `Y`,
and `X` spatial axes, so one nucleus spanning several z-slices receives one
label. In 2D, connectivity is evaluated across `Y` and `X`.

Connectivity must be configurable:

- 2D: edge-only neighbors (4-connectivity) or edge/corner neighbors
  (8-connectivity);
- 3D: face-only neighbors (6-connectivity), or progressively more permissive
  edge/corner connectivity (18 or 26).

This is the required first step for volume-based object cleanup because volumes
belong to objects, not to an undifferentiated boolean foreground.

#### Distance Transform

For each foreground pixel or voxel, a Euclidean distance transform reports the
distance to the nearest background. Values are small near object boundaries and
largest near object centers.

For a roughly round nucleus, the distance map resembles a hill whose summit is
near the nucleus center. For true 3D data, physical `Z/Y/X` spacing should be
used so distance is not distorted by thick z-slices.

Distance transforms are not required merely to label already-separated
objects. They become important when:

- touching nuclei or cells form one connected component;
- approximate local thickness is useful;
- watershed markers need to be generated from object interiors.

#### Marker Generation

Markers are seed labels identifying where individual objects are believed to
be. They are commonly generated from robust local maxima in a distance map.

For two touching nuclei:

1. thresholding may produce one merged binary component;
2. the distance map may contain two interior peaks;
3. marker generation converts those peaks into two seed labels;
4. watershed grows the two seeds through the merged mask until they meet;
5. the result is two labeled nuclei.

`H-Maxima` suppresses weak/shallow peaks and is generally more controllable than
using every raw local maximum, which often creates too many watershed objects.

Markers are therefore an optional splitting mechanism, not a prerequisite for
ordinary connected-component labeling.

### P1B: Background And Feature Enhancement

| Order | Node | Input -> Output | Suggested backend | Notes |
| --- | --- | --- | --- | --- |
| 1 | Rolling-Ball Background | image -> background image | `skimage.restoration.rolling_ball` | Implemented under Filtering -> Background Correction. Defaults to per-plane processing; warns that large 3D radii are expensive. |
| 2 | Subtract Background | image -> corrected image | `skimage.restoration.rolling_ball` + dtype-preserving subtraction | Implemented as a Fiji/ImageJ-style convenience node. Use Rolling-Ball Background when the estimated background itself should be inspected or saved. |
| 3 | Grayscale White Top-Hat | image -> image | `scipy.ndimage.white_tophat` | Faster approximate background suppression for bright objects. |
| 4 | Difference of Gaussians | image -> image | `scipy.ndimage.gaussian_filter`-based slice-wise DoG | Implemented as an `Edge & Detail` filtering node; enhances puncta and structures within a size band. |
| 5 | Laplacian of Gaussian | image -> image | `scipy.ndimage.gaussian_laplace` | Useful for blob enhancement and marker generation. |
| 6 | Unsharp Mask | image -> image | `skimage.filters.unsharp_mask` | Implemented as an `Edge & Detail` filtering node for general sharpening. |

Rolling-ball background estimation remains available separately from
subtraction. This exposes the estimated background and allows users to inspect,
save, or reuse it, while `Subtract Background` provides the common one-node
correction workflow.

The target behavior should match the practical Fiji/ImageJ use case: uneven
fluorescence background correction with an inspectable estimated background and
an auditable subtraction step. Large 3D rolling-ball or sliding-paraboloid
background estimation may need the manual/cached slow-node execution model
rather than live recomputation.

### P1C: Thresholding Without Node Explosion

Implemented named global threshold nodes:

- Otsu;
- Li;
- Yen;
- Isodata;
- Triangle;
- Minimum;
- fixed Binary threshold.

For now these remain separate graph nodes because the algorithm is visible on
the node card. Consolidate them into an `Automatic Threshold` node with a method
selector only if the named catalogue becomes harder to scan than explicit.

Implemented edge/threshold nodes inspired by MitoMorph:

- Hysteresis Threshold, based on the older MitoMorph
  `apply_hysteresis_threshold(stack, low, high)` workflow. It exposes raw
  low/high cutoffs, input-histogram markers, and metadata-aware 2D/3D spatial
  processing. It is grouped with global threshold segmentation nodes because it
  is a double-threshold mask operation.
- Canny Edges, matching the MitoMorph use of Canny as a slice-wise edge mask
  generator. It uses quantile thresholds by default so the node is less brittle
  across uint8, uint16, and normalized float images. It is grouped with
  `Filtering > Edge & Detail` alongside Sobel and Laplace-style edge operators.

Add `Multi-Otsu Classes` separately because its output is not a binary mask. It
should output an integer class image and optionally one mask output per class.

Sauvola and Niblack thresholding are now available under
`Segmentation > Local Thresholds` for uneven brightfield, histology-like, or
other locally varying images.

### P1D: Projection And Axis Reduction

Implemented:

- `Project Image`: contextual axis dropdown with automatic Z projection,
  explicit detected-axis choices, an all-non-YX-spatial shortcut, canonical
  internal axis values such as `auto`, `axis:2`, `name:z`, and
  `non_yx_spatial`, and common reducer methods: maximum, minimum, mean, sum,
  median, and standard deviation;
- `Orthogonal Projection`: XY/XZ/YZ montage projection for 3D volumes while
  preserving non-spatial axes such as time and channel;
- projection metadata updates so downstream nodes see the reduced or replaced
  axes.

Deferred polish:

- add a richer multi-axis picker if workflows need arbitrary axis combinations
  beyond the common single-axis and all-non-YX-spatial cases;
- optionally add a keep-reduced-axis mode if downstream workflows need
  singleton axes instead of removed axes.

### P1E: Object Measurements

Basic table support, the first label-only measurement node, named typed input
slots, the first intensity-aware measurement node, table merge, and metadata
annotation are implemented.

| Order | Node | Inputs -> Outputs | Suggested backend |
| --- | --- | --- | --- |
| Implemented | Measure Objects | labels -> table | `skimage.measure.regionprops_table` |
| Implemented | Measure Objects + Intensity | labels + image -> table | named heterogeneous input ports plus intensity statistics |
| Implemented | Merge Tables | tables -> table | stable object identity column joins; row-position fallback for equal-length tables |
| Implemented | Add Metadata Columns | table -> table | constant treatment, replicate, batch, or condition columns |
| Implemented | Filter Labels By Property | labels + table -> labels | table-derived label remapping |
| Implemented | Extended Region Properties | labels -> table | selectable `regionprops_table` property groups |
| Implemented | Derived Object Morphology | labels -> table | additional `Measure Objects` groups for derived ratios, 2D circularity, and Hu moments |
| Implemented | Select Table Columns | table -> table | detected-column checklist with Select all/Deselect all and output-order controls |
| Implemented | Summarize Measurements | table -> table | grouped NumPy statistics for treatment/PCA summaries |
| Implemented | Measure 3D Mesh Morphology | labels -> table, optional mesh later | `skimage` marching cubes, `skimage` mesh surface area, local mesh-volume helper, and `scipy.spatial.ConvexHull`; defer `trimesh` and `porespy` |
| Later | Save Table | table -> table | CSV/TSV writer |

The implemented first `Measure Objects` property set is intentionally small:

- label ID;
- pixel/voxel count and physical area/volume;
- centroid;
- bounding box;
- equivalent diameter;
- extent;
- Euler number.

Mean, minimum, maximum, sum, and standard deviation of intensity are implemented
in `Measure Objects + Intensity`, which accepts separate `Labels` and
`Intensity image` ports.

Selectable extended morphology groups are implemented on both object
measurement nodes. The current groups are:

- shape descriptors: bounding-box size and filled size for 2D/3D, plus convex
  area, solidity, and maximum Feret diameter for 2D;
- axis/inertia descriptors: major/minor axis length and inertia tensor
  eigenvalues for 2D/3D, plus eccentricity and orientation for 2D;
- 2D boundary descriptors: perimeter and Crofton perimeter, hidden for true 3D
  inputs in the inspector.
- derived shape ratios: major/minor axis ratio, bounding-box side lengths,
  bounding-box aspect ratios, fill fraction, and inertia eigenvalue ratios;
- 2D shape moments: Crofton-based circularity, perimeter-to-area ratio, and Hu
  moments, hidden for true 3D inputs in the inspector.

`Filter Labels By Property` is implemented as the table-driven cleanup step
after measurement. It keeps or removes labels using any numeric table column and
matches stable index columns such as `t_index` back to the corresponding
non-spatial label block.

`Measure 3D Mesh Morphology` is implemented as the first explicit 3D
surface-morphology table node. It measures voxel volume, mesh volume, mesh
surface area, surface-to-volume ratio, equivalent sphere radius/diameter,
sphericity, mesh extents/extent ratios, optional convex-hull area/volume,
3D solidity, and surface-area-to-hull-area ratio. The node is manual/cached,
uses carried Z/Y/X scale metadata, skips tiny labels below the minimum voxel
count with a row-level status, and records mesh or convex-hull failures as
`NaN` metrics instead of failing the whole table.

Calibrated physical variants for non-mesh extended length/shape columns remain
future work. The current extended non-mesh length and area descriptors are
explicitly labeled in pixels or voxels.

The concrete next morphology plan is tracked in
[object-mesh-morphology-plan.md](object-mesh-morphology-plan.md). The intended
split is: cheap derived morphology groups live on the existing measurement
nodes; first-pass mesh/surface morphology lives in the manual
`Measure 3D Mesh Morphology` node; next add calibrated physical variants for
remaining non-mesh length/shape columns and later optional mesh export/preview.
The first implementation stays within the existing `scikit-image`/`scipy`
stack; `trimesh` and `porespy` remain deferred optional dependencies.

The intended end state is broader than a single measurement node: users should
be able to compute selected morphology, intensity, mesh, and skeleton feature
families, merge them into one object-level table, annotate treatments or batch
metadata, and export the result for PCA or other statistical analysis. See
[mitomorph-feature-parity.md](mitomorph-feature-parity.md).

## Next Implementation Recommendation

Table outputs, basic object measurement, intensity measurement, table merge,
metadata annotation, grouped table summaries, and base skeleton-network
measurement are now implemented. Skeleton QC outputs and pruning are also
implemented. Manual/cached `Calculate`/`Recalculate` execution is implemented
for the first expensive table nodes. Branch-level skeleton measurements and
RGB graph overlays are implemented. Explicit skeleton graph node/edge export
and overall-network measurements are implemented, including branch-summary
distributions and normalized connectedness summaries. Next priorities are
mesh export/preview, calibrated non-mesh shape variants, and then
colocalization/localization tables.

## Recommended First Milestone

The first cohesive feature milestone should be split into a required label
cleanup path and an optional touching-object separation path.

### Milestone 1A: Labels And Volume Cleanup

Completed:

1. `labels` type and napari Labels inspection.
2. Label Connected Components.
3. Filter Labels By Volume.
4. Clear Border Objects.
5. Fill Holes.
6. Relabel Sequential.
7. Filter Labels By Property when a measurement table is available.

Remaining:

1. Explicit kept/removed object counts.

The primary pipeline is:

```text
Image
  -> threshold / segmentation mask
  -> binary mask cleanup
  -> Label Connected Components
  -> Filter Labels By Volume
  -> Relabel Sequential
  -> cleaned Labels
```

`Filter Labels By Volume` should:

- accept integer labels without merging neighboring labels: implemented;
- support minimum and optional maximum volume: implemented;
- offer voxel/pixel count immediately: implemented;
- offer calibrated physical area/volume when trustworthy scale and units exist;
  implemented through `Measure Objects` plus `Filter Labels By Property`;
- preserve retained label IDs until an explicit `Relabel Sequential` node:
  implemented;
- report how many labels were kept and removed;
- operate over full `ZYX` volumes for 3D data and `YX` for 2D data:
  implemented;
- expand to filtering by morphological and intensity properties: implemented
  through measurement tables and `Filter Labels By Property`.

The inspector already reports the incoming object count, median, and largest
volume and displays the input volume distribution with live thresholds. It does
not yet report kept and removed counts separately.

`Remove Small Objects` covers minimum-size cleanup for masks and labels.
`Filter Labels By Volume` remains separate because it also supports a maximum
cutoff and an object-volume distribution for choosing thresholds.

### Milestone 1B: Split Touching Objects

Implemented:

1. Euclidean Distance Transform.
2. H-Maxima Markers.
3. Marker-Controlled Watershed.
4. Expand Labels.

This optional branch is inserted when thresholded structures touch:

```text
binary mask
  -> Distance transform
  -> H-maxima markers
  -> Watershed
  -> Filter Labels By Volume
  -> cleaned Labels
```

The watershed branch is now technically unblocked because named heterogeneous
input ports are implemented and the branch itself is available in the node
palette. Remaining polish is marker QC visualization, better defaults based on
real microscopy examples, and optional marker/label validation summaries.

## P2: Add After Core Object Analysis

### Denoising And Restoration

| Node | Suggested backend | Reason for P2 |
| --- | --- | --- |
| Total Variation Denoise | `skimage.restoration.denoise_tv_chambolle` | Useful nD denoising, but parameter meaning needs clear documentation. |
| Non-Local Means Denoise | `skimage.restoration.denoise_nl_means` | Powerful but computationally expensive and parameter-heavy. |
| Wavelet Denoise | `skimage.restoration.denoise_wavelet` | Useful for fluorescence noise, but dtype and channel semantics need care. |
| Estimate Noise Sigma | `skimage.restoration.estimate_sigma` | Helps configure denoisers and can output a scalar/table value. |
| Richardson-Lucy Deconvolution | `skimage.restoration.richardson_lucy` | Important for microscopy, but requires a PSF input or a defensible PSF generator. |
| Wiener Deconvolution | `skimage.restoration.wiener` | Useful once PSF handling is established. |

Deconvolution should not be shipped as a single sigma slider. The user must be
able to inspect or supply the point-spread function, understand iteration count,
and preserve physical spacing. Deconvolution should use the manual/cached
slow-node execution model because iteration count, PSF size, and stack size can
make live recomputation impractical.

### Registration And Drift Correction

| Node | Suggested backend | Notes |
| --- | --- | --- |
| Estimate Translation | `skimage.registration.phase_cross_correlation` | Two images in, transform/shift result out. |
| Apply Translation | `scipy.ndimage.shift` | Interpolation must depend on image versus labels. |
| Register Stack To Reference | repeated phase cross-correlation | Useful for time-lapse drift correction. |
| Affine Transform | `scipy.ndimage.affine_transform` or `skimage.transform.warp` | Requires a first-class transform representation. |

Registration is high-value, but a transform type and interpolation policy should
be designed before exposing several registration algorithms.

### Geometry And Sampling

Implemented:

- `Set Pixel Size / Units`: metadata-only calibration repair for X/Y pixel size,
  optional Z step size, and physical units;
- `Rescale Axes`: pixel-grid resampling by X/Y/Z scale factors, optional X/Y
  aspect-ratio locking, nearest/linear/cubic/spline interpolation, intensity
  downsampling anti-aliasing, nearest-neighbor preservation for masks/labels,
  and inverse physical-scale metadata updates;
- `Orthogonal Projection` uses calibrated Z/Y/X spacing for physically scaled
  XY/XZ/YZ montages when spacing metadata is available.

Deferred:

- resample to target pixel/voxel spacing;
- pad/crop to a target shape;
- flip and rotate by right angles;
- arbitrary rotation;
- isotropic resampling for 3D visualization or analysis.

These operations must update scale, translation, and axes correctly. Label
inputs require nearest-neighbor interpolation.

### Shape And Structure

- implemented: skeletonize / skeletonize 3D;
- implemented: Analyze Skeleton table for per-component skeleton voxel count,
  endpoint voxels, junction voxels, isolated nodes, graph node/edge counts,
  voxel-graph edge count, cycle count, connected-component context, and
  calibrated length;
- implemented: Skeleton Keypoints endpoint/junction/isolated-node masks;
- implemented: Skeleton Graph Overlay for RGB edge/node QC visualization;
- implemented: Label Skeleton Components and Label Skeleton Branches;
- implemented: Prune Skeleton Branches for short terminal spurs and isolated
  skeleton voxels;
- implemented: Measure Skeleton Branches for row-per-branch length,
  endpoint-distance, tortuosity, start/end coordinates, and calibrated physical
  length when metadata is available;
- implemented: Skeleton Graph Tables for explicit graph node and graph edge
  table export;
- implemented: Summarize Skeleton Branches for branch-length/tortuosity
  distributions and branch-type count/fraction summaries;
- implemented: Measure Overall Skeleton Network for per-block connectedness,
  fragmentation, branch-count, graph-edge, branch-length, and normalized
  connectedness whole-network metrics;
- implemented: physical-unit pruning threshold support in Prune Skeleton
  Branches when pixel-size metadata is available;
- medial axis;
- grayscale erosion, dilation, opening, and closing;
- label erosion and label-safe expansion;
- convex hull per object;
- object boundary distance or thickness.

### Domain-Specific Filters

- Frangi vesselness;
- Sato tubeness;
- Meijering neuriteness;
- Hessian-based ridge filters;
- Gabor filters.

These are valuable for vessels, fibers, and neurites, but should follow a clear
target workflow because their scale parameters and polarity options are easy to
misuse.

### Spot And Blob Detection

- Blob LoG;
- Blob DoG;
- peak local maximum;
- spot intensity measurement;
- count spots per labeled object.

These become much cleaner after `points` and `table` output types exist.

### Colocalization And Segmentation Quality

- Pearson correlation within an optional mask;
- Manders overlap coefficients;
- object overlap / intersection-over-union;
- Dice coefficient;
- adapted Rand error;
- variation of information;
- contingency table between label images.

These should produce scalar or table results, not synthetic images.
Pixel-based colocalization, object-based association, nearest-neighbor
localization, and event localization should remain distinct enough that users
can tell whether they are measuring intensity overlap, object overlap, object
distance, or time-lapse event position. Expensive variants over large 3D/time
series data are candidates for manual/cached execution with stale-state
feedback.

## P3: Specialized Or Expensive

- random walker segmentation;
- active contours and morphological snakes;
- graph-cut or graph-based segmentation;
- superpixels such as SLIC;
- optical flow;
- non-rigid registration;
- image stitching and mosaics;
- object tracking across time;
- 3D surface/mesh extraction and measurement;
- frequency-domain notch filtering;
- blind deconvolution;
- learned denoisers;
- Cellpose, StarDist, ilastik, or other model-backed segmentation.

Model-backed nodes should be optional integrations with isolated dependencies,
model provenance, device selection, and reproducible model/version metadata.
They should not make the core plugin installation heavy.

## Defer For Now

The following are common in general computer-vision libraries but are not good
early priorities for a bioimage workflow composer:

- many separate edge operators beyond the implemented Sobel and Canny nodes,
  such as Scharr, Prewitt, and Roberts;
- Hough line and circle transforms;
- contour hierarchy operations;
- polygon approximation and rotated boxes;
- ORB/SIFT-like keypoints and descriptors;
- template matching;
- face/object detection APIs;
- camera calibration and video-stream processing;
- broad color-space conversion catalogues;
- artistic transforms, image pyramids, and inpainting;
- duplicate OpenCV versions of operations already provided well by SciPy or
  scikit-image.

Keep additional edge detectors selective. Sobel and Canny are now available for
QC, watershed support, and MitoMorph parity, but implementing every edge
detector would add palette noise without completing a bioimage workflow.

Slice-wise stack processing is now a required UX disclosure. Present and future
nodes that process only the current `YX` plane on stacks should set
`OperationSpec.stack_processing_note`, so the inspector warns users to use
`Reorder Axes` first when another plane or slice axis is intended.

## Suggested Palette Structure

```text
Image Data
  Source & Output
  Axes & Regions
  Channels & Composites
  Utilities
  Math & Logic

Intensity & Contrast

Filtering
  Background Correction
  Smoothing & Denoising
  Edge & Detail

Segmentation
  Global Thresholds
  Local Thresholds
  Object Separation

Binary Morphology
Label Operations
Measurements
Registration
Restoration
```

`Filtering` can remain during migration, but splitting enhancement by intent
will become easier to navigate than one long filter list.

## Example Target Workflows

### Fluorescent Nuclei

```text
Extract channel
  -> Rolling-ball background
  -> Gaussian blur
  -> Automatic threshold
  -> Remove small objects
  -> Fill Holes, optionally limited by hole area/volume
  -> Distance transform
  -> H-maxima markers
  -> Watershed
  -> Measure objects
```

### Cell Regions From Nuclear Seeds

```text
Nuclear labels
  -> Expand labels
  -> Optional membrane/intensity mask constraint
  -> Measure per-cell channel intensity
```

### Fluorescent Puncta

```text
Extract channel
  -> Background correction
  -> Difference of Gaussians or Laplacian of Gaussian
  -> Peak/blob detection
  -> Points/table
  -> Count spots per cell label
```

### Mitochondrial Morphology

Mitochondria can be analyzed either as separate organelles or as a connected
network. Connected-component labeling and volume filtering are useful for
removing debris and measuring fragmented organelles, but a connected network
may intentionally form one large label.

```text
Extract mitochondrial channel
  -> background correction
  -> ridge/tubeness enhancement or denoising
  -> threshold
  -> binary cleanup
  -> labels for fragment analysis
  -> skeleton/network analysis for branches and connectivity
```

This means future mitochondrial support needs both object measurements and
skeleton/graph measurements; volume filtering alone is not sufficient. The
generic `Skeletonize` and `Analyze Skeleton` nodes now cover the first shared
network metrics, including graph edges, isolate counts, and cycle counts in 3D.
Future mitochondrial-specific work should add validated fragmentation,
domain-normalized connectedness, branch-distribution summaries, and
mesh/surface metrics only where the biological assumptions are explicit.

The old MitoMorph implementation used `regionprops`, mesh-like surface/volume
estimates, skeletonization, and graph analysis. Treat that as inspiration for a
future dedicated mitochondrial measurement family, not as a direct API to copy
into the generic `Measure Objects` node.
The higher-level parity target is documented in
[mitomorph-feature-parity.md](mitomorph-feature-parity.md).

### Colocalization

Pixel-based and object-based colocalization should remain distinct:

```text
Pixel-based:
channel A + channel B + optional ROI mask
  -> Pearson / Manders metrics
  -> result table

Object-based:
labels A + labels B
  -> overlap, nearest-neighbor distance, or object association
  -> result table and optional matched labels
```

Colocalization therefore follows table outputs and benefits from labels, but
does not require registration in its first version when channels are already
spatially aligned.

### Time-Lapse Drift Correction

```text
Select reference frame
  -> Estimate translation per timepoint
  -> Apply translation
  -> Preserve T/C/Z/Y/X metadata
```

## Decisions And Open Discussion

Decided:

1. Label connectivity exposes both face and full connectivity; full
   connectivity is the current default.
2. Object labels are created independently over leading non-spatial axes, while
   a recognized `ZYX` block is treated as one volume.
3. Registration and deconvolution remain later milestones.
4. The initial workflow priority covers nuclei/cells, puncta, mitochondria, and
   colocalization rather than only one of those domains.

Still open:

1. Should `Filter Labels By Volume` default to calibrated units whenever scale
   metadata exists, or keep pixels/voxels as the default with physical units as
   an explicit mode?
2. How should mixed or incompatible spatial units be handled?
3. Which mitochondrial workflow should come first after generic measurements:
   fragmented-organelle object measurements or connected-network skeleton
   analysis?
4. Should `Automatic Threshold` consolidate existing threshold nodes in the
   palette, or do named nodes make workflows easier to scan?

## Library Direction

The current environment has SciPy and scikit-image but not OpenCV. That is a
reasonable default:

- SciPy provides efficient multidimensional filtering, morphology, distance
  transforms, interpolation, and labeling.
- scikit-image provides bioimage-relevant segmentation, labels, region
  measurements, restoration, feature detection, and registration.
- OpenCV is strongest for broad 2D computer vision, contours, real-time/video
  workflows, and highly optimized implementations. Those strengths do not yet
  justify another required dependency for VIPP.

OpenCV can be reconsidered if profiling shows a meaningful performance gap or a
selected node has no adequate SciPy/scikit-image implementation.

## Primary References

- [scikit-image segmentation API](https://scikit-image.org/docs/stable/api/skimage.segmentation.html)
- [scikit-image morphology API](https://scikit-image.org/docs/stable/api/skimage.morphology.html)
- [scikit-image measurement API](https://scikit-image.org/docs/stable/api/skimage.measure.html)
- [scikit-image filters API](https://scikit-image.org/docs/stable/api/skimage.filters.html)
- [scikit-image restoration API](https://scikit-image.org/docs/stable/api/skimage.restoration.html)
- [scikit-image feature API](https://scikit-image.org/docs/stable/api/skimage.feature.html)
- [scikit-image registration API](https://scikit-image.org/docs/stable/api/skimage.registration.html)
- [scikit-image transform API](https://scikit-image.org/docs/stable/api/skimage.transform.html)
- [SciPy multidimensional image processing API](https://docs.scipy.org/doc/scipy/reference/ndimage.html)
- [OpenCV image-processing overview](https://docs.opencv.org/4.x/d7/da8/tutorial_table_of_content_imgproc.html)
